import numpy as np
import tensorflow as tf
import os
import time
import pandas as pd
from pathlib import Path
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv1D, Flatten, Dense, Conv1DTranspose, Reshape, Input, Layer, LSTM
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean

# Set random seeds for reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.keras.utils.set_random_seed(123)
tf.keras.backend.set_floatx('float32')

# -------------------------------------------------
# TimeVAE Model Definition (Complete with Synthetic Generation)
# -------------------------------------------------

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def temporal_consistency_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff_true = y_true[:, 1:] - y_true[:, :-1]
    diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
    return tf.reduce_mean(tf.square(diff_true - diff_pred))

class BaseVariationalAutoencoder(Model):
    model_name = None
    def __init__(
        self,
        seq_len,
        feat_dim,
        cond_dim,
        latent_dim,
        reconstruction_wt=3.0,
        dtw_wt=1.0,
        beta=2.0,
        forecast_wt=1.0,
        batch_size=16,
        **kwargs,
    ):
        super(BaseVariationalAutoencoder, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.dtw_wt = dtw_wt
        self.beta = beta
        self.forecast_wt = forecast_wt
        self.batch_size = batch_size

        # Metrics
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.dtw_loss_tracker = Mean(name="dtw_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.forecast_loss_tracker = Mean(name="forecast_loss")
        self.val_reconstruction_loss_tracker = Mean(name="val_reconstruction_loss")
        self.val_dtw_loss_tracker = Mean(name="val_dtw_loss")
        self.val_kl_loss_tracker = Mean(name="val_kl_loss")
        self.val_forecast_loss_tracker = Mean(name="val_forecast_loss")
        self.val_total_loss_tracker = Mean(name="val_total_loss")

        self.compile(optimizer=Adam(learning_rate=1e-3))

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.dtw_loss_tracker,
            self.kl_loss_tracker,
            self.forecast_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_dtw_loss_tracker,
            self.val_kl_loss_tracker,
            self.val_forecast_loss_tracker,
        ]

    def _get_reconstruction_loss(self, X, X_recons):
        X = tf.cast(X, tf.float32)
        X_recons = tf.cast(X_recons, tf.float32)
        err = tf.math.squared_difference(X, X_recons)
        return tf.reduce_mean(err)

    def _get_forecast_loss(self, x, x_forecast, next_x):
        x_forecast = tf.cast(x_forecast, tf.float32)
        next_x = tf.cast(next_x, tf.float32)
        err = tf.math.squared_difference(next_x, x_forecast)
        return tf.reduce_mean(err)

class TrendLayer(Layer):
    def __init__(self, feat_dim, trend_poly, seq_len, **kwargs):
        super(TrendLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.trend_poly = trend_poly
        self.seq_len = seq_len
        self.trend_dense1 = Dense(self.feat_dim * self.trend_poly, activation="relu")
        self.trend_dense2 = Dense(self.feat_dim * self.trend_poly)
        self.reshape_layer = Reshape(target_shape=(self.feat_dim, self.trend_poly))

    def call(self, z):
        trend_params = self.trend_dense1(z)
        trend_params = self.trend_dense2(trend_params)
        trend_params = self.reshape_layer(trend_params)
        lin_space = tf.range(0, float(self.seq_len), 1) / self.seq_len
        poly_space = tf.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], axis=0)
        trend_vals = tf.matmul(trend_params, poly_space)
        trend_vals = tf.transpose(trend_vals, perm=[0, 2, 1])
        return tf.cast(trend_vals, tf.float32)

class SeasonalLayer(Layer):
    def __init__(self, feat_dim, seq_len, custom_seas, **kwargs):
        super(SeasonalLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.custom_seas = custom_seas
        self.dense_layers = [
            Dense(feat_dim * num_seasons) for num_seasons, _ in custom_seas
        ]
        self.reshape_layers = [
            Reshape(target_shape=(feat_dim, num_seasons))
            for num_seasons, _ in custom_seas
        ]

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = tf.range(num_seasons)[:, None] + tf.zeros(
            (num_seasons, len_per_season), dtype=tf.int32
        )
        season_indexes = tf.reshape(season_indexes, [-1])
        season_indexes = tf.tile(season_indexes, [self.seq_len // len_per_season + 1])[: self.seq_len]
        return season_indexes

    def call(self, z):
        N = tf.shape(z)[0]
        ones_tensor = tf.ones(shape=[N, self.feat_dim, self.seq_len], dtype=tf.int32)
        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)
            season_params = self.reshape_layers[i](season_params)
            season_indexes_over_time = self._get_season_indexes_over_seq(num_seasons, len_per_season)
            dim2_idxes = ones_tensor * tf.reshape(season_indexes_over_time, shape=(1, 1, -1))
            season_vals = tf.gather(season_params, dim2_idxes, batch_dims=-1)
            all_seas_vals.append(season_vals)
        all_seas_vals = K.stack(all_seas_vals, axis=-1)
        all_seas_vals = tf.reduce_sum(all_seas_vals, axis=-1)
        all_seas_vals = tf.transpose(all_seas_vals, perm=[0, 2, 1])
        return all_seas_vals

class TimeVAE(BaseVariationalAutoencoder):
    model_name = "ConditionalTimeVAE"
    def __init__(
        self,
        hidden_layer_sizes=None,
        trend_poly=2,
        custom_seas=[(24, 1), (7, 24), (30, 24)],
        use_residual_conn=True,
        forecast_horizon=24,
        trend_dim=10,
        seasonal_dim=10,
        noise_dim=12,
        forecast_wt=1.0,
        **kwargs,
    ):
        self.trend_dim = trend_dim
        self.seasonal_dim = seasonal_dim
        self.noise_dim = noise_dim
        self.use_residual_conn = use_residual_conn
        if 'latent_dim' in kwargs and kwargs['latent_dim'] != trend_dim + seasonal_dim + noise_dim:
            raise ValueError("latent_dim must equal trend_dim + seasonal_dim + noise_dim")
        super(TimeVAE, self).__init__(forecast_wt=forecast_wt, **kwargs)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 128, 256]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.forecast_horizon = forecast_horizon

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.forecaster = self._get_forecaster()
        self.compile(optimizer=Adam(learning_rate=1e-3), weighted_metrics=[])

    def _get_encoder(self):
        encoder_inputs = Input(shape=(self.seq_len, self.feat_dim), name="encoder_input")
        if self.cond_dim > 0:
            cond_inputs = Input(shape=(self.seq_len, self.cond_dim), name="cond_input")
            x = tf.concat([encoder_inputs, cond_inputs], axis=-1)
        else:
            cond_inputs = None
            x = encoder_inputs

        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D(filters=num_filters, kernel_size=3, strides=1,
                       dilation_rate=2**i, activation="relu",
                       padding="same", name=f"enc_conv_{i}")(x)

        x = Flatten(name="enc_flatten")(x)
        self.encoder_last_dense_dim = x.shape[-1]

        z_trend_mean = Dense(self.trend_dim)(x)
        z_trend_log_var = Dense(self.trend_dim)(x)
        z_seasonal_mean = Dense(self.seasonal_dim)(x)
        z_seasonal_log_var = Dense(self.seasonal_dim)(x)
        z_noise_mean = Dense(self.noise_dim)(x)
        z_noise_log_var = Dense(self.noise_dim)(x)

        z_mean = tf.concat([z_trend_mean, z_seasonal_mean, z_noise_mean], axis=-1)
        z_log_var = tf.concat([z_trend_log_var, z_seasonal_log_var, z_noise_log_var], axis=-1)
        z = Sampling()([z_mean, z_log_var])

        return Model([encoder_inputs, cond_inputs], [z_mean, z_log_var, z], name="encoder") if self.cond_dim > 0 else Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def _get_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dim,), name="decoder_input")
        cond_inputs = Input(shape=(self.seq_len, self.cond_dim), name="cond_input") if self.cond_dim > 0 else None

        z_trend = decoder_inputs[:, :self.trend_dim]
        z_seasonal = decoder_inputs[:, self.trend_dim:self.trend_dim + self.seasonal_dim]
        z_noise = decoder_inputs[:, self.trend_dim + self.seasonal_dim:]
        z_combined = tf.concat([z_trend, z_seasonal, z_noise], axis=-1)

        enc_feat_dim = self.hidden_layer_sizes[-1]
        x = Dense(self.seq_len * enc_feat_dim, activation="relu")(z_combined)
        x = Reshape((self.seq_len, enc_feat_dim))(x)

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(filters=num_filters, kernel_size=5,
                                strides=1, padding="same", activation="relu")(x)

        conv_out = Conv1DTranspose(filters=self.feat_dim, kernel_size=3,
                                   strides=1, padding="same", activation="relu")(x)

        outputs = conv_out
        if self.use_residual_conn:
            outputs = outputs + x

        if self.trend_poly > 0:
            outputs += TrendLayer(self.feat_dim, self.trend_poly, self.seq_len)(z_trend)
        if self.custom_seas:
            outputs += SeasonalLayer(self.feat_dim, self.seq_len, self.custom_seas)(z_seasonal)

        if self.cond_dim > 0:
            outputs = tf.concat([outputs, cond_inputs], axis=-1)

        outputs = Dense(self.feat_dim, activation="softplus")(outputs)

        return Model([decoder_inputs, cond_inputs], outputs, name="decoder") if self.cond_dim > 0 else Model(decoder_inputs, outputs, name="decoder")

    def _get_forecaster(self):
        z_inputs = Input(shape=(self.latent_dim,), name="forecast_z_input")
        cond_inputs = Input(shape=(self.seq_len, self.cond_dim), name="forecast_cond_input") if self.cond_dim > 0 else None

        z_trend = z_inputs[:, :self.trend_dim]
        z_seasonal = z_inputs[:, self.trend_dim:self.trend_dim + self.seasonal_dim]
        z_noise = z_inputs[:, self.trend_dim + self.seasonal_dim:]
        z_combined = tf.concat([z_trend, z_seasonal, z_noise], axis=-1)

        enc_feat_dim = self.hidden_layer_sizes[-1]
        x = Dense(self.seq_len * enc_feat_dim, activation="relu")(z_combined)
        x = Reshape((self.seq_len, enc_feat_dim))(x)

        if self.cond_dim > 0:
            cond_proj = Dense(enc_feat_dim, activation="relu")(cond_inputs)
            if self.use_residual_conn:
                x = x + cond_proj
            else:
                x = tf.concat([x, cond_proj], axis=-1)

        x = LSTM(128, return_sequences=False)(x)
         
        x = Dense(self.forecast_horizon * self.feat_dim, activation="relu")(x)
        x = Reshape((self.forecast_horizon, self.feat_dim))(x)
        outputs = Dense(self.feat_dim, activation="softplus")(x)

        return Model([z_inputs, cond_inputs], outputs, name="forecaster") if self.cond_dim > 0 else Model(z_inputs, outputs, name="forecaster")

    def train_step(self, data):
        if isinstance(data, (list, tuple)) and len(data) == 3:
            x, cond, next_x = data
        else:
            x, next_x = data
            cond = None
        with tf.GradientTape() as tape:
            if self.cond_dim > 0:
                z_mean, z_log_var, z = self.encoder([x, cond])
                reconstruction = self.decoder([z, cond])
                forecast = self.forecaster([z, cond])
            else:
                z_mean, z_log_var, z = self.encoder(x)
                reconstruction = self.decoder(z)
                forecast = self.forecaster(z)
            reconstruction_loss = self._get_reconstruction_loss(x, reconstruction)
            temporal_loss = temporal_consistency_loss(x, reconstruction)
            forecast_loss = self._get_forecast_loss(x, forecast, next_x)
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            total_loss = (
                self.reconstruction_wt * reconstruction_loss +
                self.dtw_wt * temporal_loss +
                self.beta * kl_loss +
                self.forecast_wt * forecast_loss
            )
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.dtw_loss_tracker.update_state(temporal_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.forecast_loss_tracker.update_state(forecast_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "dtw_loss": self.dtw_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "forecast_loss": self.forecast_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, (list, tuple)) and len(data) == 3:
            x, cond, next_x = data
        else:
            x, next_x = data
            cond = None
        if self.cond_dim > 0:
            z_mean, z_log_var, z = self.encoder([x, cond])
            reconstruction = self.decoder([z, cond])
            forecast = self.forecaster([z, cond])
        else:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            forecast = self.forecaster(z)
        val_reconstruction_loss = self._get_reconstruction_loss(x, reconstruction)
        val_temporal_loss = temporal_consistency_loss(x, reconstruction)
        val_forecast_loss = self._get_forecast_loss(x, forecast, next_x)
        val_kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        val_total_loss = (
            self.reconstruction_wt * val_reconstruction_loss +
            self.dtw_wt * val_temporal_loss +
            self.beta * val_kl_loss +
            self.forecast_wt * val_forecast_loss
        )
        self.val_total_loss_tracker.update_state(val_total_loss)
        self.val_reconstruction_loss_tracker.update_state(val_reconstruction_loss)
        self.val_dtw_loss_tracker.update_state(val_temporal_loss)
        self.val_kl_loss_tracker.update_state(val_kl_loss)
        self.val_forecast_loss_tracker.update_state(val_forecast_loss)
        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "dtw_loss": self.val_dtw_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
            "forecast_loss": self.val_forecast_loss_tracker.result(),
        }

    # -------------------------------------------------
    # SYNTHETIC DATA GENERATION METHODS
    # -------------------------------------------------
    
    def get_prior_samples(self, num_samples, cond=None):
        """Generate synthetic samples from prior distribution"""
        z = np.random.normal(0, 1, (num_samples, self.latent_dim))
        
        if self.cond_dim > 0 and cond is not None:
            if cond.shape[0] != num_samples:
                cond = np.tile(cond, (num_samples, 1, 1))
            samples = self.decoder.predict([z, cond], verbose=0)
        else:
            samples = self.decoder.predict(z, verbose=0)
        
        return samples
    
    def generate_from_learned_distribution(self, real_x, real_cond=None, num_samples=100):
        """Generate samples using statistics learned from real data"""
        if self.cond_dim > 0 and real_cond is not None:
            z_mean, z_log_var, _ = self.encoder.predict([real_x, real_cond], verbose=0)
        else:
            z_mean, z_log_var, _ = self.encoder.predict(real_x, verbose=0)
        
        empirical_mean = np.mean(z_mean, axis=0)
        empirical_std = np.sqrt(np.mean(np.exp(z_log_var), axis=0))
        
        z_samples = np.random.normal(empirical_mean, empirical_std, (num_samples, self.latent_dim))
        
        if self.cond_dim > 0:
            if real_cond is not None:
                if real_cond.shape[0] >= num_samples:
                    selected_cond = real_cond[:num_samples]
                else:
                    repeats = (num_samples // real_cond.shape[0]) + 1
                    selected_cond = np.tile(real_cond, (repeats, 1, 1))[:num_samples]
            else:
                selected_cond = np.random.normal(0, 1, (num_samples, self.seq_len, self.cond_dim))
            
            samples = self.decoder.predict([z_samples, selected_cond], verbose=0)
        else:
            samples = self.decoder.predict(z_samples, verbose=0)
        
        return samples
    
    def generate_conditional_samples(self, cond_features, num_samples=100):
        """Generate samples conditioned on specific features"""
        if self.cond_dim == 0:
            raise ValueError("Model was not trained with conditional features")
            
        z = np.random.normal(0, 1, (num_samples, self.latent_dim))
        cond_batch = np.tile(cond_features[np.newaxis, :, :], (num_samples, 1, 1))
        samples = self.decoder.predict([z, cond_batch], verbose=0)
        return samples
    
    def interpolate_samples(self, start_data, end_data, start_cond=None, end_cond=None, num_steps=10):
        """Generate interpolated samples between two data points"""
        if self.cond_dim > 0 and start_cond is not None and end_cond is not None:
            z_start, _, _ = self.encoder.predict([start_data[np.newaxis, :, :], 
                                                start_cond[np.newaxis, :, :]], verbose=0)
            z_end, _, _ = self.encoder.predict([end_data[np.newaxis, :, :], 
                                             end_cond[np.newaxis, :, :]], verbose=0)
        else:
            z_start, _, _ = self.encoder.predict(start_data[np.newaxis, :, :], verbose=0)
            z_end, _, _ = self.encoder.predict(end_data[np.newaxis, :, :], verbose=0)
        
        alphas = np.linspace(0, 1, num_steps)
        interpolated_z = []
        interpolated_cond = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * z_start + alpha * z_end
            interpolated_z.append(z_interp[0])
            
            if self.cond_dim > 0 and start_cond is not None and end_cond is not None:
                cond_interp = (1 - alpha) * start_cond + alpha * end_cond
                interpolated_cond.append(cond_interp)
        
        interpolated_z = np.array(interpolated_z)
        
        if self.cond_dim > 0 and len(interpolated_cond) > 0:
            interpolated_cond = np.array(interpolated_cond)
            samples = self.decoder.predict([interpolated_z, interpolated_cond], verbose=0)
        else:
            samples = self.decoder.predict(interpolated_z, verbose=0)
        
        return samples
    
    def generate_with_noise_control(self, num_samples, cond=None, 
                                  trend_scale=1.0, seasonal_scale=1.0, noise_scale=1.0):
        """Generate samples with controlled noise in different latent components"""
        z_trend = np.random.normal(0, trend_scale, (num_samples, self.trend_dim))
        z_seasonal = np.random.normal(0, seasonal_scale, (num_samples, self.seasonal_dim))
        z_noise = np.random.normal(0, noise_scale, (num_samples, self.noise_dim))
        
        z = np.concatenate([z_trend, z_seasonal, z_noise], axis=1)
        
        if self.cond_dim > 0 and cond is not None:
            if cond.shape[0] != num_samples:
                cond = np.tile(cond, (num_samples, 1, 1))
            samples = self.decoder.predict([z, cond], verbose=0)
        else:
            samples = self.decoder.predict(z, verbose=0)
        
        return samples