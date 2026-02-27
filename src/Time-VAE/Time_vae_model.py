import os, warnings, sys
from re import T

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from abc import ABC, abstractmethod
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv1D,
    Flatten,
    Dense,
    Conv1DTranspose,
    Reshape,
    Input,
    Layer,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.backend import random_normal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================================
# BASE CLASS
# ============================================================================

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class BaseVariationalAutoencoder(Model, ABC):
    model_name = None

    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        reconstruction_wt=3.0,
        batch_size=16,
        **kwargs,
    ):
        super(BaseVariationalAutoencoder, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.encoder = None
        self.decoder = None

    def fit_on_data(self, train_data, max_epochs=1000, verbose=0):
        loss_to_monitor = "loss"
        early_stopping = EarlyStopping(
            monitor=loss_to_monitor, min_delta=1e-2, patience=50, mode="min"
        )
        reduce_lr = ReduceLROnPlateau(
            monitor=loss_to_monitor, factor=0.5, patience=30, mode="min"
        )
        self.fit(
            train_data,
            epochs=max_epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose,
        )

    def call(self, X):
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1:
            x_decoded = x_decoded.reshape((1, -1))
        return x_decoded

    def get_num_trainable_variables(self):
        trainableParams = int(
            np.sum([np.prod(v.get_shape()) for v in self.trainable_weights])
        )
        nonTrainableParams = int(
            np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights])
        )
        totalParams = trainableParams + nonTrainableParams
        return trainableParams, nonTrainableParams, totalParams

    def get_prior_samples(self, num_samples):
        Z = np.random.randn(num_samples, self.latent_dim)
        samples = self.decoder.predict(Z, verbose=0)
        return samples

    def get_prior_samples_given_Z(self, Z):
        samples = self.decoder.predict(Z)
        return samples

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_c, axis):
            x_r = tf.reduce_mean(X, axis=axis)
            x_c_r = tf.reduce_mean(X_recons, axis=axis)
            err = tf.math.squared_difference(x_r, x_c_r)
            loss = tf.reduce_sum(err)
            return loss

        # overall
        err = tf.math.squared_difference(X, X_recons)
        reconst_loss = tf.reduce_sum(err)

        reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[2])  # by time axis
        return reconst_loss

    def train_step(self, X):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X)

            reconstruction = self.decoder(z)

            reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))

            total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        reconstruction = self.decoder(z)
        reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))

        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def save_weights(self, model_dir):
        if self.model_name is None:
            raise ValueError("Model name not set.")
        encoder_wts = self.encoder.get_weights()
        decoder_wts = self.decoder.get_weights()
        joblib.dump(
            encoder_wts, os.path.join(model_dir, f"{self.model_name}_encoder_wts.h5")
        )
        joblib.dump(
            decoder_wts, os.path.join(model_dir, f"{self.model_name}_decoder_wts.h5")
        )

    def load_weights(self, model_dir):
        encoder_wts = joblib.load(
            os.path.join(model_dir, f"{self.model_name}_encoder_wts.h5")
        )
        decoder_wts = joblib.load(
            os.path.join(model_dir, f"{self.model_name}_decoder_wts.h5")
        )

        self.encoder.set_weights(encoder_wts)
        self.decoder.set_weights(decoder_wts)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.save_weights(model_dir)
        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)


# ============================================================================
# TIMEVAE HELPER LAYERS
# ============================================================================

class TrendLayer(Layer):
    def __init__(self, feat_dim, trend_poly, seq_len, **kwargs):
        super(TrendLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.trend_poly = trend_poly
        self.seq_len = seq_len
        self.trend_dense1 = Dense(
            self.feat_dim * self.trend_poly, activation="relu", name="trend_params"
        )
        self.trend_dense2 = Dense(self.feat_dim * self.trend_poly, name="trend_params2")
        self.reshape_layer = Reshape(target_shape=(self.feat_dim, self.trend_poly))

    def call(self, z):
        trend_params = self.trend_dense1(z)
        trend_params = self.trend_dense2(trend_params)
        trend_params = self.reshape_layer(trend_params)  # shape: N x D x P

        lin_space = (
            tf.range(0, float(self.seq_len), 1) / self.seq_len
        )
        poly_space = tf.stack(
            [lin_space ** float(p + 1) for p in range(self.trend_poly)], axis=0
        )  # shape: P x T

        trend_vals = tf.matmul(trend_params, poly_space)  # shape (N, D, T)
        trend_vals = tf.transpose(trend_vals, perm=[0, 2, 1])  # shape: (N, T, D)
        trend_vals = tf.cast(trend_vals, tf.float32)

        return trend_vals


class SeasonalLayer(Layer):
    def __init__(self, feat_dim, seq_len, custom_seas, **kwargs):
        super(SeasonalLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.custom_seas = custom_seas
        self.dense_layers = [
            Dense(feat_dim * num_seasons, name=f"season_params_{i}")
            for i, (num_seasons, len_per_season) in enumerate(custom_seas)
        ]
        self.reshape_layers = [
            Reshape(target_shape=(feat_dim, num_seasons))
            for num_seasons, len_per_season in custom_seas
        ]

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = tf.range(num_seasons)[:, None] + tf.zeros(
            (num_seasons, len_per_season), dtype=tf.int32
        )
        season_indexes = tf.reshape(season_indexes, [-1])
        season_indexes = tf.tile(season_indexes, [self.seq_len // len_per_season + 1])[
            : self.seq_len
        ]
        return season_indexes

    def call(self, z):
        N = tf.shape(z)[0]
        ones_tensor = tf.ones(shape=[N, self.feat_dim, self.seq_len], dtype=tf.int32)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)  # shape: (N, D * S)
            season_params = self.reshape_layers[i](season_params)  # shape: (N, D, S)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            )

            dim2_idxes = ones_tensor * tf.reshape(
                season_indexes_over_time, shape=(1, 1, -1)
            )  # shape: (N, D, T)
            season_vals = tf.gather(
                season_params, dim2_idxes, batch_dims=-1
            )  # shape (N, D, T)

            all_seas_vals.append(season_vals)

        all_seas_vals = K.stack(all_seas_vals, axis=-1)  # shape: (N, D, T, S)
        all_seas_vals = tf.reduce_sum(all_seas_vals, axis=-1)  # shape (N, D, T)
        all_seas_vals = tf.transpose(all_seas_vals, perm=[0, 2, 1])  # shape (N, T, D)
        return all_seas_vals

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len, self.feat_dim)


# ============================================================================
# TIMEVAE MODEL
# ============================================================================

class TimeVAE(BaseVariationalAutoencoder):
    model_name = "TimeVAE"

    def __init__(
        self,
        hidden_layer_sizes,
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
        **kwargs,
    ):
        """
        TimeVAE: Temporal Variational Autoencoder
        
        Args:
            hidden_layer_sizes: list of number of filters in convolutional layers
            trend_poly: integer for number of orders for trend component
            custom_seas: list of tuples of (num_seasons, len_per_season)
            use_residual_conn: boolean for residual connection
        """

        super(TimeVAE, self).__init__(**kwargs)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.compile(optimizer=Adam())

    def _get_encoder(self):
        encoder_inputs = Input(
            shape=(self.seq_len, self.feat_dim), name="encoder_input"
        )
        x = encoder_inputs
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                activation="relu",
                padding="same",
                name=f"enc_conv_{i}",
            )(x)

        x = Flatten(name="enc_flatten")(x)

        self.encoder_last_dense_dim = x.shape[-1]

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        encoder_output = Sampling()([z_mean, z_log_var])
        self.encoder_output = encoder_output

        encoder = Model(
            encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder"
        )
        return encoder

    def _get_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dim,), name="decoder_input")

        outputs = None
        outputs = self.level_model(decoder_inputs)
        
        # trend polynomials
        if self.trend_poly is not None and self.trend_poly > 0:
            trend_vals = TrendLayer(self.feat_dim, self.trend_poly, self.seq_len)(
                decoder_inputs
            )
            outputs = trend_vals if outputs is None else outputs + trend_vals

        # custom seasons
        if self.custom_seas is not None and len(self.custom_seas) > 0:
            cust_seas_vals = SeasonalLayer(
                feat_dim=self.feat_dim,
                seq_len=self.seq_len,
                custom_seas=self.custom_seas,
            )(decoder_inputs)
            outputs = cust_seas_vals if outputs is None else outputs + cust_seas_vals

        if self.use_residual_conn:
            residuals = self._get_decoder_residual(decoder_inputs)
            outputs = residuals if outputs is None else outputs + residuals

        if outputs is None:
            raise ValueError(
                "Error: No decoder model to use. "
                "You must use one or more of: "
                "trend, custom seasonality, and/or residual connection."
            )

        decoder = Model(decoder_inputs, [outputs], name="decoder")
        return decoder

    def level_model(self, z):
        level_params = Dense(self.feat_dim, name="level_params", activation="relu")(z)
        level_params = Dense(self.feat_dim, name="level_params2")(level_params)
        level_params = Reshape(target_shape=(1, self.feat_dim))(
            level_params
        )  # shape: (N, 1, D)

        ones_tensor = tf.ones(
            shape=[1, self.seq_len, 1], dtype=tf.float32
        )  # shape: (1, T, 1)

        level_vals = level_params * ones_tensor
        return level_vals

    def _get_decoder_residual(self, x):
        x = Dense(self.encoder_last_dense_dim, name="dec_dense", activation="relu")(x)
        x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(
            x
        )

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
                name=f"dec_deconv_{i}",
            )(x)

        # last de-convolution
        x = Conv1DTranspose(
            filters=self.feat_dim,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            name=f"dec_deconv_final",
        )(x)

        x = Flatten(name="dec_flatten")(x)
        x = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
        residuals = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        return residuals

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        super().save(model_dir)

        if self.custom_seas is not None:
            custom_seas_serializable = [
                (int(num_seasons), int(len_per_season))
                for num_seasons, len_per_season in self.custom_seas
            ]
        else:
            custom_seas_serializable = None

        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "trend_poly": self.trend_poly,
            "custom_seas": custom_seas_serializable,
            "use_residual_conn": self.use_residual_conn,
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

    @classmethod
    def load(cls, model_dir: str) -> "TimeVAE":
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = TimeVAE(**dict_params)
        vae_model.load_weights(model_dir)
        vae_model.compile(optimizer=Adam())
        return vae_model