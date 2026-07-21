import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import time
import gc
from pathlib import Path

warnings.filterwarnings('ignore')

# ================================================================
# Path-independent configuration
# ================================================================
root = Path(__file__).resolve().parents[2]
PROCESSED_DIR = root / "outputs" / "temporal_split_data"
SYNTHETIC_DIR = root / "outputs" / "CFT-VAE"
SAVE_DIR = root / "outputs" / "CFT-VAE" / "CFT-VAE-forecasting_results"

RANDOM_STATE = 42
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Model Wrappers
# =========================
class LSTMRegressorWrapper:
    def __init__(self, units=64, epochs=30, batch_size=32):
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
   
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_arr = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]).astype(np.float32)
        y_arr = y.values.astype(np.float32)
       
        self.model = Sequential([
            LSTM(self.units, activation='relu', input_shape=(1, X.shape[1])),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
       
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        self.model.fit(X_arr, y_arr, epochs=self.epochs, batch_size=self.batch_size,
                       validation_split=0.1, shuffle=True, callbacks=[early_stop], verbose=0)
   
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_arr = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]).astype(np.float32)
        return self.model.predict(X_arr, verbose=0).flatten()

class TCNRegressorWrapper:
    def __init__(self, filters=64, kernel_size=3, epochs=30, batch_size=32):
        self.filters = filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_arr = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1).astype(np.float32)
        y_arr = y.values.astype(np.float32)
        
        self.model = Sequential([
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, dilation_rate=1,
                   padding='causal', activation='relu', input_shape=(X.shape[1], 1)),
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, dilation_rate=2,
                   padding='causal', activation='relu'),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_arr, y_arr, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_arr = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1).astype(np.float32)
        return self.model.predict(X_arr, verbose=0).flatten()

# =========================
# Metrics & Helpers (Same as 4-models)
# =========================
def calculate_mape(y_true, y_pred):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if len(df) == 0: return np.nan
    mask = np.abs(df["y_true"]) > 1e-10
    if mask.sum() == 0: return np.nan
    mape = np.mean(np.abs((df.loc[mask, "y_true"] - df.loc[mask, "y_pred"]) / df.loc[mask, "y_true"])) * 100
    return mape

def calculate_metrics(y_true, y_pred):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if len(df) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mape': np.nan, 'n': 0}
   
    rmse = np.sqrt(mean_squared_error(df["y_true"], df["y_pred"]))
    mae = mean_absolute_error(df["y_true"], df["y_pred"])
    r2 = r2_score(df["y_true"], df["y_pred"])
    mape = calculate_mape(df["y_true"], df["y_pred"])
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape, 'n': len(df)}

def load_real_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        return {
            'data': data['data'].astype(np.float32),
            'norm_params': data['norm_params'].item(),
            'feature_names': list(data['feature_names']),
            'metadata': data['metadata'].item()
        }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_synthetic_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        return data['synthetic_samples']
    except Exception as e:
        print(f"Error loading synthetic data {file_path}: {e}")
        return None

def denormalize_data(data, norm_params, feature_name='total_demand_clipped'):
    if norm_params is None or feature_name not in norm_params:
        return data
    norm_info = norm_params[feature_name]
    if 'min' in norm_info and 'max' in norm_info:
        return data * (norm_info['max'] - norm_info['min'] + 1e-7) + norm_info['min']
    return data


def extract_real_target(dataset, feature_name='total_demand_clipped'):
    """Extract and denormalize the same demand feature modeled by CFT-VAE."""
    feature_names = dataset['feature_names']
    if feature_name not in feature_names:
        raise KeyError(f"Feature '{feature_name}' not found in processed dataset")
    feature_idx = feature_names.index(feature_name)
    normalized_target = dataset['data'][:, :, feature_idx]
    return denormalize_data(normalized_target, dataset['norm_params'], feature_name)


def create_features_and_targets(time_series_data):
    if len(time_series_data.shape) == 3:
        flattened = time_series_data[:, :, 0].flatten()
    else:
        flattened = time_series_data.flatten()
   
    time_index = pd.date_range(start='2020-01-01', periods=len(flattened), freq='H')
    df = pd.DataFrame({'demand': flattened}, index=time_index)
    features = pd.DataFrame(index=df.index)
   
    for lag in [1, 2, 3, 24, 25, 48, 168]:
        features[f'lag_{lag}'] = df['demand'].shift(lag)
   
    features['roll_24h_mean'] = df['demand'].shift(1).rolling(24, min_periods=12).mean()
    features['roll_24h_std'] = df['demand'].shift(1).rolling(24, min_periods=12).std()
    features['roll_168h_mean'] = df['demand'].shift(1).rolling(168, min_periods=24).mean()
   
    features['hour'] = features.index.hour
    features['dayofweek'] = features.index.dayofweek
    features['month'] = features.index.month
    features['hour_sin'] = np.sin(2 * np.pi * features.index.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features.index.hour / 24)
    features['dayofweek_sin'] = np.sin(2 * np.pi * features.index.dayofweek / 7)
    features['dayofweek_cos'] = np.cos(2 * np.pi * features.index.dayofweek / 7)
   
    combined = pd.concat([features, df['demand'].rename('target')], axis=1).dropna()
    X = combined.drop('target', axis=1)
    y = combined['target']
    return X, y

# =========================
# Discovery Helpers (Same as 4-models)
# =========================
def ratio_to_suffix(ratio):
    """Return the filename-safe ratio format used by CFT-VAE training."""
    return str(float(ratio)).replace(".", "p")


def discover_available_levels():
    train_dir = PROCESSED_DIR / "train"
    if not train_dir.exists(): return []
    train_files = list(train_dir.glob("train_preprocessed_*.npz"))
    levels = sorted({int(re.search(r"level_(\d+)_trial_", f.stem).group(1)) for f in train_files})
    return levels

def discover_available_trials(level):
    train_dir = PROCESSED_DIR / "train"
    if not train_dir.exists(): return []
    trials = []
    for file in train_dir.glob(f"train_preprocessed_level_{level}_trial_*.npz"):
        match = re.search(r"trial_(\d+)", file.stem)
        if match:
            trial = int(match.group(1))
            test_file = PROCESSED_DIR / "test" / f"test_preprocessed_level_{level}_trial_{trial}.npz"
            if test_file.exists():
                trials.append(trial)
    return sorted(trials)

def discover_available_ratios(level, trial):
    available_ratios = []
    supported_ratios = [1, 1.5, 2, 5, 10, 50]
    for ratio in supported_ratios:
        ratio_suffix = ratio_to_suffix(ratio)
        file_path = SYNTHETIC_DIR / "synthetic_data" / f"ratio_{ratio_suffix}" / f"synthetic_level{level}_trial{trial}_ratio{ratio_suffix}.npz"
        if file_path.exists():
            available_ratios.append(ratio)
    return sorted(available_ratios)

def get_synthetic_file_path(level, trial, ratio):
    ratio_suffix = ratio_to_suffix(ratio)
    file_path = SYNTHETIC_DIR / "synthetic_data" / f"ratio_{ratio_suffix}" / f"synthetic_level{level}_trial{trial}_ratio{ratio_suffix}.npz"
    return str(file_path) if file_path.exists() else None

# =========================
# Main Logic
# =========================
def run_forecasting_evaluation():
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    
    print("================================================================================")
    print("CFT-VAE FORECASTING EVALUATION - LSTM & TCN")
    print("================================================================================")
   
    available_levels = discover_available_levels()
    if not available_levels: 
        print(f"Error: No data found in {PROCESSED_DIR}")
        return

    print(f"\nAvailable Levels: {available_levels}")
    selected_level = int(input("Enter aggregation level: "))

    available_trials = discover_available_trials(selected_level)
    print(f"Available Trials ({len(available_trials)}): {available_trials}")
    num_trials = int(input(f"How many trials to process (1-{len(available_trials)}): "))
    selected_trials = available_trials[:num_trials]

    available_ratios = discover_available_ratios(selected_level, selected_trials[0])
    print(f"Available Ratios: {available_ratios}")
    selected_ratio = float(input("Enter synthetic ratio: "))

    all_results = []
   
    models_dict = {
        'LSTM': LSTMRegressorWrapper(units=64, epochs=30),
        'TCN': TCNRegressorWrapper(filters=64, epochs=30)
    }

    for trial in selected_trials:
        print(f"\n   --- Trial {trial} ---")
        train_path = PROCESSED_DIR / "train" / f"train_preprocessed_level_{selected_level}_trial_{trial}.npz"
        test_path = PROCESSED_DIR / "test" / f"test_preprocessed_level_{selected_level}_trial_{trial}.npz"
        syn_path = get_synthetic_file_path(selected_level, trial, selected_ratio)

        train_data = load_real_data(str(train_path))
        test_data = load_real_data(str(test_path))
        synthetic_raw = load_synthetic_data(syn_path)
       
        if not train_data or not test_data or synthetic_raw is None:
            print(f"  Skipping trial {trial} (missing data)")
            continue

        train_real = extract_real_target(train_data)
        test_real = extract_real_target(test_data)
        X_train_real, y_train_real = create_features_and_targets(train_real)
        X_test_real, y_test_real = create_features_and_targets(test_real)

        X_syn, y_syn = create_features_and_targets(synthetic_raw)
        
        common_feats = X_train_real.columns.intersection(X_syn.columns)
        X_train_real = X_train_real[common_feats]
        X_test_real = X_test_real[common_feats]
        X_syn = X_syn[common_feats]

        X_aug = pd.concat([X_train_real, X_syn], ignore_index=True)
        y_aug = pd.concat([y_train_real, y_syn], ignore_index=True)

        for name, model_template in models_dict.items():
            tf.random.set_seed(RANDOM_STATE)

            if name == 'LSTM':
                m = LSTMRegressorWrapper(units=model_template.units, epochs=model_template.epochs, batch_size=model_template.batch_size)
            else:
                m = TCNRegressorWrapper(filters=model_template.filters, kernel_size=model_template.kernel_size,
                                      epochs=model_template.epochs, batch_size=model_template.batch_size)
            
            # Base
            m.fit(X_train_real, y_train_real)
            res_base = calculate_metrics(y_test_real, m.predict(X_test_real))

            # Synthetic Only
            tf.random.set_seed(RANDOM_STATE)
            if name == 'LSTM':
                m = LSTMRegressorWrapper(units=model_template.units, epochs=model_template.epochs, batch_size=model_template.batch_size)
            else:
                m = TCNRegressorWrapper(filters=model_template.filters, kernel_size=model_template.kernel_size,
                                      epochs=model_template.epochs, batch_size=model_template.batch_size)
            m.fit(X_syn, y_syn)
            res_syn = calculate_metrics(y_test_real, m.predict(X_test_real))

            # Augmented
            tf.random.set_seed(RANDOM_STATE)
            if name == 'LSTM':
                m = LSTMRegressorWrapper(units=model_template.units, epochs=model_template.epochs, batch_size=model_template.batch_size)
            else:
                m = TCNRegressorWrapper(filters=model_template.filters, kernel_size=model_template.kernel_size,
                                      epochs=model_template.epochs, batch_size=model_template.batch_size)
            m.fit(X_aug, y_aug)
            res_aug = calculate_metrics(y_test_real, m.predict(X_test_real))

            imp_pct = ((res_base['rmse'] - res_aug['rmse']) / res_base['rmse'] * 100) if res_base.get('rmse', 0) > 0 else 0

            print(f"      {name:18}: Base {res_base['rmse']:.3f} | Syn {res_syn['rmse']:.3f} | Aug {res_aug['rmse']:.3f} ({imp_pct:+.2f}%)")

            for stype, res in [('baseline', res_base), ('synthetic_only', res_syn), ('augmented', res_aug)]:
                all_results.append({
                    'level': selected_level, 'trial': trial, 'ratio': selected_ratio,
                    'model': name, 'training_type': stype,
                    'result_rmse': res['rmse'], 'result_mae': res['mae'],
                    'result_r2': res['r2'], 'result_mape': res['mape']
                })

        gc.collect()

    # Save
    if all_results:
        df_res = pd.DataFrame(all_results)
        ratio_str = ratio_to_suffix(selected_ratio)
        fname = f"results_level{selected_level}_ratio{ratio_str}_lstm_tcn.csv"
        df_res.to_csv(SAVE_DIR / fname, index=False)
        print(f"\nDONE. Saved to: {SAVE_DIR / fname}")

if __name__ == "__main__":
    run_forecasting_evaluation()
