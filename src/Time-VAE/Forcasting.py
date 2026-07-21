import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
import re
import time
import gc

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
root = Path(__file__).resolve().parents[2]
PROCESSED_DIR = root / "outputs" / "temporal_split_data"
SYNTHETIC_DIR = root / "outputs" / "Time-VAE" / "original_timevae_results"
SAVE_DIR = root / "outputs" / "Time-VAE" / "Time-vae_forecasting_results"

RANDOM_STATE = 42
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================================================================
# METRICS & HELPERS
# ==============================================================================
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
            'index': data['index'] if 'index' in data.files else None,
            'metadata': data['metadata'].item()
        }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_synthetic_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        return {
            'synthetic_samples': data['synthetic_samples'].astype(np.float32),
            'norm_params': data['norm_params'].item() if 'norm_params' in data.files else None,
            'norm_key': data['norm_key'].item() if 'norm_key' in data.files else None,
        }
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
    """Extract and denormalize the selected target from processed real data."""
    feature_names = dataset['feature_names']
    if feature_name not in feature_names:
        raise KeyError(f"Feature '{feature_name}' not found in processed dataset")
    feature_idx = feature_names.index(feature_name)
    normalized_target = dataset['data'][:, :, feature_idx]
    return denormalize_data(normalized_target, dataset['norm_params'], feature_name)


def extract_synthetic_target(synthetic_dataset, reference_dataset,
                             feature_name='total_demand_clipped'):
    """Extract and denormalize the matching target from Time-VAE samples."""
    samples = synthetic_dataset['synthetic_samples']
    feature_names = reference_dataset['feature_names']
    if feature_name not in feature_names:
        raise KeyError(f"Feature '{feature_name}' not found in processed dataset")
    feature_idx = feature_names.index(feature_name)
    if samples.ndim != 3:
        raise ValueError(f"Expected 3D synthetic samples, received shape {samples.shape}")
    if samples.shape[2] == 1:
        feature_idx = 0
    elif feature_idx >= samples.shape[2]:
        raise ValueError(
            f"Synthetic feature dimension {samples.shape[2]} does not contain "
            f"target index {feature_idx}"
        )
    norm_params = synthetic_dataset['norm_params'] or reference_dataset['norm_params']
    return denormalize_data(samples[:, :, feature_idx], norm_params, feature_name)


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

# ==============================================================================
# DISCOVERY HELPERS
# ==============================================================================
def ratio_to_suffix(ratio):
    """Return the filename-safe ratio format used by Time-VAE training."""
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
        file_path = SYNTHETIC_DIR / "synthetic_data" / f"ratio_{ratio_suffix}" / f"synthetic_original_level{level}_trial{trial}_ratio{ratio_suffix}.npz"
        if file_path.exists():
            available_ratios.append(ratio)
    return sorted(available_ratios)

def get_synthetic_file_path(level, trial, ratio):
    ratio_suffix = ratio_to_suffix(ratio)
    file_path = SYNTHETIC_DIR / "synthetic_data" / f"ratio_{ratio_suffix}" / f"synthetic_original_level{level}_trial{trial}_ratio{ratio_suffix}.npz"
    return str(file_path) if file_path.exists() else None

# ==============================================================================
# MAIN EVALUATION LOGIC
# ==============================================================================
def run_forecasting_evaluation():
    # --- START TIMER ---
    overall_start_time = time.time()
    
    print("================================================================================")
    print("TIMEVAE FORECASTING EVALUATION (Baseline, Synthetic-Only, Augmented)")
    print("Targeting: Original TimeVAE Results")
    print("================================================================================")
    
    available_levels = discover_available_levels()
    if not available_levels: return

    print(f"\nAvailable Levels: {available_levels}")
    selected_level = int(input("Enter aggregation level: "))

    available_trials = discover_available_trials(selected_level)
    print(f"Available Trials ({len(available_trials)}): {available_trials}")
    num_trials = int(input(f"How many trials to process (1-{len(available_trials)}): "))
    selected_trials = available_trials[:num_trials]

    available_ratios = discover_available_ratios(selected_level, selected_trials[0])
    print(f"Available Ratios: {available_ratios}")
    selected_ratio = float(input("Enter synthetic ratio: "))

    BATCH_SIZE = 10
    REST_TIME = 15
    
    all_results = []
    models_dict = {
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE),
        'SVM': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    }
    
    trial_batches = [selected_trials[i:i + BATCH_SIZE] for i in range(0, len(selected_trials), BATCH_SIZE)]

    for batch_idx, batch_trials in enumerate(trial_batches):
        print(f"\n>>> PROCESSING BATCH {batch_idx + 1}/{len(trial_batches)} <<<")
        
        for trial in batch_trials:
            print(f"\n   --- Trial {trial} ---")
            train_path = PROCESSED_DIR / "train" / f"train_preprocessed_level_{selected_level}_trial_{trial}.npz"
            test_path = PROCESSED_DIR / "test" / f"test_preprocessed_level_{selected_level}_trial_{trial}.npz"
            syn_path = get_synthetic_file_path(selected_level, trial, selected_ratio)

            train_data = load_real_data(str(train_path))
            test_data = load_real_data(str(test_path))
            synthetic_data = load_synthetic_data(syn_path)
            
            if not train_data or not test_data or synthetic_data is None:
                print(f"   [!] Missing files for trial {trial}. Skipping...")
                continue

            # Create Real Datasets
            train_real = extract_real_target(train_data)
            test_real = extract_real_target(test_data)
            X_train_real, y_train_real = create_features_and_targets(train_real)
            X_test_real, y_test_real = create_features_and_targets(test_real)

            # Create Synthetic Datasets
            synthetic_target = extract_synthetic_target(synthetic_data, train_data)
            X_syn, y_syn = create_features_and_targets(synthetic_target)
            
            # Align features
            common_feats = X_train_real.columns.intersection(X_syn.columns)
            X_train_real, X_test_real, X_syn = X_train_real[common_feats], X_test_real[common_feats], X_syn[common_feats]

            # Augmented Dataset
            X_aug = pd.concat([X_train_real, X_syn], ignore_index=True)
            y_aug = pd.concat([y_train_real, y_syn], ignore_index=True)

            for name, model_template in models_dict.items():
                # --- 1. BASELINE ---
                m_base = model_template.__class__(**model_template.get_params())
                m_base.fit(X_train_real, y_train_real)
                res_base = calculate_metrics(y_test_real, m_base.predict(X_test_real))

                # --- 2. SYNTHETIC ONLY ---
                m_syn = model_template.__class__(**model_template.get_params())
                m_syn.fit(X_syn, y_syn)
                res_syn = calculate_metrics(y_test_real, m_syn.predict(X_test_real))

                # --- 3. AUGMENTED ---
                m_aug = model_template.__class__(**model_template.get_params())
                m_aug.fit(X_aug, y_aug)
                res_aug = calculate_metrics(y_test_real, m_aug.predict(X_test_real))

                # Calculate Improvement (Baseline vs Augmented)
                imp_pct = ((res_base['rmse'] - res_aug['rmse']) / res_base['rmse'] * 100) if res_base['rmse'] > 0 else 0
                print(f"      {name:18}: Base {res_base['rmse']:.3f} | Syn {res_syn['rmse']:.3f} | Aug {res_aug['rmse']:.3f} ({imp_pct:+.2f}%)")

                # Store for report
                for stype, metrics in [('baseline', res_base), ('synthetic_only', res_syn), ('augmented', res_aug)]:
                    all_results.append({
                        'level': selected_level, 'trial': trial, 'ratio': selected_ratio, 
                        'model': name, 'training_type': stype,
                        'baseline_rmse': res_base['rmse'], 
                        'result_rmse': metrics['rmse'], 
                        'rmse_improvement_pct': imp_pct if stype == 'augmented' else 0,
                        'result_r2': metrics['r2'], 'result_mape': metrics['mape']
                    })

        if batch_idx < len(trial_batches):
            gc.collect()
            time.sleep(REST_TIME)

    # Save results and print final benchmark
    if all_results:
        df_res = pd.DataFrame(all_results)
        
        # Define the specific filename and full path
        fname = f"results_level{selected_level}_ratio{ratio_to_suffix(selected_ratio)}.csv"
        save_path = SAVE_DIR / fname
        
        # Save to the CSV file
        df_res.to_csv(save_path, index=False)
        
        # --- TIMER CALCULATION ---
        overall_end_time = time.time()
        total_seconds = overall_end_time - overall_start_time
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        
        # Final Terminal Output
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print(f"⏱️  Total Execution Time: {hours} hours and {minutes} minutes")
        print(f"📂 Results saved to: {save_path}")
        print("=" * 80)
        
    else:
        print("\nNo results generated. Please check if your synthetic and real data paths are correct.")

    return all_results

if __name__ == "__main__":
    run_forecasting_evaluation()
