import numpy as np
import tensorflow as tf
import os
import time
import re
import gc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CFT_VAE_SOURCE_DIR = REPO_ROOT / "src" / "CFT-VAE"
PROCESSED_DATA_DIR = REPO_ROOT / "outputs" / "temporal_split_data"
ABLATION_BASE = REPO_ROOT / "outputs" / "CFT-VAE" / "ablation"

if str(CFT_VAE_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(CFT_VAE_SOURCE_DIR))

from Model_definition import TimeVAE

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def load_temporal_dataset(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        return {
            'data': data['data'].astype(np.float32),
            'norm_params': data['norm_params'].item(),
            'feature_names': list(data['feature_names'])
        }
    except Exception as e: 
        return None

def discover_available_trials(base_dir, level):
    train_dir = os.path.join(base_dir, "train")
    if not os.path.exists(train_dir): return []
    return sorted([int(re.search(rf"level_{level}_trial_(\d+)", f.stem).group(1)) 
                   for f in Path(train_dir).glob(f"train_preprocessed_level_{level}_trial_*.npz")])

def get_feature_mask(feature_names, exclude_cats, norm_key):
    keep_indices = []
    targets = [norm_key, 'total_demand', 'active_users', 'num_users']
    
    for i, name in enumerate(feature_names):
        if name in targets:
            continue
            
        is_masked = False
        
        # Category 1 — Lags
        if 1 in exclude_cats and 'lag_' in name: 
            is_masked = True
        
        # Category 2 — Rolling and Change
        if 2 in exclude_cats and any(x in name for x in 
            ['rolling_', 'hourly_change', 'daily_change', 'weekly_change']): 
            is_masked = True
        
        # Category 3 — Calendar
        if 3 in exclude_cats and any(x in name for x in 
            ['hour_', 'dow_', 'dom_', 'month_', 'day_of_week_', 'is_weekend']): 
            is_masked = True
        
        # Category 4 — Daily Context
        if 4 in exclude_cats and any(x in name for x in 
            ['cumsum_day', 'pct_of_daily', 'is_peak_hour', 
             'relative_to_peak', 'hours_from_noon', 'hours_from_midnight']): 
            is_masked = True
        
        # Category 5 — Stability
        if 5 in exclude_cats and any(x in name for x in 
            ['cv_', 'pattern_stability', 'deviation_']): 
            is_masked = True
            
        if not is_masked:
            keep_indices.append(i)
            
    return keep_indices

def generate_synthetic_data(vae_model, level, trial, mask_path, real_x, real_cond, norm_params, norm_key):
    target_dir = os.path.join(mask_path, "synthetic_data")
    os.makedirs(target_dir, exist_ok=True)
    
    z_samples = np.random.normal(0, 1, (real_x.shape[0], vae_model.latent_dim)).astype(np.float32)
    # Corrected conditional generation logic based on cond_dim
    if vae_model.cond_dim > 0 and real_cond is not None:
        synthetic_samples = vae_model.decoder.predict([z_samples, real_cond], verbose=0)
    else:
        synthetic_samples = vae_model.decoder.predict(z_samples, verbose=0)

    min_val, max_val = norm_params[norm_key]['min'], norm_params[norm_key]['max']
    synthetic_samples_denorm = synthetic_samples[:, :, 0:1] * (max_val - min_val + 1e-7) + min_val
    
    fname = f"synthetic_lvl{level}_trial{trial}.npz"
    np.savez_compressed(os.path.join(target_dir, fname), synthetic_samples=synthetic_samples_denorm)
    print(f"      [Saved Synthetic Data to {target_dir}]", flush=True)

# ---------------------------------------------------
# Main Ablation Training Loop
# ---------------------------------------------------
def train_temporal_vae_ablation(data_dir, norm_key='total_demand_clipped'):
    print("=== CFT-VAE ABLATION & DATA GENERATION ===")
    levels_input = input("Enter levels (comma separated): ")
    levels_to_run = [int(lvl.strip()) for lvl in levels_input.split(",")]
    
    print("\nSelect Categories to EXCLUDE:")
    print("  1: Lag Features     (lag_1h ... lag_168h)")
    print("  2: Rolling & Change (rolling_mean, rolling_std, hourly/daily/weekly_change)")
    print("  3: Calendar         (hour_sin/cos, dow, dom, month, day_of_week dummies, is_weekend)")
    print("  4: Daily Context    (cumsum_day, pct_of_daily, is_peak_hour, relative_to_peak, hours_from_noon/midnight)")
    print("  5: Stability        (cv_6h, cv_24h, pattern_stability, deviation_6h/24h)")
    print("")
    print("  Example inputs:")
    print("  Enter nothing    → Full model (all 54 conditional features)")
    print("  Enter: 1         → Remove lags only")
    print("  Enter: 1,2       → Remove lags and rolling")
    print("  Enter: 1,2,3,4,5 → Unconditional model (cond_dim=0)")
    print("")
    exclude_input = input("Enter category IDs to exclude (comma separated, or Enter for none): ")
    exclude_cats = [int(x.strip()) for x in exclude_input.split(",")] if exclude_input.strip() else []
    
    mask_label = f"mask_{exclude_cats}"
    CURRENT_MASK_DIR = os.path.join(str(ABLATION_BASE), mask_label)
    SAVE_WEIGHTS_DIR = os.path.join(CURRENT_MASK_DIR, "weights")
    os.makedirs(SAVE_WEIGHTS_DIR, exist_ok=True)
    
    n_trials = int(input("Trials per level: "))
    
    for sel_level in levels_to_run:
        print(f"\n>>> PROCESSING LEVEL {sel_level} <<<")
        trials = discover_available_trials(data_dir, sel_level)[:n_trials]
        
        for trial in trials:
            print(f"--- Level {sel_level} | Trial {trial} ---", flush=True)
            
            data = load_temporal_dataset(os.path.join(data_dir, "train", f"train_preprocessed_level_{sel_level}_trial_{trial}.npz"))
            if data is None: continue
            
            feats = data['feature_names']
            keep_idx = get_feature_mask(feats, exclude_cats, norm_key)
            cond_idx = [i for i in keep_idx if i != feats.index(norm_key)]
            # Add this right after calculating cond_idx
            # print(f"DEBUG: All feature names: {feats}")
            # print(f"DEBUG: Feature indices being kept for condition: {cond_idx}")
            # for idx in cond_idx:
            #     print(f"DEBUG: HIDDEN CONDITION FEATURE -> {feats[idx]}")
            
            X = data['data'][:-1].astype(np.float32)
            train_x = X[:, :, [feats.index(norm_key)]]
            train_cond = X[:, :, cond_idx] if len(cond_idx) > 0 else None
            y_future = data['data'][1:, :, [feats.index(norm_key)]].astype(np.float32)
            
            # --- Dynamic Model Initialization ---
            vae = TimeVAE(
                seq_len=24, 
                feat_dim=1, 
                cond_dim=len(cond_idx), 
                latent_dim=32, 
                hidden_layer_sizes=[64, 128], 
                forecast_horizon=24
            )
            
            # --- Dynamic Dataset Selection ---
            if train_cond is not None:
                ds = tf.data.Dataset.from_tensor_slices((train_x, train_cond, y_future)).shuffle(1024).batch(32)
            else:
                ds = tf.data.Dataset.from_tensor_slices((train_x, y_future)).shuffle(1024).batch(32)
            # print(f"DEBUG: Model Cond Dim: {vae.cond_dim}")
            # if train_cond is not None:
            #     print(f"DEBUG: train_cond shape: {train_cond.shape}")
            # else:
            #     print("DEBUG: train_cond is None - Model should be UNCONDITIONAL")
            print(f"      Cond features: {len(cond_idx)} | Shape: {train_cond.shape if train_cond is not None else 'None'}", flush=True)
            
            vae.fit(ds, epochs=100, verbose=0)
            
            # Save inside mask-specific subfolder
            vae.save_weights(os.path.join(SAVE_WEIGHTS_DIR, f"weights_lvl{sel_level}_trial{trial}.h5"))
            generate_synthetic_data(vae, sel_level, trial, CURRENT_MASK_DIR, train_x, train_cond, data['norm_params'], norm_key)
            
            tf.keras.backend.clear_session()
            gc.collect()

    print(f"\nExperiment Complete. Data/Weights saved to: {CURRENT_MASK_DIR}")

if __name__ == "__main__":
    train_temporal_vae_ablation(str(PROCESSED_DATA_DIR))
