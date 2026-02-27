import numpy as np
import tensorflow as tf
import os
import time
from pathlib import Path
import re
import random
import pandas as pd
import gc  # Added for memory cleanup

# Import your TimeVAE model definition
from Model_definition import TimeVAE

# ---------------------------------------------------
# Global Reproducibility
# ---------------------------------------------------
np.random.seed(123)
random.seed(123)
tf.keras.utils.set_random_seed(123)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '123'

# ---------------------------------------------------
# Loss Monitor
# ---------------------------------------------------
class LossMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0: 
            total_loss = logs.get('loss', 0)
            recon_loss = logs.get('reconstruction_loss', 0)
            print(f"    Epoch {epoch + 1}: Total={total_loss:.4f}, Recon={recon_loss:.4f}")

        if np.isnan(logs.get('loss', 0)) or np.isinf(logs.get('loss', 0)):
            print("    WARNING: Loss is NaN/Inf -> stopping training")
            self.model.stop_training = True

# ---------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------
def load_temporal_dataset(file_path: str):
    try:
        data = np.load(file_path, allow_pickle=True)
        return {
            'data': data['data'].astype(np.float32),
            'norm_params': data['norm_params'].item(),
            'feature_names': list(data['feature_names']),
            'index': data['index'],
            'metadata': data['metadata'].item()
        }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_matching_datasets(base_dir: str, level: int, trial: int):
    base_name = f"level_{level}_trial_{trial}"
    return {
        'train': os.path.join(base_dir, "train", f"train_preprocessed_{base_name}.npz"),
        'val': os.path.join(base_dir, "val", f"val_preprocessed_{base_name}.npz"),
        'test': os.path.join(base_dir, "test", f"test_preprocessed_{base_name}.npz")
    }

def discover_available_trials(base_dir: str, level: int):
    train_dir = os.path.join(base_dir, "train")
    if not os.path.exists(train_dir):
        return []

    available_trials = []
    pattern = f"train_preprocessed_level_{level}_trial_*.npz"

    for file_path in Path(train_dir).glob(pattern):
        match = re.search(rf"level_{level}_trial_(\d+)", file_path.stem)
        if match:
            trial_num = int(match.group(1))
            paths = get_matching_datasets(base_dir, level, trial_num)
            if all(os.path.exists(p) for p in paths.values()):
                available_trials.append(trial_num)

    return sorted(available_trials)

def discover_available_levels(base_dir: str):
    train_dir = os.path.join(base_dir, "train")
    if not os.path.exists(train_dir):
        return []
    levels = set()
    for file_path in Path(train_dir).glob("train_preprocessed_level_*.npz"):
        match = re.search(r"level_(\d+)_trial_", file_path.stem)
        if match:
            levels.add(int(match.group(1)))
    return sorted(list(levels))

def prepare_data_with_forecast(dataX, feature_idx, forecast_horizon=24):
    X = dataX[:-1].astype(np.float32)
    diff_data = dataX[1:, :, feature_idx].astype(np.float32) 
    y_future = diff_data.reshape(-1, forecast_horizon, 1)
    return X, y_future

# ---------------------------------------------------
# Synthetic Generation
# ---------------------------------------------------
def generate_synthetic_data_conditional(vae_model, level, trial, ratio, save_dir, 
                                      real_x, real_cond, norm_params, norm_key):
    num_real = real_x.shape[0]
    num_synthetic_samples = int(num_real * ratio)
    
    ratio_str = str(ratio).replace(".", "p")
    synthetic_dir = os.path.join(save_dir, "synthetic_data", f"ratio_{ratio_str}")
    os.makedirs(synthetic_dir, exist_ok=True)

    try:
        gen_seed = 123 + level * 1000 + trial * 100
        np.random.seed(gen_seed)
        tf.random.set_seed(gen_seed)
        
        if vae_model.cond_dim > 0 and real_cond is not None:
            if real_cond.shape[0] >= num_synthetic_samples:
                selected_x = real_x[:num_synthetic_samples]
                selected_cond = real_cond[:num_synthetic_samples]
            else:
                repeats = (num_synthetic_samples // real_cond.shape[0]) + 1
                selected_x = np.tile(real_x, (repeats, 1, 1))[:num_synthetic_samples]
                selected_cond = np.tile(real_cond, (repeats, 1, 1))[:num_synthetic_samples]

            z_mean, z_log_var, _ = vae_model.encoder.predict([selected_x, selected_cond], verbose=0)
            eps = np.random.normal(size=z_mean.shape).astype(np.float32)
            z_samples = z_mean + np.exp(0.5 * z_log_var) * eps
            synthetic_samples = vae_model.decoder.predict([z_samples, selected_cond], verbose=0)
        else:
            z_samples = np.random.normal(0, 1, (num_synthetic_samples, vae_model.latent_dim)).astype(np.float32)
            synthetic_samples = vae_model.decoder.predict(z_samples, verbose=0)

        min_val = norm_params[norm_key]['min']
        max_val = norm_params[norm_key]['max']
        synthetic_samples_denorm = synthetic_samples[:, :, 0:1] * (max_val - min_val + 1e-7) + min_val

        fname = f"synthetic_level{level}_trial{trial}_ratio{ratio_str}.npz"
        save_path = os.path.join(synthetic_dir, fname)
        
        np.savez_compressed(
            save_path,
            synthetic_samples=synthetic_samples_denorm,
            conditioning_data=selected_cond if vae_model.cond_dim > 0 else None,
            metadata={'level': level, 'trial': trial, 'ratio': ratio, 'seed': gen_seed}
        )
        print(f"    ✓ Synthetic Data Saved")
        return save_path

    except Exception as e:
        print(f"    ✗ Error generating synthetic data: {e}")
        return None

# ---------------------------------------------------
# Main Training Function (With Batching)
# ---------------------------------------------------
def train_temporal_vae(data_dir, norm_key='total_demand_clipped', generate_synthetic=True):
    # --- START TIMER ---
    total_start_time = time.time()

    print("================================================================================")
    print("CFT-VAE TRAINING PIPELINE (Batched Mode)")
    print(f"Data Directory: {data_dir}")
    print("================================================================================")
    
    models_dir = os.path.join(data_dir, "metamodels")
    history_dir = os.path.join(data_dir, "history")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    
    # 1. Selection
    levels = discover_available_levels(data_dir)
    print(f"\nAvailable Levels: {levels}")
    sel_level = int(input(f"Enter Level to train: "))

    all_trials = discover_available_trials(data_dir, sel_level)
    print(f"Available Trials ({len(all_trials)}): {all_trials}")
    n_trials = int(input(f"How many trials to use (1-{len(all_trials)}): "))
    trials_to_run = all_trials[:n_trials]

    # 2. Batch Settings
    BATCH_RUN_SIZE = 10   # Run 10 trials
    REST_TIME = 30        # Rest 30 seconds
    
    ratios = [1, 2, 5, 10]
    print(f"Available Ratios: {ratios}")
    sel_ratio = float(input("Enter Synthetic Ratio: "))

    # ---------------------------------------------------
    # BATCHED TRAINING LOOP
    # ---------------------------------------------------
    print(f"\nStarting Training: Level {sel_level} | {len(trials_to_run)} Trials | Batches of {BATCH_RUN_SIZE}")
    
    # Split trials into chunks of 10
    trial_batches = [trials_to_run[i:i + BATCH_RUN_SIZE] for i in range(0, len(trials_to_run), BATCH_RUN_SIZE)]
    
    for batch_idx, batch_trials in enumerate(trial_batches):
        print(f"\n>>> PROCESSING BATCH {batch_idx + 1}/{len(trial_batches)} (Trials: {batch_trials}) <<<")
        
        for i, trial in enumerate(batch_trials):
            print(f"\n--- Trial {trial} ---")
            
            # A. Reproducibility
            trial_seed = 123 + sel_level * 1000 + trial * 100
            np.random.seed(trial_seed)
            tf.random.set_seed(trial_seed)
            random.seed(trial_seed)
            
            # B. Load & Prep
            paths = get_matching_datasets(data_dir, sel_level, trial)
            train_data = load_temporal_dataset(paths['train'])
            val_data = load_temporal_dataset(paths['val'])
            
            feats = train_data['feature_names']
            demand_idx = feats.index(norm_key)
            cond_dim = len(feats) - 1 
            
            train_X_raw, train_y_future = prepare_data_with_forecast(train_data['data'], demand_idx)
            val_X_raw, val_y_future = prepare_data_with_forecast(val_data['data'], demand_idx)
            
            train_x = train_X_raw[:, :, demand_idx:demand_idx+1]
            val_x = val_X_raw[:, :, demand_idx:demand_idx+1]
            
            train_cond = None
            val_cond = None
            if cond_dim > 0:
                cond_indices = [k for k in range(len(feats)) if k != demand_idx]
                train_cond = train_X_raw[:, :, cond_indices]
                val_cond = val_X_raw[:, :, cond_indices]
            
            BATCH_SIZE = 32
            if cond_dim > 0:
                train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_cond, train_y_future)).shuffle(1024).batch(BATCH_SIZE)
                val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_cond, val_y_future)).batch(BATCH_SIZE)
            else:
                train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y_future)).shuffle(1024).batch(BATCH_SIZE)
                val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y_future)).batch(BATCH_SIZE)

            # C. Train
            vae = TimeVAE(
                seq_len=24, feat_dim=1, cond_dim=cond_dim, latent_dim=32,
                trend_dim=10, seasonal_dim=10, noise_dim=12,
                hidden_layer_sizes=[64, 128], 
                forecast_horizon=24, 
                reconstruction_wt=3.0, dtw_wt=1.0, forecast_wt=1.0
            )
            
            callbacks = [
                LossMonitor(),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                tf.keras.callbacks.CSVLogger(os.path.join(history_dir, f"history_level{sel_level}_trial{trial}.csv"))
            ]
            
            vae.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks, verbose=0)
            
            # D. Save
            weights_path = os.path.join(models_dir, f"weights_level{sel_level}_trial{trial}.h5")
            vae.save_weights(weights_path)
            print(f"    ✓ Weights Saved")
            
            if generate_synthetic:
                generate_synthetic_data_conditional(
                    vae, sel_level, trial, sel_ratio, data_dir,
                    train_x, train_cond, train_data['norm_params'], norm_key
                )

        # ---------------------------------------------------
        # CLEANUP & REST AFTER BATCH
        # ---------------------------------------------------
        if batch_idx < len(trial_batches) - 1: # Don't sleep after the last batch
            print(f"\n[Batch Complete] Cleaning memory and resting for {REST_TIME} seconds...")
            
            # 1. Clear Keras Session (Frees GPU/Graph memory)
            tf.keras.backend.clear_session()
            
            # 2. Garbage Collect (Frees Python RAM)
            gc.collect()
            
            # 3. Sleep
            time.sleep(REST_TIME)
            print(">>> Resuming Next Batch >>>")

    # --- STOP TIMER & PRINT DURATION ---
    overall_end_time = time.time()
    total_seconds = overall_end_time - total_start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE.")
    print(f"⏱️  Total Execution Time: {hours} hours and {minutes} minutes")
    print("=" * 80)

if __name__ == "__main__":
    DATA_DIR = r"C:\Users\bin150\OneDrive - UBC\Desktop\Publication\WR2\cft-vae\data"
    train_temporal_vae(DATA_DIR, norm_key='total_demand_clipped', generate_synthetic=True)