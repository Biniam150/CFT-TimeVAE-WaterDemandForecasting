import numpy as np
import tensorflow as tf
import os
import time
import re
import random
import gc
from pathlib import Path

from Time_vae_model import TimeVAE 

# Set reproducibility seeds
np.random.seed(123)
random.seed(123)
tf.keras.utils.set_random_seed(123)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '123'

def clear_memory():
    gc.collect()
    tf.keras.backend.clear_session()

def load_temporal_dataset(file_path: str):
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

def get_matching_datasets(base_dir: str, level: int, trial: int):
    base_name = f"level_{level}_trial_{trial}"
    return {
        'train': os.path.join(base_dir, "train", f"train_preprocessed_{base_name}.npz"),
        'val': os.path.join(base_dir, "val", f"val_preprocessed_{base_name}.npz"),
        'test': os.path.join(base_dir, "test", f"test_preprocessed_{base_name}.npz")
    }

def discover_available_trials(processed_dir: str, level: int):
    train_dir = os.path.join(processed_dir, "train")
    if not os.path.exists(train_dir):
        return []
    available_trials = []
    pattern = f"train_preprocessed_level_{level}_trial_*.npz"
    for file_path in Path(train_dir).glob(pattern):
        match = re.search(rf"level_{level}_trial_(\d+)", file_path.stem)
        if match:
            trial_num = int(match.group(1))
            dataset_paths = get_matching_datasets(processed_dir, level, trial_num)
            if all(os.path.exists(p) for p in dataset_paths.values()):
                available_trials.append(trial_num)
    return sorted(available_trials)

def validate_dataset_compatibility(train_data, val_data):
    issues = []
    if train_data['data'].shape[1:] != val_data['data'].shape[1:]:
        issues.append(f"Shape mismatch - Train: {train_data['data'].shape}, Val: {val_data['data'].shape}")
    if train_data['feature_names'] != val_data['feature_names']:
        issues.append("Feature name mismatch between train and val")
    if (train_data['index'] is not None and val_data['index'] is not None
            and train_data['index'][-1] >= val_data['index'][0]):
        issues.append(f"Temporal overlap - Train ends: {train_data['index'][-1]}, Val starts: {val_data['index'][0]}")
    return issues

def save_training_history(history, save_path):
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = np.array(values)
    np.savez_compressed(save_path, **history_dict)
    print(f"  ✓ Training history saved: {save_path}")

def ratio_to_suffix(ratio: float) -> str:
    """Return the filename-safe ratio format shared with forecasting."""
    return str(float(ratio)).replace(".", "p")


def generate_synthetic_data_original(vae_model, level, trial, ratio, generation_idx, save_dir,
                                     real_data, norm_params, norm_key, num_synthetic_samples=500):
    ratio_str = ratio_to_suffix(ratio)
    synthetic_dir = os.path.join(save_dir, "synthetic_data", f"ratio_{ratio_str}")
    os.makedirs(synthetic_dir, exist_ok=True)
    try:
        generation_seed = 123 + level * 1000 + trial * 100 + generation_idx * 10
        np.random.seed(generation_seed)
        tf.random.set_seed(generation_seed)
        print(f"  Generating {num_synthetic_samples} synthetic samples (Level {level}, Trial {trial}, Ratio {ratio}, seed: {generation_seed})...")
        synthetic_samples = vae_model.get_prior_samples(num_samples=num_synthetic_samples)
        synthetic_file = os.path.join(synthetic_dir, f"synthetic_original_level{level}_trial{trial}_ratio{ratio_str}.npz")
        np.savez_compressed(synthetic_file, synthetic_samples=synthetic_samples, norm_params=norm_params, norm_key=norm_key,
                            metadata={'num_samples': num_synthetic_samples, 'model_type': 'original_timeVAE', 'generation_method': 'prior_sampling',
                                      'level': level, 'trial': trial, 'ratio': ratio, 'generation_seed': generation_seed, 'num_features': synthetic_samples.shape[2]})
        print(f"  ✓ Synthetic data saved to {synthetic_file}")
        return synthetic_file
    except Exception as e:
        print(f"  ✗ Error generating synthetic data: {e}")
        return None

def train_original_timevae(processed_dir, save_dir, selected_level, selected_trials, selected_ratio,
                           norm_key='total_demand_clipped', generate_synthetic=True,
                           batch_size=10, rest_minutes=1):
    print("="*80)
    print("ORIGINAL TimeVAE Training Pipeline")
    print("="*80)
    total_start_time = time.time()
    train_dir = os.path.join(processed_dir, "train")
    if not os.path.exists(train_dir):
        print(f"Error: Train directory not found at {train_dir}")
        return
    os.makedirs(save_dir, exist_ok=True)
    analysis_dir = os.path.join(save_dir, "training_histories")
    models_dir = os.path.join(save_dir, "models")
    meta_dir = os.path.join(save_dir, "meta_histories")
    os.makedirs(analysis_dir, exist_ok=True); os.makedirs(models_dir, exist_ok=True); os.makedirs(meta_dir, exist_ok=True)
    all_results = []
    trial_batches = [selected_trials[i:i+batch_size] for i in range(0, len(selected_trials), batch_size)]
    num_batches = len(trial_batches)

    for batch_idx, trial_batch in enumerate(trial_batches, 1):
        print(f"\nBATCH {batch_idx}/{num_batches}: Trials {trial_batch[0]}-{trial_batch[-1]}")
        batch_start_time = time.time()
        for trial_in_batch, trial in enumerate(trial_batch, 1):
            print(f"\n[Batch {batch_idx}/{num_batches}] Trial {trial_in_batch}/{len(trial_batch)}: Trial {trial}")
            trial_start_time = time.time()
            trial_seed = 123 + selected_level * 1000 + trial * 100
            np.random.seed(trial_seed); tf.random.set_seed(trial_seed); random.seed(trial_seed)
            try:
                dataset_paths = get_matching_datasets(processed_dir, selected_level, trial)
                train_data = load_temporal_dataset(dataset_paths['train'])
                val_data = load_temporal_dataset(dataset_paths['val'])
                if train_data is None or val_data is None: continue
                train_x = train_data['data'].astype(np.float32)
                num_features = train_x.shape[2]; seq_len = train_x.shape[1]
                vae_model = TimeVAE(seq_len=seq_len, feat_dim=num_features, latent_dim=32, hidden_layer_sizes=[64, 128, 256],
                                    trend_poly=2, custom_seas=[(24, 1), (7, 24)], use_residual_conn=True, reconstruction_wt=3.0, batch_size=32)
                vae_model.fit_on_data(train_x, max_epochs=100, verbose=1)
                model_save_path = os.path.join(models_dir, f"original_timevae_level{selected_level}_trial{trial}")
                os.makedirs(model_save_path, exist_ok=True); vae_model.save(model_save_path)
                history_path = os.path.join(analysis_dir, f"history_original_level{selected_level}_trial{trial}.npz")
                class FakeHistory:
                    def __init__(self): self.history = {'loss': [], 'reconstruction_loss': [], 'kl_loss': []}
                save_training_history(FakeHistory(), history_path)
                meta_path = os.path.join(meta_dir, f"trial_meta_original_level{selected_level}_trial{trial}.npz")
                np.savez_compressed(meta_path, level=selected_level, trial=trial, seq_len=seq_len, num_features=num_features,
                                    latent_dim=32, training_seed=trial_seed, model_type='original_timeVAE')
                synthetic_files = {}
                if generate_synthetic:
                    num_synth = int(train_x.shape[0] * selected_ratio)
                    synthetic_file = generate_synthetic_data_original(vae_model, selected_level, trial, selected_ratio, 0, save_dir,
                                                                      train_x, train_data['norm_params'], norm_key, num_synthetic_samples=num_synth)
                    synthetic_files[selected_ratio] = synthetic_file
                all_results.append({'level': selected_level, 'trial': trial, 'time_minutes': (time.time() - trial_start_time)/60})
            except Exception as e:
                print(f"  ✗ Error in trial {trial}: {e}")
            finally: clear_memory()
        if batch_idx < num_batches:
            print(f"\n⏸️  Resting for {rest_minutes} minutes..."); time.sleep(rest_minutes * 60)

    ratio_str = ratio_to_suffix(selected_ratio)
    summary_path = os.path.join(meta_dir, f"training_summary_original_level{selected_level}_ratio{ratio_str}.npz")
    np.savez_compressed(summary_path, results=all_results, level=selected_level, ratio=selected_ratio)
    return all_results

if __name__ == "__main__":
    # Interactive Input
    print("\n" + "="*50)
    SELECTED_LEVEL = int(input("Enter aggregation level (e.g., 700): "))
    NUM_TRIALS = int(input("Enter number of trials to run (e.g., 1): "))
    print("="*50 + "\n")

    root = Path(__file__).resolve().parents[2]
    PROCESSED_DIR = root / "outputs" / "temporal_split_data"
    SAVE_DIR = root / "outputs" / "Time-VAE" / "original_timevae_results"

    available_trials = discover_available_trials(str(PROCESSED_DIR), SELECTED_LEVEL)
    if not available_trials:
        raise FileNotFoundError(
            f"No complete train/validation/test trials found for level {SELECTED_LEVEL} "
            f"under {PROCESSED_DIR}"
        )
    selected_trials = available_trials[:NUM_TRIALS]
    
    # Start timer
    overall_start_time = time.time()

    results = train_original_timevae(
        processed_dir=str(PROCESSED_DIR),
        save_dir=str(SAVE_DIR),
        selected_level=SELECTED_LEVEL,
        selected_trials=selected_trials,
        selected_ratio=1.0,
        norm_key='total_demand_clipped',
        generate_synthetic=True,
        batch_size=10,
        rest_minutes=0.5
    )

    # Calculate Total Simulation Time
    overall_end_time = time.time()
    total_seconds = overall_end_time - overall_start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    print("\n" + "="*80)
    print("ALL SIMULATIONS COMPLETE")
    print(f"⏱️  Total Execution Time: {hours} hours and {minutes} minutes")
    print(f"📂 Results saved to: {SAVE_DIR}")
    print("="*80)
