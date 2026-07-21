"""
Classical Augmentation — Synthetic Data Generation
====================================================
Generates synthetic training data using three classical augmentation methods:
  1. Gaussian Noise Injection  — adds data-driven noise (sigma from training std)
  2. Time Warping              — stretches/compresses sequences via interpolation
  3. Window Slicing            — random crop resampled back to original length

Follows the exact same pipeline and output format as CFT-VAE synthetic data:
  - Denormalized output saved as .npz with key 'synthetic_samples'
  - 1:1 ratio — number of synthetic windows equals number of real training windows
  - Saved under: AUGMENTATION_BASE / {method_name} / synthetic_data /
                 synthetic_lvl{level}_trial{trial}.npz

Input:  train_preprocessed_level_{level}_trial_{trial}.npz
Output: synthetic_lvl{level}_trial{trial}.npz  (per method, per trial)
"""

import numpy as np
import os
import re
from pathlib import Path
from scipy.interpolate import interp1d

# ==============================================================================
# CONFIGURATION
# ==============================================================================
REPO_ROOT          = Path(__file__).resolve().parents[4]
PROCESSED_DATA_DIR = REPO_ROOT / "outputs" / "temporal_split_data"
AUGMENTATION_BASE  = REPO_ROOT / "outputs" / "CFT-VAE" / "ablation"

NORM_KEY           = 'total_demand_clipped'
RANDOM_STATE       = 42

# Time warping — warp factor range (relative, data-driven)
WARP_FACTOR_RANGE  = (0.8, 1.2)   # stretch/compress by up to 20%

# Window slicing — minimum slice fraction of sequence length
SLICE_MIN_FRACTION = 0.7          # at least 70% of sequence retained


# ==============================================================================
# HELPERS — matching original pipeline
# ==============================================================================
def load_temporal_dataset(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        return {
            'data':         data['data'].astype(np.float32),
            'norm_params':  data['norm_params'].item(),
            'feature_names': list(data['feature_names'])
        }
    except Exception as e:
        print(f"    [!] Could not load {file_path}: {e}")
        return None


def discover_available_trials(base_dir, level):
    train_dir = os.path.join(base_dir, "train")
    if not os.path.exists(train_dir):
        return []
    return sorted([
        int(re.search(rf"level_{level}_trial_(\d+)", f.stem).group(1))
        for f in Path(train_dir).glob(f"train_preprocessed_level_{level}_trial_*.npz")
        if re.search(rf"level_{level}_trial_(\d+)", f.stem)
    ])


def denormalize(data_norm, norm_params, norm_key):
    """Denormalize from [0,1] back to real scale — matches CFT-VAE output."""
    min_val = norm_params[norm_key]['min']
    max_val = norm_params[norm_key]['max']
    return data_norm * (max_val - min_val + 1e-7) + min_val


def save_synthetic(synthetic_denorm, level, trial, method_dir):
    """Save synthetic data in same format as CFT-VAE generate_synthetic_data()."""
    target_dir = os.path.join(method_dir, "synthetic_data")
    os.makedirs(target_dir, exist_ok=True)
    fname = f"synthetic_lvl{level}_trial{trial}.npz"
    np.savez_compressed(
        os.path.join(target_dir, fname),
        synthetic_samples=synthetic_denorm
    )
    print(f"      [Saved] {os.path.join(target_dir, fname)}", flush=True)


# ==============================================================================
# AUGMENTATION METHOD 1 — Gaussian Noise Injection
# ==============================================================================
def gaussian_noise_augmentation(real_x_norm, norm_params, norm_key, rng, noise_factor=0.2):
    """
    Improved Gaussian Noise Augmentation
    - noise_factor: controls strength (0.1 = weak, 0.25 = medium, 0.4 = strong)
    - Optional temporal smoothing to preserve smoothness
    """
    min_val = norm_params[norm_key]['min']
    max_val = norm_params[norm_key]['max']
    
    # Denormalize
    real_denorm = denormalize(real_x_norm, norm_params, norm_key)
    
    # Data-driven sigma
    sigma = noise_factor * float(np.std(real_denorm))
    
    # Add noise
    noise = rng.normal(0, sigma, real_denorm.shape).astype(np.float32)
    synthetic = real_denorm + noise
    
    # Optional: Light temporal smoothing (recommended)
    # synthetic = gaussian_filter1d(synthetic, sigma=0.8, axis=1)  # if you import from scipy.ndimage
    
    # Clip
    synthetic = np.clip(synthetic, min_val, max_val)
    
    print(f"  → Gaussian Noise | factor={noise_factor} | σ={sigma:.2f} | "
          f"Mean: {synthetic.mean():.2f} | Std: {synthetic.std():.2f}")
    
    return synthetic


# ==============================================================================
# AUGMENTATION METHOD 2 — Time Warping
# ==============================================================================
def time_warp_sequence(sequence, warp_factor, rng):
    """
    Stretches or compresses a single sequence via cubic interpolation.
    sequence shape: (seq_len,)
    Returns warped sequence of same length.
    """
    seq_len   = len(sequence)
    # Original time axis
    t_orig    = np.linspace(0, 1, seq_len)
    # Warped time axis — compressed or stretched
    t_warped  = np.linspace(0, 1, int(seq_len * warp_factor))

    # Interpolate original onto warped grid
    interp_fn = interp1d(t_orig, sequence, kind='cubic',
                         fill_value='extrapolate')
    warped     = interp_fn(t_warped)

    # Resample back to original length
    t_back     = np.linspace(0, 1, len(warped))
    back_fn    = interp1d(t_back, warped, kind='cubic',
                          fill_value='extrapolate')
    resampled  = back_fn(t_orig)

    return resampled.astype(np.float32)


def time_warping_augmentation(real_x_norm, norm_params, norm_key, rng):
    """
    Applies random time warping to each window independently.
    Warp factor sampled uniformly from WARP_FACTOR_RANGE per window.
    Returns denormalized synthetic windows (shape: n_windows, seq_len, 1).
    """
    min_val    = norm_params[norm_key]['min']
    max_val    = norm_params[norm_key]['max']

    real_denorm = denormalize(real_x_norm, norm_params, norm_key)
    n_windows, seq_len, _ = real_denorm.shape
    synthetic  = np.zeros_like(real_denorm)

    for i in range(n_windows):
        warp_factor = rng.uniform(*WARP_FACTOR_RANGE)
        warped      = time_warp_sequence(real_denorm[i, :, 0], warp_factor, rng)
        synthetic[i, :, 0] = warped

    # Clip to physically plausible range
    synthetic = np.clip(synthetic, min_val, max_val)

    return synthetic   # shape: (n_windows, seq_len, 1)


# ==============================================================================
# AUGMENTATION METHOD 3 — Window Slicing
# ==============================================================================
def window_slicing_augmentation(real_x_norm, norm_params, norm_key, rng):
    """
    For each window, randomly slices a contiguous sub-segment of at least
    SLICE_MIN_FRACTION of the sequence length, then resamples back to the
    original length via linear interpolation.
    Returns denormalized synthetic windows (shape: n_windows, seq_len, 1).
    """
    min_val    = norm_params[norm_key]['min']
    max_val    = norm_params[norm_key]['max']

    real_denorm = denormalize(real_x_norm, norm_params, norm_key)
    n_windows, seq_len, _ = real_denorm.shape
    synthetic  = np.zeros_like(real_denorm)

    min_slice  = int(np.ceil(seq_len * SLICE_MIN_FRACTION))

    for i in range(n_windows):
        # Random slice length between min_slice and full seq_len
        slice_len  = rng.integers(min_slice, seq_len + 1)
        # Random start point
        max_start  = seq_len - slice_len
        start      = rng.integers(0, max_start + 1)
        sliced     = real_denorm[i, start:start + slice_len, 0]

        # Resample back to original length
        t_slice    = np.linspace(0, 1, len(sliced))
        t_orig     = np.linspace(0, 1, seq_len)
        interp_fn  = interp1d(t_slice, sliced, kind='linear',
                              fill_value='extrapolate')
        resampled  = interp_fn(t_orig)
        synthetic[i, :, 0] = resampled

    # Clip to physically plausible range
    synthetic = np.clip(synthetic, min_val, max_val)

    return synthetic   # shape: (n_windows, seq_len, 1)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    np.random.seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    print("=== CLASSICAL AUGMENTATION — SYNTHETIC DATA GENERATION ===\n")

    # -------------------------------------------------------
    # User inputs — matching original script style
    # -------------------------------------------------------
    levels_input  = input("Enter levels (comma separated, e.g. 700 or 300,700,1000): ")
    levels_to_run = [int(lvl.strip()) for lvl in levels_input.split(",")]

    print("\nMethods to generate:")
    print("  1: Gaussian Noise Injection")
    print("  2: Time Warping")
    print("  3: Window Slicing")
    print("  Enter nothing → run all three methods")
    methods_input = input("Enter method IDs (comma separated, or Enter for all): ")

    if methods_input.strip():
        selected_ids = [int(x.strip()) for x in methods_input.split(",")]
    else:
        selected_ids = [1, 2, 3]

    method_map = {
        1: ("gaussian_noise",  gaussian_noise_augmentation),
        2: ("time_warping",    time_warping_augmentation),
        3: ("window_slicing",  window_slicing_augmentation),
    }

    selected_methods = [(method_map[i]) for i in selected_ids if i in method_map]

    print(f"\nSelected methods: {[name for name, _ in selected_methods]}")
    print(f"Levels to run:    {levels_to_run}\n")

    # -------------------------------------------------------
    # Generation loop
    # -------------------------------------------------------
    for sel_level in levels_to_run:
        print(f"\n>>> PROCESSING LEVEL {sel_level} <<<")

        trials = discover_available_trials(PROCESSED_DATA_DIR, sel_level)
        print(f"  Found {len(trials)} trials: {trials}\n")

        for trial in trials:
            print(f"--- Level {sel_level} | Trial {trial} ---", flush=True)

            # Load real training data
            train_path = os.path.join(
                PROCESSED_DATA_DIR, "train",
                f"train_preprocessed_level_{sel_level}_trial_{trial}.npz"
            )
            dataset = load_temporal_dataset(train_path)
            if dataset is None:
                continue

            # Extract demand signal — normalized, shape (n_windows, seq_len, 1)
            # Matches: train_x = X[:, :, [feats.index(norm_key)]] in original
            feats    = dataset['feature_names']
            norm_idx = feats.index(NORM_KEY)
            X        = dataset['data'][:-1].astype(np.float32)
            real_x   = X[:, :, [norm_idx]]   # (n_windows, 24, 1)

            norm_params = dataset['norm_params']
            n_windows   = real_x.shape[0]

            print(f"      Windows: {n_windows} | Seq len: {real_x.shape[1]}", flush=True)

            # Generate and save for each method
            for method_name, aug_fn in selected_methods:
                method_dir = os.path.join(AUGMENTATION_BASE, method_name)

                synthetic_denorm = aug_fn(real_x, norm_params, NORM_KEY, rng)

                print(
                    f"      [{method_name}] "
                    f"Generated {synthetic_denorm.shape[0]} windows | "
                    f"Mean: {synthetic_denorm.mean():.2f} | "
                    f"Std: {synthetic_denorm.std():.2f}",
                    flush=True
                )

                save_synthetic(synthetic_denorm, sel_level, trial, method_dir)

        print(f"\n>>> Level {sel_level} complete <<<")

    print("\n=== All methods complete. ===")
    print(f"Outputs saved under: {AUGMENTATION_BASE}")
    print("Folders created:")
    for method_name, _ in selected_methods:
        print(f"  {os.path.join(AUGMENTATION_BASE, method_name, 'synthetic_data')}")


if __name__ == "__main__":
    main()
