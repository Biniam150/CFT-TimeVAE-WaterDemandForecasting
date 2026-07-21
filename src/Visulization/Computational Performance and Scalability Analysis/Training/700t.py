import numpy as np
import tensorflow as tf
import os
import time
import re
import gc
import csv
import tracemalloc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CFT_VAE_SOURCE_DIR = REPO_ROOT / "src" / "CFT-VAE"
PROCESSED_DIR = REPO_ROOT / "outputs" / "temporal_split_data"
ABLATION_BASE = REPO_ROOT / "outputs" / "CFT-VAE" / "ablation"

if str(CFT_VAE_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(CFT_VAE_SOURCE_DIR))

from Model_definition import TimeVAE

try:
    import psutil
except ImportError:
    psutil = None


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
    if not os.path.exists(train_dir):
        return []

    return sorted([
        int(re.search(rf"level_{level}_trial_(\d+)", f.stem).group(1))
        for f in Path(train_dir).glob(f"train_preprocessed_level_{level}_trial_*.npz")
    ])


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


def get_process_memory_mb():
    """
    Returns current process memory in MB.
    Uses psutil if available. Otherwise returns None.
    """
    if psutil is None:
        return None

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_param_count(model):
    """
    Counts model parameters after the model has been built/trained.
    """
    try:
        return int(np.sum([np.prod(v.shape) for v in model.trainable_weights + model.non_trainable_weights]))
    except Exception:
        return None


def measure_inference_latency(vae_model, real_cond):
    """
    Measures decoder inference latency in seconds using one generated sample.
    """
    z_sample = np.random.normal(0, 1, (1, vae_model.latent_dim)).astype(np.float32)

    if vae_model.cond_dim > 0 and real_cond is not None:
        cond_sample = real_cond[:1].astype(np.float32)

        start_time = time.time()
        _ = vae_model.decoder.predict([z_sample, cond_sample], verbose=0)
        infer_latency = time.time() - start_time
    else:
        start_time = time.time()
        _ = vae_model.decoder.predict(z_sample, verbose=0)
        infer_latency = time.time() - start_time

    return infer_latency


def save_metrics_csv(metrics_records, save_path):
    """
    Saves raw computational metrics for all trials.
    """
    if len(metrics_records) == 0:
        return

    csv_path = os.path.join(save_path, "computational_metrics_raw1.csv")

    fieldnames = [
        "level",
        "trial",
        "cond_features",
        "train_time_sec",
        "param_count",
        "memory_mb",
        "infer_latency_sec"
    ]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_records)

    print(f"\n[Saved raw computational metrics to {csv_path}]", flush=True)


def print_metrics_summary(metrics_records):
    """
    Prints min, max, and mean summary for computational metrics.
    """
    if len(metrics_records) == 0:
        return

    metric_keys = [
        ("Train Time (sec)", "train_time_sec"),
        ("Param Count", "param_count"),
        ("Memory (MB)", "memory_mb"),
        ("Infer Latency", "infer_latency_sec")
    ]

    print("\n=== COMPUTATIONAL METRICS SUMMARY ===")
    print(f"{'':<20}{'Min':>10}{'Max':>10}{'Mean':>10}")

    for label, key in metric_keys:
        values = [record[key] for record in metrics_records if record[key] is not None]

        if len(values) == 0:
            print(f"{label:<20}{'N/A':>10}{'N/A':>10}{'N/A':>10}")
            continue

        values = np.array(values, dtype=np.float64)

        if key == "param_count":
            print(f"{label:<20}{values.min():>10.0f}{values.max():>10.0f}{values.mean():>10.0f}")
        else:
            print(f"{label:<20}{values.min():>10.2f}{values.max():>10.2f}{values.mean():>10.2f}")


def print_trial_raw_metrics(trial, train_time_sec, param_count, memory_mb, infer_latency_sec):
    """
    Prints raw computational metrics for one trial.
    """
    print(f"\n=== TRIAL {trial} RAW DATA ===")
    print(f"Train Time:    {train_time_sec:.2f} sec")

    if param_count is not None:
        print(f"Param Count:   {param_count}")
    else:
        print("Param Count:   N/A")

    if memory_mb is not None:
        print(f"Memory:        {memory_mb:.2f} MB")
    else:
        print("Memory:        N/A")

    print(f"Infer Latency: {infer_latency_sec:.4f} sec")


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
    print("  Enter nothing     -> Full model (all available conditional features)")
    print("  Enter: 1          -> Remove lags only")
    print("  Enter: 1,2        -> Remove lags and rolling")
    print("  Enter: 1,2,3,4,5  -> Unconditional model (cond_dim=0)")
    print("")

    exclude_input = input("Enter category IDs to exclude (comma separated, or Enter for none): ")
    exclude_cats = [int(x.strip()) for x in exclude_input.split(",")] if exclude_input.strip() else []

    mask_label = f"mask_{exclude_cats}"
    CURRENT_MASK_DIR = os.path.join(str(ABLATION_BASE), mask_label)
    SAVE_WEIGHTS_DIR = os.path.join(CURRENT_MASK_DIR, "weights")
    os.makedirs(SAVE_WEIGHTS_DIR, exist_ok=True)

    n_trials = int(input("Trials per level: "))

    metrics_records = []

    for sel_level in levels_to_run:
        print(f"\n>>> PROCESSING LEVEL {sel_level} <<<")
        # trials = discover_available_trials(data_dir, sel_level)[:n_trials]
        available_trials = discover_available_trials(data_dir, sel_level)
        trial_to_run = int(input("Enter specific trial number to run: "))
        trials = [trial_to_run]
        
        for trial in trials:
            print(f"--- Level {sel_level} | Trial {trial} ---", flush=True)

            data = load_temporal_dataset(os.path.join(data_dir, "train", f"train_preprocessed_level_{sel_level}_trial_{trial}.npz"))
            if data is None:
                continue

            feats = data['feature_names']
            keep_idx = get_feature_mask(feats, exclude_cats, norm_key)
            cond_idx = [i for i in keep_idx if i != feats.index(norm_key)]

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

            print(f"      Cond features: {len(cond_idx)} | Shape: {train_cond.shape if train_cond is not None else 'None'}", flush=True)

            # ---------------------------------------------------
            # Computational Metrics Start
            # ---------------------------------------------------
            tracemalloc.start()
            memory_before = get_process_memory_mb()

            train_start_time = time.time()
            vae.fit(ds, epochs=100, verbose=0)
            train_time_sec = time.time() - train_start_time

            memory_after = get_process_memory_mb()
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            if memory_after is not None:
                memory_mb = memory_after
            else:
                memory_mb = peak_mem / (1024 ** 2)

            param_count = get_param_count(vae)
            infer_latency_sec = measure_inference_latency(vae, train_cond)
            # ---------------------------------------------------
            # Computational Metrics End
            # ---------------------------------------------------

            print_trial_raw_metrics(
                trial=trial,
                train_time_sec=train_time_sec,
                param_count=param_count,
                memory_mb=memory_mb,
                infer_latency_sec=infer_latency_sec
            )

            metrics_records.append({
                "level": sel_level,
                "trial": trial,
                "cond_features": len(cond_idx),
                "train_time_sec": round(train_time_sec, 4),
                "param_count": param_count,
                "memory_mb": round(memory_mb, 4) if memory_mb is not None else None,
                "infer_latency_sec": round(infer_latency_sec, 6)
            })

            # Save inside mask-specific subfolder
            vae.save_weights(os.path.join(SAVE_WEIGHTS_DIR, f"weights_lvl{sel_level}_trial{trial}.h5"))
            generate_synthetic_data(vae, sel_level, trial, CURRENT_MASK_DIR, train_x, train_cond, data['norm_params'], norm_key)

            tf.keras.backend.clear_session()
            gc.collect()

    print_metrics_summary(metrics_records)
    save_metrics_csv(metrics_records, CURRENT_MASK_DIR)

    print(f"\nExperiment Complete. Data/Weights saved to: {CURRENT_MASK_DIR}")


if __name__ == "__main__":
    train_temporal_vae_ablation(str(PROCESSED_DIR))
