import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import ks_2samp

# =============================================================================
# CONFIGURATION
# =============================================================================
root = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIR = root / "outputs" / "temporal_split_data"
SYNTHETIC_DATA_DIR = root / "outputs" / "CFT-VAE" / "synthetic_data" / "ratio_1p0"
SAVE_DIR = root / "outputs" / "Visualization" / "Direct Fidelity Assessment"

LEVELS = [20, 70, 200, 700]
NORM_KEY = "total_demand_clipped"

ACF_MAX_LAGS = 168
PSD_NPERSEG = 168
FS = 1.0
EXTREME_HIGH_PCT = 95

os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# LOAD DATA
# =============================================================================
def load_real_demand(level, trial):
    path = os.path.join(
        PROCESSED_DATA_DIR,
        "train",
        f"train_preprocessed_level_{level}_trial_{trial}.npz"
    )

    if not os.path.exists(path):
        return None

    try:
        data = np.load(path, allow_pickle=True)
        norm_params = data["norm_params"].item()
        feat_names = list(data["feature_names"])

        idx = feat_names.index(NORM_KEY)
        raw = data["data"][:-1, :, idx].astype(np.float64)

        norm_p = norm_params[NORM_KEY]
        denorm = raw * (norm_p["max"] - norm_p["min"] + 1e-7) + norm_p["min"]

        return denorm.flatten()

    except Exception:
        return None


def load_synthetic_demand(level, trial):
    path = os.path.join(
        SYNTHETIC_DATA_DIR,
        f"synthetic_level{level}_trial{trial}_ratio1p0.npz"
    )

    if not os.path.exists(path):
        return None

    try:
        data = np.load(path, allow_pickle=True)
        return data["synthetic_samples"].astype(np.float64).flatten()

    except Exception:
        return None


def discover_trials(level):
    train_dir = os.path.join(PROCESSED_DATA_DIR, "train")

    trials = []
    for f in Path(train_dir).glob(f"train_preprocessed_level_{level}_trial_*.npz"):
        match = re.search(rf"level_{level}_trial_(\d+)", f.stem)
        if match:
            trials.append(int(match.group(1)))

    return sorted(trials)


# =============================================================================
# METRICS
# =============================================================================
def compute_acf(x, max_lags):
    x = x - np.mean(x)
    var = np.var(x)

    if var == 0:
        return np.zeros(max_lags + 1)

    full = np.correlate(x, x, mode="full")
    acf = full[len(x) - 1:len(x) + max_lags] / (var * len(x))

    return acf


def acf_mae(real, synth):
    acf_real = compute_acf(real, ACF_MAX_LAGS)
    acf_synth = compute_acf(synth, ACF_MAX_LAGS)

    return np.mean(np.abs(acf_real - acf_synth))


def psd_log_mae(real, synth):
    _, psd_real = welch(real, fs=FS, nperseg=PSD_NPERSEG)
    _, psd_synth = welch(synth, fs=FS, nperseg=PSD_NPERSEG)

    eps = 1e-10
    return np.mean(np.abs(np.log(psd_real + eps) - np.log(psd_synth + eps)))


def ks_peak(real, synth):
    real_peaks = real[real >= np.percentile(real, EXTREME_HIGH_PCT)]
    synth_peaks = synth[synth >= np.percentile(synth, EXTREME_HIGH_PCT)]

    ks_value, _ = ks_2samp(real_peaks, synth_peaks)
    return ks_value


def compute_trial_metrics(level, trial):
    real = load_real_demand(level, trial)
    synth = load_synthetic_demand(level, trial)

    if real is None or synth is None:
        return None

    return {
        "ACF MAE": acf_mae(real, synth),
        "PSD Log-MAE": psd_log_mae(real, synth),
        "KS Peak": ks_peak(real, synth),
        "Mean Ratio": np.mean(synth) / (np.mean(real) + 1e-7),
        "Std Ratio": np.std(synth) / (np.std(real) + 1e-7),
    }


# =============================================================================
# MAIN TABLE
# =============================================================================
rows = []

for level in LEVELS:
    print(f"Processing level {level}...")

    trials = discover_trials(level)
    records = []

    for trial in trials:
        metrics = compute_trial_metrics(level, trial)
        if metrics is not None:
            records.append(metrics)

    df_level = pd.DataFrame(records)

    if df_level.empty:
        print(f"  No matching real and synthetic trials found for level {level}; skipping.")
        continue

    row = {
        "Level": level,
        "N Trials": len(df_level),
        "ACF MAE": f"{df_level['ACF MAE'].mean():.2f} ± {df_level['ACF MAE'].std():.2f}",
        "PSD Log-MAE": f"{df_level['PSD Log-MAE'].mean():.2f} ± {df_level['PSD Log-MAE'].std():.2f}",
        "KS Peak": f"{df_level['KS Peak'].mean():.2f} ± {df_level['KS Peak'].std():.2f}",
        "Mean Ratio": f"{df_level['Mean Ratio'].mean():.2f} ± {df_level['Mean Ratio'].std():.2f}",
        "Std Ratio": f"{df_level['Std Ratio'].mean():.2f} ± {df_level['Std Ratio'].std():.2f}",
    }

    rows.append(row)

summary_table = pd.DataFrame(rows)

print("\nTable 4. Direct Fidelity Assessment of CFT-VAE Synthetic Demand Sequences")
print("Across Aggregation Levels (Mean ± Standard Deviation Over 100 Trials)")
print("=" * 100)
print(summary_table.to_string(index=False))
print("=" * 100)

save_path = os.path.join(SAVE_DIR, "reported_fidelity_summary_table.csv")
summary_table.to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"\nSaved table to: {save_path}")
