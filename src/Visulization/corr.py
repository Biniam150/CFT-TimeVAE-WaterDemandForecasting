import numpy as np
import pandas as pd
import os
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROCESSED_DIR = r"C:\Users\bin150\OneDrive - UBC\Desktop\Publication\WR2\cft-vae\data"
LEVELS = [1, 20, 70, 200, 300, 500, 700]
NUM_TRIALS = 100 
TARGET_COL = 'total_demand_clipped'

def calculate_lag_correlation(data, lag=24):
    """Calculates autocorrelation with a guard against zero variance."""
    if len(data) <= lag: 
        return np.nan
    
    # Check for zero variance to avoid RuntimeWarning: divide by zero
    if np.std(data) < 1e-9:
        return 0.0
        
    try:
        corr = np.corrcoef(data[lag:], data[:-lag])[0, 1]
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0

def process_baseline_metrics():
    print(f"Analyzing {len(LEVELS)} levels across {NUM_TRIALS} trials...")
    all_summary_stats = []

    for level in LEVELS:
        trial_cvs, trial_corrs, trial_means, trial_windows = [], [], [], []
        valid_count = 0
        
        for trial in range(1, NUM_TRIALS + 1):
            f_path = os.path.join(PROCESSED_DIR, "train", f"train_preprocessed_level_{level}_trial_{trial}.npz")
            if not os.path.exists(f_path): continue
                
            try:
                data_obj = np.load(f_path, allow_pickle=True)
                feat_names = list(data_obj['feature_names'])
                
                if TARGET_COL not in feat_names: continue
                
                target_idx = feat_names.index(TARGET_COL)
                demand_norm = data_obj['data'][:, :, target_idx].flatten()
                
                # REPRODUCIBLE DENORMALIZATION
                norm_params = data_obj['norm_params'].item()[TARGET_COL]
                actual_demand = demand_norm * (norm_params['max'] - norm_params['min'] + 1e-7) + norm_params['min']
                
                m = np.mean(actual_demand)
                s = np.std(actual_demand)
                
                # Guard against division by zero in CV
                trial_cvs.append(s / m if m > 1e-9 else 0.0)
                trial_corrs.append(calculate_lag_correlation(actual_demand, 24))
                trial_means.append(m)
                trial_windows.append(data_obj['data'].shape[0])
                valid_count += 1
                
            except Exception:
                continue

        if valid_count > 0:
            # Aggregate stats using nanmean to ignore any remaining issues
            all_summary_stats.append({
                'Aggregation Level': level,
                'Num Windows': int(np.mean(trial_windows)),
                'CV (mean ± std)': f"{np.nanmean(trial_cvs):.3f} ± {np.nanstd(trial_cvs):.3f}",
                'Lag-24 Corr (mean ± std)': f"{np.nanmean(trial_corrs):.3f} ± {np.nanstd(trial_corrs):.3f}",
                'Demand (mean ± std)': f"{np.nanmean(trial_means):.2f} ± {np.nanstd(trial_means):.2f}",
                'Num Trials': valid_count
            })

    return pd.DataFrame(all_summary_stats)

if __name__ == "__main__":
    results_table = process_baseline_metrics()
    print("\n" + "="*110)
    print("TABLE I: STATISTICAL CHARACTERISTICS OF WATER DEMAND (FIXED)")
    print("="*110)
    print(results_table.to_string(index=False))
    print("="*110)