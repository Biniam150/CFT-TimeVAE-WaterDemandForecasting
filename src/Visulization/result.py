import pandas as pd
import numpy as np
import os

# ==============================================================================
# WR2 CONFIGURATION
# ==============================================================================
PROCESSED_DATA_DIR = r"C:\Users\bin150\OneDrive - UBC\Desktop\Publication\WR2\cft-vae\data"
RESULTS_DIR = r"C:\Users\bin150\OneDrive - UBC\Desktop\Publication\WR2\cft-vae\forecasting_results"

LEVELS = [20, 70, 200, 700] # Ensure 200 is in the list
RATIO = 1.0

def get_mean_demand(processed_dir, level, trial=1):
    train_path = os.path.join(processed_dir, "train", f"train_preprocessed_level_{level}_trial_{trial}.npz")
    if not os.path.exists(train_path): return None
    try:
        data = np.load(train_path, allow_pickle=True)
        # Use your feature_names logic to be safe
        feat_names = list(data['feature_names'])
        idx = feat_names.index('total_demand_clipped')
        norm_params = data['norm_params'].item()['total_demand_clipped']
        
        val_norm = data['data'][:, :, idx]
        val_real = val_norm * (norm_params['max'] - norm_params['min'] + 1e-7) + norm_params['min']
        return np.mean(val_real)
    except: return None

def run_analysis():
    print("\n" + "="*120)
    print("WR2 PUBLICATION: CFT-VAE FORECASTING EVALUATION")
    print("="*120)

    for level in LEVELS:
        ratio_str = str(RATIO).replace(".", "p")
        fname = f"results_level{level}_ratio{ratio_str}.csv"
        fpath = os.path.join(RESULTS_DIR, fname)

        if not os.path.exists(fpath):
            # This is why levels might be missing
            print(f"\n[!] Skipping Level {level}: File {fname} not found in results directory.")
            continue

        df = pd.read_csv(fpath)
        mean_dem = get_mean_demand(PROCESSED_DATA_DIR, level)
        if mean_dem is None: continue

        print(f"\nExtracting mean demand for Level {level}...")
        print(f"Mean demand (training data): {mean_dem:.2f} m³/h")

        models = sorted(df['model'].unique())
        summary_rows = []

        for model in models:
            m_df = df[df['model'] == model]
            
            # Baseline is always available in your 'augmented' rows as 'baseline_rmse'
            base_nrmse = m_df['baseline_rmse'].values / mean_dem
            
            # Synthetic Only - checking if 'synthetic_only' exists in training_type
            s_df = m_df[m_df['training_type'] == 'synthetic_only']
            
            # Augmented
            a_df = m_df[m_df['training_type'] == 'augmented']

            # Stats
            b_m, b_s = np.mean(base_nrmse), np.std(base_nrmse)
            
            if len(s_df) > 0:
                s_nrmse = s_df['result_rmse'].values / mean_dem
                s_m, s_s = np.mean(s_nrmse), np.std(s_nrmse)
                s_imp = f"{((b_m - s_m)/b_m*100):+.2f}%"
            else:
                s_m, s_s, s_imp = np.nan, np.nan, "N/A"

            if len(a_df) > 0:
                a_nrmse = a_df['result_rmse'].values / mean_dem
                a_m, a_s = np.mean(a_nrmse), np.std(a_nrmse)
                a_imp = f"{((b_m - a_m)/b_m*100):+.2f}%"
            else:
                a_m, a_s, a_imp = np.nan, np.nan, "N/A"

            print("-" * 120)
            print(f"Model: {model.upper()}")
            print(f"  BASELINE       nRMSE: {b_m:.4f} ± {b_s:.4f}")
            print(f"  SYNTH_ONLY     nRMSE: {f'{s_m:.4f} ± {s_s:.4f}' if not np.isnan(s_m) else 'N/A'}")
            print(f"  AUGMENTED      nRMSE: {f'{a_m:.4f} ± {a_s:.4f}' if not np.isnan(a_m) else 'N/A'}")
            print(f"  Aug Improvement: {a_imp}")

            summary_rows.append([model, f"{b_m:.4f}±{b_s:.4f}", 
                                f"{s_m:.4f}±{s_s:.4f}" if not np.isnan(s_m) else "N/A", 
                                f"{a_m:.4f}±{a_s:.4f}", s_imp, a_imp])

        print("\n" + "="*120)
        print(f"SUMMARY TABLE - nRMSE RESULTS (LEVEL {level})")
        print("="*120)
        print(f"{'Model':<20} | {'Baseline':<18} | {'Synthetic-Only':<18} | {'Augmented':<18} | {'Synth Improv':<15} | {'Aug Improv':<15}")
        print("-" * 120)
        for row in summary_rows:
            print(f"{row[0]:<20} | {row[1]:<18} | {row[2]:<18} | {row[3]:<18} | {row[4]:<15} | {row[5]:<15}")

if __name__ == "__main__":
    run_analysis()