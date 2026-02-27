import pandas as pd
import numpy as np
import os

# --- 1. FIND FILE (HIC 2026 Naming Convention) ---
def find_forecasting_results(results_dir, level, ratio=1.0):
    """Finds results_level{X}_ratio{Y}.csv in the HIC 2026 folder."""
    ratio_str = str(ratio).replace(".", "p")
    filename = f"results_level{level}_ratio{ratio_str}.csv"
    filepath = os.path.join(results_dir, filename)
    
    if os.path.exists(filepath):
        return filepath
    return None

# --- 2. ROBUST MEAN DEMAND (Smart Column Lookup) ---
def get_mean_demand(processed_dir, level, trial):
    """
    Extracts mean demand from real training data.
    Uses 'feature_names' to find the exact column index (fixes Level 100+ issue).
    """
    train_path = os.path.join(
        processed_dir, "train",
        f"train_preprocessed_level_{level}_trial_{trial}.npz"
    )
    
    if not os.path.exists(train_path):
        return None
    
    try:
        data = np.load(train_path, allow_pickle=True)
        train_data = data['data']
        norm_params = data['norm_params'].item()
        feature_names = data['feature_names']
        target_col = 'total_demand_clipped'
        
        # Smart Lookup: Find where demand is hiding
        if target_col in feature_names:
            idx = np.where(feature_names == target_col)[0][0]
            demand_normalized = train_data[:, :, idx]
        else:
            # Fallback for Level 30 if names missing
            demand_normalized = train_data[:, :, 0]

        # Denormalize
        norm_info = norm_params.get(target_col, {})
        if 'min' in norm_info and 'max' in norm_info:
            demand_actual = demand_normalized * (norm_info['max'] - norm_info['min'] + 1e-7) + norm_info['min']
        else:
            demand_actual = demand_normalized
        
        return np.mean(demand_actual)
    except:
        return None

# --- 3. CALCULATE SUCCESS RATE ---
def analyze_success_rates(csv_file):
    """
    Counts how many times Augmented RMSE < Baseline RMSE.
    """
    df = pd.read_csv(csv_file)
    models = sorted(df['model'].unique())
    trials = sorted(df['trial'].unique())
    
    results = []

    for model in models:
        model_data = df[df['model'] == model]
        
        wins = 0
        total_valid_trials = 0
        
        # Check every trial individually
        for trial in trials:
            # Get the row for 'augmented' training (which contains both baseline & aug RMSE)
            trial_row = model_data[
                (model_data['trial'] == trial) & 
                (model_data['training_type'] == 'augmented')
            ]
            
            if len(trial_row) == 1:
                row = trial_row.iloc[0]
                base_rmse = row['baseline_rmse']
                aug_rmse = row['result_rmse']
                
                # Verify data exists
                if pd.notna(base_rmse) and pd.notna(aug_rmse):
                    total_valid_trials += 1
                    
                    # SUCCESS CONDITION: Augmented Error < Baseline Error
                    if aug_rmse < base_rmse:
                        wins += 1
        
        if total_valid_trials > 0:
            win_rate = (wins / total_valid_trials) * 100
            results.append({
                'Level': level,
                'Model': model,
                'Wins': wins,
                'Total': total_valid_trials,
                'Success_Rate': win_rate
            })
            
    return pd.DataFrame(results)

# --- 4. MAIN ---
if __name__ == "__main__":
    # Correct HIC 2026 Paths
    PROCESSED_DATA_DIR = r"C:\Users\bin150\OneDrive - UBC\Desktop\Publication\WR2\cft-vae\data"
    RESULTS_DIR = r"C:\Users\bin150\OneDrive - UBC\Desktop\Publication\WR2\cft-vae\forecasting_results"
    
    LEVELS = [20, 70, 200, 700]
    RATIO = 1.0
    
    print(f"\n{'Level':<6} | {'Model':<18} | {'Win Count':<10} | {'Success Rate':<12}")
    print("-" * 60)
    
    for level in LEVELS:
        # 1. Find the CSV
        file = find_forecasting_results(RESULTS_DIR, level, RATIO)
        
        if file:
            # 2. Analyze Success
            df = analyze_success_rates(file)
            
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    print(f"{row['Level']:<6} | {row['Model']:<18} | {row['Wins']}/{row['Total']:<8} | {row['Success_Rate']:.1f}%")
        else:
            print(f"{level:<6} | {'(File Not Found)':<18} | -          | -")