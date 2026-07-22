import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# ================================================================
# 1. Load & Filter
# ================================================================
def load_and_filter_data(file_path: str, min_records_per_user: int = 10000) -> pd.DataFrame:
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['datetime'])
    except Exception as e:
        print(f"Error loading file: {e}")
        return pd.DataFrame()

    df = df[df['diff'] >= 0].copy()
    counts = df['user key'].value_counts()
    keep_users = counts[counts >= min_records_per_user].index
    df = df[df['user key'].isin(keep_users)].copy()

    print(f"Total users after filtering: {len(keep_users)}")
    return df

# ================================================================
# 2. Temporal Splits
# ================================================================
def create_temporal_splits_simple(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp("2016-09-30 23:59:59")
    val_end = pd.Timestamp("2016-12-31 23:59:59")

    train_df = df[df["datetime"] <= train_end].copy()
    val_df = df[(df["datetime"] > train_end) & (df["datetime"] <= val_end)].copy()
    test_df = df[df["datetime"] > val_end].copy()
    
    return train_df, val_df, test_df

# ================================================================
# 3. Feature Engineering (Matches your Local Scenario B)
# ================================================================
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lags = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in lags:
        df[f"lag_{lag}h"] = df["total_demand_clipped"].shift(lag)

    df["rolling_mean_6h"] = df["total_demand_clipped"].rolling(6, min_periods=1).mean()
    df["rolling_mean_12h"] = df["total_demand_clipped"].rolling(12, min_periods=1).mean()
    df["rolling_mean_24h"] = df["total_demand_clipped"].rolling(24, min_periods=1).mean()
    df["rolling_std_6h"] = df["total_demand_clipped"].rolling(6, min_periods=1).std()
    df["rolling_std_24h"] = df["total_demand_clipped"].rolling(24, min_periods=1).std()

    df["hourly_change"] = df["total_demand_clipped"].diff()
    df["daily_change"] = df["total_demand_clipped"].diff(24)
    df["weekly_change"] = df["total_demand_clipped"].diff(168)
    return df.fillna(0)

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["date"] = df.index.date
    daily_stats = df.groupby("date")["total_demand_clipped"].agg(["sum", "max", "mean", "std"])
    daily_stats.columns = ["daily_sum", "daily_max", "daily_mean", "daily_std"]
    df["cumsum_day"] = df.groupby("date")["total_demand_clipped"].cumsum()
    df = df.join(daily_stats, on="date")
    df["pct_of_daily"] = df["cumsum_day"] / (df["daily_sum"] + 1e-7)
    df["is_peak_hour"] = (df["total_demand_clipped"] > 0.9 * df["daily_max"]).astype(int)
    df["relative_to_peak"] = df["total_demand_clipped"] / (df["daily_max"] + 1e-7)
    df["hours_from_noon"] = np.abs(df["hour"] - 12)
    df["hours_from_midnight"] = np.minimum(df["hour"], 24 - df["hour"])
    return df.drop(["hour", "date", "daily_sum", "daily_max", "daily_mean", "daily_std"], axis=1)

def add_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cv_6h"] = df["rolling_std_6h"] / (df["rolling_mean_6h"] + 1e-7)
    df["cv_24h"] = df["rolling_std_24h"] / (df["rolling_mean_24h"] + 1e-7)
    df["pattern_stability"] = 1 / (1 + df["rolling_std_24h"])
    df["deviation_6h"] = (df["total_demand_clipped"] - df["rolling_mean_6h"]) / (df["rolling_std_6h"] + 1e-7)
    df["deviation_24h"] = (df["total_demand_clipped"] - df["rolling_mean_24h"]) / (df["rolling_std_24h"] + 1e-7)
    return df.replace([np.inf, -np.inf], 0)

def add_time_features_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_month"] = df.index.day

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df = pd.concat([df,
                    pd.get_dummies(df["day_of_week"], prefix="day_of_week").reindex(columns=[f"day_of_week_{i}" for i in range(7)], fill_value=0),
                    pd.get_dummies(df["month"], prefix="month").reindex(columns=[f"month_{i}" for i in range(1,13)], fill_value=0)], axis=1)
    
    return df.drop(["hour", "day_of_week", "month", "day_of_month"], axis=1, errors='ignore')

# ================================================================
# 4. Aggregation & Normalization
# ================================================================
def aggregate_users_demand(df: pd.DataFrame, selected_users: List) -> pd.DataFrame:
    user_df = df[df["user key"].isin(selected_users)].copy()
    user_df["dt"] = user_df["datetime"].dt.floor("h")
    agg = user_df.groupby("dt").agg(total_demand=("diff", "sum"),
                                    active_users=("user key", "nunique")).sort_index()

    full_index = pd.date_range(start=agg.index.min(), end=agg.index.max(), freq="h")
    agg = agg.reindex(full_index).fillna({"total_demand": 0, "active_users": 0})

    q_low, q_high = agg["total_demand"].quantile([0.001, 0.999])
    agg["total_demand_clipped"] = agg["total_demand"].clip(lower=q_low, upper=q_high)

    agg = add_time_features_enhanced(agg)
    agg = add_lag_features(agg)
    agg = add_context_features(agg)
    agg = add_stability_features(agg)
    return agg

def save_datasets_to_files(all_datasets: Dict, save_dir: str, split_name: str, feature_columns: List[str]):
    os.makedirs(save_dir, exist_ok=True)
    for dataset_key, dataset in all_datasets.items():
        agg_data_subset = dataset["data"][feature_columns].astype(float)
        
        # Per-trial Normalization
        trial_norm_params = {col: {"min": float(agg_data_subset[col].min()), 
                                   "max": float(agg_data_subset[col].max())} for col in feature_columns}

        normed_data = agg_data_subset.copy()
        for col in feature_columns:
            mi, ma = trial_norm_params[col]["min"], trial_norm_params[col]["max"]
            normed_data[col] = (normed_data[col] - mi) / (ma - mi + 1e-7) if (ma - mi) > 1e-7 else 0.0

        num_windows = len(normed_data) // 24
        dataX = normed_data.values[:num_windows*24].reshape(num_windows, 24, len(feature_columns))
        data_index = normed_data.index[:num_windows*24].to_numpy()

        # FILENAME MATCH: split_preprocessed_level_X_trial_Y.npz
        save_path = os.path.join(save_dir, f"{split_name}_preprocessed_{dataset_key}.npz")
        np.savez(save_path, data=dataX, norm_params=trial_norm_params,
                 feature_names=feature_columns, index=data_index,
                 metadata={"users": dataset["users"], "level": dataset["level"], "trial": dataset["trial"]})

# ================================================================
# 5. Main Path-Independent Execution
# ================================================================
def main():
    # --- DYNAMIC PATHS FOR GITHUB ---
    try:
        # This script is stored under src/Aggregation&Temporal_Preprocessing.
        root = Path(__file__).resolve().parents[2]
    except NameError:
        # If running in Jupyter Notebook
        root = Path(os.getcwd()).resolve()

    RAW_DATA_PATH = root / "data" / "Processed_data" / "Good_Data.csv"
    SAVE_BASE_DIR = root / "outputs" / "temporal_split_data"

    print(f"Working Directory: {root}")
    print(f"Input: {RAW_DATA_PATH}")
    print(f"Output: {SAVE_BASE_DIR}")

    df = load_and_filter_data(str(RAW_DATA_PATH))
    if df.empty: return

    train_df, val_df, test_df = create_temporal_splits_simple(df)
    
    
    selected_level = 20  # Rerun with a different aggregation level if needed.
    n_trials = 100 

    available_users = df["user key"].value_counts().index.tolist()

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"Processing {split_name}...")
        split_datasets = {}
        for t in range(n_trials):
            # Seed matches your local deterministic selection
            np.random.seed(42 + (selected_level * 1000) + t)
            users = np.random.choice(available_users, size=selected_level, replace=False).tolist()
            
            agg_data = aggregate_users_demand(split_df, users)
            split_datasets[f"level_{selected_level}_trial_{t+1}"] = {
                "data": agg_data, "level": selected_level, "users": users, "trial": t+1
            }
        
        feature_cols = [c for c in list(split_datasets.values())[0]["data"].columns if c != "user_list"]
        save_datasets_to_files(split_datasets, str(SAVE_BASE_DIR / split_name), split_name, feature_cols)

    print("\n✓ Pipeline sync complete.")

if __name__ == "__main__":
    main()
