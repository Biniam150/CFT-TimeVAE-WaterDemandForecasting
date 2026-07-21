import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import re
import warnings

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
root = Path(__file__).resolve().parents[2]
PROCESSED_DIR = root / "outputs" / "temporal_split_data"
SAVE_DIR = root / "outputs" / "Visualization"

os.makedirs(SAVE_DIR, exist_ok=True)

LEVELS = [1, 10, 20, 50, 70, 100, 200, 300, 500, 700, 800]
TARGET_COL = "total_demand_clipped"
NUM_TRIALS = 100

# =============================================================================
# HELPERS
# =============================================================================
def load_real_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    return {
        "data": data["data"].astype(np.float32),
        "norm_params": data["norm_params"].item(),
        "feature_names": list(data["feature_names"]),
    }


def denormalize_data(data, norm_params):
    norm_info = norm_params[TARGET_COL]
    return data * (norm_info["max"] - norm_info["min"] + 1e-7) + norm_info["min"]


def extract_target(dataset):
    """Extract and denormalize total_demand_clipped by feature name."""
    if TARGET_COL not in dataset["feature_names"]:
        raise KeyError(f"Feature '{TARGET_COL}' not found in processed dataset")
    target_idx = dataset["feature_names"].index(TARGET_COL)
    normalized_target = dataset["data"][:, :, target_idx]
    return denormalize_data(normalized_target, dataset["norm_params"])


def create_features_and_targets(time_series_data):
    flattened = (
        time_series_data.flatten()
        if len(time_series_data.shape) == 3
        else np.asarray(time_series_data).flatten()
    )

    df = pd.DataFrame({"demand": flattened})

    for lag in [1, 2, 3, 24, 25, 48, 168]:
        df[f"lag_{lag}"] = df["demand"].shift(lag)

    for window in [6, 12, 24]:
        df[f"rolling_mean_{window}h"] = (
            df["demand"].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}h"] = (
            df["demand"].shift(1).rolling(window=window, min_periods=1).std()
        )

    df["hour"] = df.index % 24
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    combined = df.dropna()
    X = combined.drop(["demand"], axis=1)
    y = combined["demand"]

    return X, y


# =============================================================================
# 1. RUN GRADIENT BOOSTING BASELINE AND SAVE FULL RMSE RESULTS
# =============================================================================
def run_baseline_trials():
    print("=== Running Gradient Boosting Baseline ===")

    all_trials = []

    for level in LEVELS:
        print(f"\nProcessing Level {level}")

        train_dir = os.path.join(PROCESSED_DIR, "train")
        train_files = list(Path(train_dir).glob(f"train_preprocessed_level_{level}_trial_*.npz"))

        available_trials = sorted([
            int(re.search(r"trial_(\d+)", f.stem).group(1))
            for f in train_files
        ])

        selected_trials = available_trials[:NUM_TRIALS]

        level_rmse = []

        for trial in selected_trials:
            train_path = os.path.join(
                PROCESSED_DIR,
                "train",
                f"train_preprocessed_level_{level}_trial_{trial}.npz"
            )

            test_path = os.path.join(
                PROCESSED_DIR,
                "test",
                f"test_preprocessed_level_{level}_trial_{trial}.npz"
            )

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                continue

            train_data = load_real_data(train_path)
            test_data = load_real_data(test_path)

            train_real = extract_target(train_data)
            test_real = extract_target(test_data)

            X_train, y_train = create_features_and_targets(train_real)
            X_test, y_test = create_features_and_targets(test_real)

            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            level_rmse.append(rmse)

            all_trials.append({
                "Level": level,
                "Trial": trial,
                "RMSE": rmse
            })

        if level_rmse:
            print(
                f"  Mean RMSE: {np.mean(level_rmse):.3f} "
                f"+/- {np.std(level_rmse):.3f} "
                f"({len(level_rmse)} trials)"
            )

    df_full = pd.DataFrame(all_trials)

    full_csv_path = os.path.join(SAVE_DIR, "baseline_full_trials_results.csv")
    df_full.to_csv(full_csv_path, index=False)

    print(f"\nFull trial results saved to: {full_csv_path}")
    return df_full


# =============================================================================
# 2. COMPUTE MEAN DEMAND PER LEVEL
# =============================================================================
def calculate_mean_demand():
    mean_demands = {}

    print("\nCalculating Mean Demand per level...")

    for level in LEVELS:
        demands = []

        for trial in range(1, NUM_TRIALS + 1):
            f_path = os.path.join(
                PROCESSED_DIR,
                "train",
                f"train_preprocessed_level_{level}_trial_{trial}.npz"
            )

            if not os.path.exists(f_path):
                continue

            try:
                data_obj = np.load(f_path, allow_pickle=True)

                feat_names = list(data_obj["feature_names"])
                if TARGET_COL not in feat_names:
                    continue

                target_idx = feat_names.index(TARGET_COL)
                demand_norm = data_obj["data"][:, :, target_idx].flatten()

                norm_params = data_obj["norm_params"].item()[TARGET_COL]

                actual_demand = (
                    demand_norm * (norm_params["max"] - norm_params["min"] + 1e-7)
                    + norm_params["min"]
                )

                demands.extend(actual_demand)

            except Exception:
                continue

        if demands:
            mean_demands[level] = np.mean(demands)
            print(f"  Level {level:3d} -> Mean Demand = {mean_demands[level]:.3f} m3/h")

    return mean_demands


# =============================================================================
# 3. CREATE SUMMARY TABLE WITH RMSE AND NRMSE
# =============================================================================
def create_summary(df_full, mean_demand_dict):
    df_summary = df_full.groupby("Level").agg(
        Mean_RMSE=("RMSE", "mean"),
        Std_RMSE=("RMSE", "std"),
        N_Trials=("RMSE", "count")
    ).reset_index()

    df_summary["Mean_Demand"] = df_summary["Level"].map(mean_demand_dict)

    df_summary["Mean_NRMSE"] = (
        df_summary["Mean_RMSE"] / df_summary["Mean_Demand"]
    ) * 100

    df_summary["Std_NRMSE"] = (
        df_summary["Std_RMSE"] / df_summary["Mean_Demand"]
    ) * 100

    summary_csv_path = os.path.join(SAVE_DIR, "baseline_rmse_nrmse_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY TABLE")
    print("=" * 100)
    print(
        df_summary[
            ["Level", "Mean_Demand", "Mean_RMSE", "Std_RMSE", "Mean_NRMSE", "Std_NRMSE", "N_Trials"]
        ].round(3).to_string(index=False)
    )

    print(f"\nSummary results saved to: {summary_csv_path}")

    return df_summary


# =============================================================================
# 4. PLOT RMSE AND NRMSE
# =============================================================================
def plot_rmse_nrmse(df_summary):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    colors = ["#1f77b4", "#d62728"]

    # ==================== LEFT: RMSE ====================
    ax = axes[0]

    ax.errorbar(
        df_summary["Level"],
        df_summary["Mean_RMSE"],
        yerr=df_summary["Std_RMSE"],
        fmt="o-",
        capsize=5,
        linewidth=2.8,
        markersize=9,
        color=colors[0],
        ecolor=colors[0],
        label="Gradient Boosting Baseline (RMSE)"
    )

    ax.set_xlabel("Aggregation Level (Number of Users)", fontsize=16, fontweight="bold")
    ax.set_ylabel("RMSE (m3/h)", fontsize=16, fontweight="bold")
    ax.set_xlim(-10, 820)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=12.5)

    # ==================== RIGHT: NRMSE ====================
    ax = axes[1]

    ax.errorbar(
        df_summary["Level"],
        df_summary["Mean_NRMSE"],
        yerr=df_summary["Std_NRMSE"],
        fmt="o-",
        capsize=5,
        linewidth=2.8,
        markersize=9,
        color=colors[1],
        ecolor=colors[1],
        label="Gradient Boosting Baseline (NRMSE)"
    )

    ax.set_xlabel("Aggregation Level (Number of Users)", fontsize=16, fontweight="bold")
    ax.set_ylabel("NRMSE (%)", fontsize=16, fontweight="bold")
    ax.set_xlim(-10, 820)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=12.5)

    # Panel labels
    axes[0].text(
        0.5, -0.13, "(a)",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold"
    )

    axes[1].text(
        0.5, -0.13, "(b)",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    png_path = os.path.join(SAVE_DIR, "baseline_rmse_nrmse_dual_plot.png")
    tiff_path = os.path.join(SAVE_DIR, "baseline_rmse_nrmse_dual_plot.tiff")

    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=400, bbox_inches="tight")

    plt.show()

    print(f"\nPlots saved to:")
    print(png_path)
    print(tiff_path)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    df_full = run_baseline_trials()
    mean_demand_dict = calculate_mean_demand()
    df_summary = create_summary(df_full, mean_demand_dict)
    plot_rmse_nrmse(df_summary)

    print("\nDone.")
