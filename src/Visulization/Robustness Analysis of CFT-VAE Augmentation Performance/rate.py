import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ==============================================================================
# WR2 VISUALIZATION CONFIGURATION
# ==============================================================================
root = Path(__file__).resolve().parents[3]
CFT_VAE_RESULTS = root / "outputs" / "CFT-VAE" / "CFT-VAE-forecasting_results"
TIME_VAE_RESULTS = root / "outputs" / "Time-VAE" / "Time-vae_forecasting_results"

CFT_LSTM_TCN_RESULTS = CFT_VAE_RESULTS
TIME_LSTM_TCN_RESULTS = TIME_VAE_RESULTS

SAVE_VIS_DIR = root / "outputs" / "Visualization" / "Robustness Analysis of CFT-VAE Augmentation Performance"
os.makedirs(SAVE_VIS_DIR, exist_ok=True)

LEVELS = [20, 70, 200, 700]
MODELS = ['GradientBoosting', 'RandomForest', 'SVM', 'XGBoost', 'LSTM', 'TCN']
RATIO = 1.0


# ==============================================================================
# LOADERS
# ==============================================================================
def convert_lstm_tcn_to_long(deep_df):
    deep_df.columns = deep_df.columns.str.strip()

    if {"training_type", "result_rmse"}.issubset(deep_df.columns):
        if "baseline_rmse" in deep_df.columns:
            return deep_df

        base_df = (
            deep_df[deep_df["training_type"] == "baseline"]
            [["level", "trial", "model", "result_rmse"]]
            .rename(columns={"result_rmse": "baseline_rmse"})
        )

        out = deep_df.merge(base_df, on=["level", "trial", "model"], how="left")
        return out[out["training_type"].isin(["synthetic_only", "augmented"])].copy()

    syn_col = None
    for possible in ["synthetic_alone", "synthetic_only", "synth_alone", "syn"]:
        if possible in deep_df.columns:
            syn_col = possible
            break

    required = ["level", "trial", "model", "base", "aug"]
    missing = [c for c in required if c not in deep_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in LSTM/TCN file: {missing}")

    rows = []

    for _, row in deep_df.iterrows():
        if syn_col is not None:
            rows.append({
                "level": row["level"],
                "trial": row["trial"],
                "model": row["model"],
                "training_type": "synthetic_only",
                "baseline_rmse": row["base"],
                "result_rmse": row[syn_col]
            })

        rows.append({
            "level": row["level"],
            "trial": row["trial"],
            "model": row["model"],
            "training_type": "augmented",
            "baseline_rmse": row["base"],
            "result_rmse": row["aug"]
        })

    return pd.DataFrame(rows)


def load_cft_results(level):
    ratio_str = str(RATIO).replace(".", "p")
    dfs = []

    classical_path = os.path.join(
        CFT_VAE_RESULTS,
        f"results_level{level}_ratio{ratio_str}.csv"
    )

    deep_path = os.path.join(
        CFT_LSTM_TCN_RESULTS,
        f"results_level{level}_ratio{ratio_str}_lstm_tcn.csv"
    )

    if os.path.exists(classical_path):
        df = pd.read_csv(classical_path)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    else:
        print(f"[!] Missing CFT-VAE classical file: {classical_path}")

    if os.path.exists(deep_path):
        df_deep = pd.read_csv(deep_path)
        df_deep_long = convert_lstm_tcn_to_long(df_deep)
        dfs.append(df_deep_long)
    else:
        print(f"[!] Missing CFT-VAE LSTM/TCN file: {deep_path}")

    if len(dfs) == 0:
        return None

    return pd.concat(dfs, ignore_index=True)


def load_timevae_results(level):
    ratio_str = str(RATIO).replace(".", "p")
    dfs = []

    classical_path = os.path.join(
        TIME_VAE_RESULTS,
        f"results_level{level}_ratio{ratio_str}.csv"
    )

    deep_path = os.path.join(
        TIME_LSTM_TCN_RESULTS,
        f"results_lvl{level}_lstm_tcn_original_timevae_ratio{ratio_str}.csv"
    )

    if os.path.exists(classical_path):
        df = pd.read_csv(classical_path)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    else:
        print(f"[!] Missing TimeVAE classical file: {classical_path}")

    if os.path.exists(deep_path):
        df_deep = pd.read_csv(deep_path)
        df_deep_long = convert_lstm_tcn_to_long(df_deep)
        dfs.append(df_deep_long)
    else:
        print(f"[!] Missing TimeVAE LSTM/TCN file: {deep_path}")

    if len(dfs) == 0:
        return None

    return pd.concat(dfs, ignore_index=True)


# ==============================================================================
# SUCCESS RATE CALCULATION
# ==============================================================================
def calculate_success_rates():
    level_data = {}

    for level in LEVELS:
        df_cft = load_cft_results(level)
        df_tvae = load_timevae_results(level)

        if df_cft is None or df_tvae is None:
            print(f"[!] Missing data for Level {level}")
            continue

        results = []

        for model in MODELS:
            # CFT-VAE augmented
            c_aug = df_cft[
                (df_cft["model"] == model) &
                (df_cft["training_type"] == "augmented")
            ][["trial", "baseline_rmse", "result_rmse"]].rename(
                columns={"result_rmse": "cft_aug_rmse"}
            )

            # TimeVAE augmented
            t_aug = df_tvae[
                (df_tvae["model"] == model) &
                (df_tvae["training_type"] == "augmented")
            ][["trial", "result_rmse"]].rename(
                columns={"result_rmse": "timevae_aug_rmse"}
            )

            # CFT-VAE synthetic only
            c_syn = df_cft[
                (df_cft["model"] == model) &
                (df_cft["training_type"] == "synthetic_only")
            ][["trial", "result_rmse"]].rename(
                columns={"result_rmse": "cft_syn_rmse"}
            )

            # TimeVAE synthetic only
            t_syn = df_tvae[
                (df_tvae["model"] == model) &
                (df_tvae["training_type"] == "synthetic_only")
            ][["trial", "result_rmse"]].rename(
                columns={"result_rmse": "timevae_syn_rmse"}
            )

            if len(c_aug) == 0:
                print(f"[!] Missing CFT-VAE augmented results for Level {level}, Model {model}")
                continue

            # 1. CFT-VAE Augmented vs Baseline
            rate_baseline = (
                c_aug["cft_aug_rmse"] < c_aug["baseline_rmse"]
            ).mean() * 100

            # 2. CFT-VAE Augmented vs TimeVAE Augmented
            merged_aug = c_aug.merge(t_aug, on="trial", how="inner")
            if len(merged_aug) > 0:
                rate_timevae_aug = (
                    merged_aug["cft_aug_rmse"] < merged_aug["timevae_aug_rmse"]
                ).mean() * 100
            else:
                rate_timevae_aug = np.nan

            # 3. CFT-VAE Synthetic vs TimeVAE Synthetic
            merged_syn = c_syn.merge(t_syn, on="trial", how="inner")
            if len(merged_syn) > 0:
                rate_timevae_syn = (
                    merged_syn["cft_syn_rmse"] < merged_syn["timevae_syn_rmse"]
                ).mean() * 100
            else:
                rate_timevae_syn = np.nan

            results.append({
                "model": model,
                "CFT-VAE Augmented vs Baseline": rate_baseline,
                "CFT-VAE Augmented vs TimeVAE Augmented": rate_timevae_aug,
                "CFT-VAE Synthetic vs TimeVAE Synthetic": rate_timevae_syn
            })

        level_data[level] = pd.DataFrame(results)

    return level_data


# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_success_rates(data_by_level):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    colors = {
        "CFT-VAE Augmented vs Baseline": "#2E86AB",
        "CFT-VAE Augmented vs TimeVAE Augmented": "#A23B72",
        "CFT-VAE Synthetic vs TimeVAE Synthetic": "#F18F01"
    }

    for idx, level in enumerate(LEVELS):
        ax = axes[idx]

        if level not in data_by_level or data_by_level[level].empty:
            ax.text(
                0.5, 0.5,
                f"Level {level} Data Missing",
                ha="center",
                va="center"
            )
            continue

        df = data_by_level[level]
        models = df["model"].tolist()

        x = np.arange(len(models))
        width = 0.25

        bars1 = ax.bar(
            x - width,
            df["CFT-VAE Augmented vs Baseline"],
            width,
            label="CFT-VAE Augmented vs Baseline",
            color=colors["CFT-VAE Augmented vs Baseline"],
            alpha=0.85
        )

        bars2 = ax.bar(
            x,
            df["CFT-VAE Augmented vs TimeVAE Augmented"],
            width,
            label="CFT-VAE Augmented vs TimeVAE Augmented",
            color=colors["CFT-VAE Augmented vs TimeVAE Augmented"],
            alpha=0.85
        )

        bars3 = ax.bar(
            x + width,
            df["CFT-VAE Synthetic vs TimeVAE Synthetic"],
            width,
            label="CFT-VAE Synthetic vs TimeVAE Synthetic",
            color=colors["CFT-VAE Synthetic vs TimeVAE Synthetic"],
            alpha=0.85
        )

        
        ax.set_xlabel("Forecasting Model", fontsize=14, fontweight="bold")
        ax.xaxis.set_label_coords(0.5, -0.15)
        ax.set_ylabel("Success Rate (%)", fontsize=14, fontweight="bold")
        ax.set_title(f"Aggregation Level {level}", fontsize=14, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10, rotation=25, ha="right")

        ax.set_ylim([0, 115])
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()

                if np.isnan(height):
                    continue

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1,
                    f"{height:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold"
                )

        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=14,
        bbox_to_anchor=(0.5, 1.0),
        frameon=True,
        edgecolor="black"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path_tiff = os.path.join(
        SAVE_VIS_DIR,
        "success_rates_6models.tiff"
    )

    save_path_png = os.path.join(
        SAVE_VIS_DIR,
        "success_rates_6models.png"
    )

    plt.savefig(
        save_path_tiff,
        dpi=300,
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
        bbox_inches="tight"
    )

    plt.savefig(
        save_path_png,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"\nHigh-res TIFF saved to:\n{save_path_tiff}")
    print(f"PNG saved to:\n{save_path_png}")

    plt.show()


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("Analyzing success rates across 100 trials...")
    success_data = calculate_success_rates()

    for level, df in success_data.items():
        print(f"\nLEVEL {level}")
        print(df)

    plot_success_rates(success_data)
