import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# =========================
# PATHS
# =========================
root = Path(__file__).resolve().parents[4]
PROCESSED_DATA_DIR = root / "outputs" / "temporal_split_data"
CFT_VAE_RESULTS = root / "outputs" / "CFT-VAE" / "CFT-VAE-forecasting_results"
CFT_LSTM_TCN_RESULTS = CFT_VAE_RESULTS

RESULTS_BASE_DIR = (
    root
    / "outputs"
    / "Visualization"
    / "Synthetic Data Generation, Forecasting, and Statistical Validation Results"
    / "Significance Test Report"
)
SAVE_DIR = RESULTS_BASE_DIR / "statistical_tests"
os.makedirs(SAVE_DIR, exist_ok=True)

PLOT_DIR = RESULTS_BASE_DIR
os.makedirs(PLOT_DIR, exist_ok=True)

LEVELS = [20, 70, 200, 700]

# CUSTOM ORDER AS REQUESTED
MODEL_ORDER = ['RandomForest', 'XGBoost', 'GradientBoosting', 'SVM', 'LSTM', 'TCN']

# =========================
# LOADERS (same as before)
# =========================
def convert_lstm_tcn_to_long(deep_df):
    deep_df = deep_df.copy()
    deep_df.columns = [col.strip() for col in deep_df.columns]

    if {"training_type", "result_rmse"}.issubset(deep_df.columns):
        if "baseline_rmse" in deep_df.columns:
            return deep_df

        base_df = (
            deep_df[deep_df["training_type"] == "baseline"]
            [["level", "trial", "model", "result_rmse"]]
            .rename(columns={"result_rmse": "baseline_rmse"})
        )
        return deep_df.merge(base_df, on=["level", "trial", "model"], how="left")
    
    rows = []
    for _, row in deep_df.iterrows():
        base = row.get("base", np.nan)
        
        if pd.notna(base):
            rows.append({"level": row["level"], "trial": row["trial"], "model": row["model"],
                         "training_type": "baseline", "baseline_rmse": base, "result_rmse": base})
        
        syn_val = None
        for col in ["synthetic_alone", "synthetic_only", "synth_alone", "syn"]:
            if col in deep_df.columns and pd.notna(row.get(col)):
                syn_val = row[col]
                break
        if pd.notna(syn_val):
            rows.append({"level": row["level"], "trial": row["trial"], "model": row["model"],
                         "training_type": "synthetic_only", "baseline_rmse": base, "result_rmse": syn_val})
        
        if pd.notna(row.get("aug")):
            rows.append({"level": row["level"], "trial": row["trial"], "model": row["model"],
                         "training_type": "augmented", "baseline_rmse": base, "result_rmse": row["aug"]})
    return pd.DataFrame(rows)


def load_cft_results(level):
    dfs = []
    ratio_str = "1p0"
    cl_path = os.path.join(CFT_VAE_RESULTS, f"results_level{level}_ratio{ratio_str}.csv")
    if os.path.exists(cl_path):
        dfs.append(pd.read_csv(cl_path))
    
    lstm_path = os.path.join(CFT_LSTM_TCN_RESULTS, f"results_level{level}_ratio{ratio_str}_lstm_tcn.csv")
    if os.path.exists(lstm_path):
        df_lstm = pd.read_csv(lstm_path)
        dfs.append(convert_lstm_tcn_to_long(df_lstm))
    
    return pd.concat(dfs, ignore_index=True) if dfs else None


def get_mean_demand(level, trial=1):
    """Return mean denormalized demand used to convert RMSE to NRMSE."""
    train_path = (
        PROCESSED_DATA_DIR
        / "train"
        / f"train_preprocessed_level_{level}_trial_{trial}.npz"
    )
    if not train_path.exists():
        return None

    with np.load(train_path, allow_pickle=True) as data:
        feature_names = list(data["feature_names"])
        target = "total_demand_clipped"
        if target not in feature_names:
            return None
        target_idx = feature_names.index(target)
        norm_info = data["norm_params"].item()[target]
        normalized = data["data"][:, :, target_idx]

    demand = normalized * (norm_info["max"] - norm_info["min"] + 1e-7) + norm_info["min"]
    return float(np.mean(demand))


# =========================
# STATISTICAL TESTS
# =========================
def run_statistical_tests():
    all_stats = []
    for level in LEVELS:
        df = load_cft_results(level)
        if df is None or df.empty:
            continue
        mean_demand = get_mean_demand(level)
        if mean_demand is None or mean_demand <= 0:
            print(f"[!] Mean demand unavailable for level {level}; skipping.")
            continue
        for model in MODEL_ORDER:   # Use custom order
            aug = df[(df["model"] == model) & (df["training_type"] == "augmented")]
            base = df[(df["model"] == model) & (df["training_type"] == "baseline")]
            if len(aug) < 3 or len(base) < 3:
                continue
            merged = aug.merge(base, on="trial", suffixes=("_aug", "_base"))
            if len(merged) < 3:
                continue
                
            baseline_nrmse = merged["baseline_rmse_base"].values / mean_demand
            cft_aug_nrmse = merged["result_rmse_aug"].values / mean_demand
            
            t_stat, p_t = stats.ttest_rel(cft_aug_nrmse, baseline_nrmse)
            try:
                _, p_w = stats.wilcoxon(cft_aug_nrmse, baseline_nrmse)
            except:
                p_w = np.nan
                
            success_rate = (cft_aug_nrmse < baseline_nrmse).mean() * 100
            
            all_stats.append({
                "Level": level, "Model": model, "N_Trials": len(merged),
                "Baseline_Mean_NRMSE": round(baseline_nrmse.mean(), 6),
                "CFT_Aug_Mean_NRMSE": round(cft_aug_nrmse.mean(), 6),
                "Improvement_Pct": round((baseline_nrmse.mean() - cft_aug_nrmse.mean()) / baseline_nrmse.mean() * 100, 2),
                "Paired_t_test_p_value": p_t,
                "Wilcoxon_p_value": p_w,
                "Success_Rate_Pct": round(success_rate, 1)
            })
    
    stat_df = pd.DataFrame(all_stats)
    output_path = os.path.join(SAVE_DIR, "baseline_vs_cftvae_augmented_statistical_tests.csv")
    stat_df.to_csv(output_path, index=False)
    print(f"Statistical tests saved to: {output_path}")
    return stat_df


# =========================
# HEATMAP WITH YOUR DESIRED ORDER
# =========================
def plot_heatmap(stat_df):
    # Ensure correct order
    stat_df["Model"] = pd.Categorical(stat_df["Model"], categories=MODEL_ORDER, ordered=True)
    p_matrix = stat_df.pivot(index="Level", columns="Model", values="Paired_t_test_p_value")
    
    def format_p(p):
        if pd.isna(p): return "NA"
        p = float(p)
        return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    
    annot_matrix = p_matrix.applymap(format_p)
    
    plt.figure(figsize=(12, 6.5))
    ax = sns.heatmap(
        -np.log10(p_matrix),
        annot=annot_matrix,
        fmt='',
        cmap="YlGnBu",
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': r'$-\log_{10}(p$-value$)$', 'shrink': 0.75}
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    
    ax.set_xlabel("Forecasting Model", fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel("Aggregation Level", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    png_path = os.path.join(PLOT_DIR, "paired_ttest_heatmap_final.png")
    tiff_path = os.path.join(PLOT_DIR, "paired_ttest_heatmap_final.tiff")
    
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    plt.savefig(tiff_path, dpi=400, format='tiff', pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')
    
    print(f"Heatmap saved to: {png_path}")
    plt.show()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    stat_df = run_statistical_tests()
    plot_heatmap(stat_df)
