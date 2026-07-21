import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from matplotlib.collections import PolyCollection

# ==============================================================================
# CONFIGURATION
# ==============================================================================
root = Path(__file__).resolve().parents[3]
OLD_RESULTS_FILE = (
    root
    / "outputs"
    / "CFT-VAE"
    / "ablation"
    / "mask_[]"
    / "forecasting_results"
    / "results_lvl700.csv"
)
UPDATED_DL_RESULTS_FILE = (
    root
    / "outputs"
    / "CFT-VAE"
    / "CFT-VAE-forecasting_results"
    / "results_level700_ratio1p0_lstm_tcn.csv"
)
SAVE_VIS_DIR = (
    root
    / "outputs"
    / "Visualization"
    / "Distributional Analysis of Augmentation Gains"
)

LEVEL = 700
os.makedirs(SAVE_VIS_DIR, exist_ok=True)


# ==============================================================================
# LOAD AND MERGE DATA
# ==============================================================================
def normalize_lstm_tcn_results(df_new_dl):
    """Return LSTM/TCN rows with the same base/aug columns as results_lvl700.csv."""
    df_new_dl = df_new_dl[df_new_dl['model'].isin(['LSTM', 'TCN'])].copy()

    if {'base', 'aug'}.issubset(df_new_dl.columns):
        return df_new_dl[['trial', 'model', 'base', 'aug']].copy()

    required_cols = {'trial', 'model', 'training_type', 'result_rmse'}
    missing_cols = required_cols - set(df_new_dl.columns)
    if missing_cols:
        raise ValueError(
            "LSTM/TCN results file must contain either base/aug columns or "
            f"{sorted(required_cols)}. Missing: {sorted(missing_cols)}"
        )

    df_wide = (
        df_new_dl[df_new_dl['training_type'].isin(['baseline', 'augmented'])]
        .pivot_table(
            index=['trial', 'model'],
            columns='training_type',
            values='result_rmse',
            aggfunc='first'
        )
        .reset_index()
        .rename(columns={'baseline': 'base', 'augmented': 'aug'})
    )

    return df_wide.dropna(subset=['base', 'aug'])[['trial', 'model', 'base', 'aug']].copy()


def load_and_process_data(old_results_file, updated_dl_results_file):
    if not os.path.exists(old_results_file):
        print(f"Error: File not found:\n{old_results_file}")
        return None

    if not os.path.exists(updated_dl_results_file):
        print(f"Error: File not found:\n{updated_dl_results_file}")
        return None

    df_old = pd.read_csv(old_results_file)
    df_new_dl = pd.read_csv(updated_dl_results_file)

    # Remove old LSTM and TCN rows from the original file
    df_old_without_dl = df_old[~df_old['model'].isin(['LSTM', 'TCN'])].copy()

    # Keep only updated LSTM and TCN rows in the same format as the old file
    df_new_dl = normalize_lstm_tcn_results(df_new_dl)

    # Combine old classical models + updated LSTM/TCN
    df = pd.concat([df_old_without_dl, df_new_dl], ignore_index=True)

    # Compute percentage improvement
    df['Improvement'] = (df['base'] - df['aug']) / df['base'] * 100

    return df[['trial', 'model', 'Improvement']].copy()


# ==============================================================================
# CLASSIFY TRIALS
# ==============================================================================
def categorize_trial(pct):
    if pct >= 15:
        return 'Strong Improvement (≥15%)'
    elif -15 <= pct < 15:
        return 'limited change (±15%)'
    else:
        return 'Degradation (<-15%)'


# ==============================================================================
# PLOT HYBRID DASHBOARD
# ==============================================================================
def plot_hybrid_dashboard(df, level, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    model_order = [
        'RandomForest',
        'XGBoost',
        'GradientBoosting',
        'SVM',
        'LSTM',
        'TCN'
    ]

    df = df.copy()
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    df = df.sort_values('model')

    # --------------------------------------------------------------------------
    # (a) STACKED BAR CHART
    # --------------------------------------------------------------------------
    ax = axes[0]

    df['Category'] = df['Improvement'].apply(categorize_trial)

    counts = df.groupby(['model', 'Category'], observed=False).size().unstack(fill_value=0)

    category_order = [
        'Strong Improvement (≥15%)',
        'limited change (±15%)',
        'Degradation (<-15%)'
    ]

    counts = counts.reindex(index=model_order, columns=category_order, fill_value=0)

    colors_bar = ['#27ae60', '#bdc3c7', '#c0392b']

    counts.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=colors_bar,
        edgecolor='black',
        width=0.7
    )

    for container in ax.containers:
        labels = [
            int(bar.get_height()) if bar.get_height() > 5 else ''
            for bar in container
        ]
        ax.bar_label(
            container,
            labels=labels,
            label_type='center',
            fontweight='bold',
            color='black',
            fontsize=10
        )

    ax.set_ylabel("Number of Trials (N=100)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Forecasting Model", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.tick_params(axis='x', rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha('right')

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=True,
        edgecolor='black',
        fontsize=14
    )

    ax.text(
        0.5, -0.18, '(a)',
        transform=ax.transAxes,
        fontsize=15,
        fontweight='bold',
        ha='center'
    )

    # --------------------------------------------------------------------------
    # (b) VIOLIN + STRIP PLOT
    # --------------------------------------------------------------------------
    ax = axes[1]

    sns.violinplot(
        data=df,
        x='model',
        y='Improvement',
        order=model_order,
        ax=ax,
        hue='model',
        palette='Set2',
        inner=None,
        linewidth=2.0,
        legend=False
    )

    for art in ax.findobj(PolyCollection):
        art.set_edgecolor('black')

    sns.stripplot(
        data=df,
        x='model',
        y='Improvement',
        order=model_order,
        ax=ax,
        color='black',
        alpha=0.5,
        jitter=0.25,
        size=4
    )

    ax.axhline(0, color='#e74c3c', linestyle='--', linewidth=2.5)
    ax.axhspan(-15, 15, color='gray', alpha=0.10)

    ax.set_ylabel("NRMSE Reduction (%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Forecasting Model", fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha('right')

    ax.text(
        0.5, -0.18, '(b)',
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        ha='center'
    )

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        f"ablation_level{level}.tiff"
    )

    plt.savefig(
        save_path,
        dpi=300,
        format='tiff',
        pil_kwargs={"compression": "tiff_lzw"},
        bbox_inches='tight'
    )

    print(f"Figure saved to:\n{save_path}")
    plt.show()


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    data = load_and_process_data(OLD_RESULTS_FILE, UPDATED_DL_RESULTS_FILE)

    if data is not None:
        plot_hybrid_dashboard(data, LEVEL, SAVE_VIS_DIR)
