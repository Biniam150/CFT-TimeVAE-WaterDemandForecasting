import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LSTM
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor


LEVEL = 700
RATIO = 1.0
RANDOM_STATE = 42
STRONG_THRESHOLD = 15.0
BATCH_SIZE = 10
REST_SECONDS = 30

REPO_ROOT = Path(__file__).resolve().parents[4]
ABLATION_BASE = REPO_ROOT / "outputs" / "CFT-VAE" / "ablation"
PROCESSED_DIR = REPO_ROOT / "outputs" / "temporal_split_data"
CFT_RESULTS_DIR = REPO_ROOT / "outputs" / "CFT-VAE" / "CFT-VAE-forecasting_results"
RESULTS_DIR = Path(__file__).resolve().parent / "strong_trial_ablation_results"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

CLASSICAL_REFERENCE = CFT_RESULTS_DIR / "results_level700_ratio1p0.csv"
DEEP_REFERENCE = CFT_RESULTS_DIR / "results_level700_ratio1p0_lstm_tcn.csv"
CHECKPOINT_FILE = RESULTS_DIR / "strong_trial_ablation_level700_results.csv"

CONDITIONS = [
    ("mask_[]", "Full Conditioning"),
    ("mask_[1]", "No Lags"),
    ("mask_[2]", "No Rolling & Change"),
    ("mask_[3]", "No Calendar"),
    ("mask_[4]", "No Daily Context"),
    ("mask_[5]", "No Stability"),
    ("gaussian_noise", "Gaussian Noise"),
    ("mask_[1, 2, 3, 4, 5]", "No Conditioning"),
]

MODELS = [
    "RandomForest",
    "XGBoost",
    "GradientBoosting",
    "SVM",
    "LSTM",
    "TCN",
]


class LSTMRegressorWrapper:
    def __init__(self, units=64, epochs=30, batch_size=32):
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_arr = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1).astype(np.float32)
        y_arr = y.to_numpy(dtype=np.float32)

        self.model = Sequential([
            LSTM(self.units, input_shape=(X.shape[1], 1)),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        )
        self.model.fit(
            X_arr,
            y_arr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            shuffle=True,
            callbacks=[early_stop],
            verbose=0,
        )

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_arr = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1).astype(np.float32)
        return self.model.predict(X_arr, verbose=0).flatten()


class TCNRegressorWrapper:
    def __init__(self, filters=64, kernel_size=3, epochs=30, batch_size=32):
        self.filters = filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_arr = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1).astype(np.float32)
        y_arr = y.to_numpy(dtype=np.float32)

        self.model = Sequential([
            Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                dilation_rate=1,
                padding="causal",
                activation="relu",
                input_shape=(X.shape[1], 1),
            ),
            Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                dilation_rate=2,
                padding="causal",
                activation="relu",
            ),
            Flatten(),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(
            X_arr,
            y_arr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_arr = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1).astype(np.float32)
        return self.model.predict(X_arr, verbose=0).flatten()


def create_model(name):
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=RANDOM_STATE,
        )
    if name == "SVM":
        return SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
    if name == "LSTM":
        return LSTMRegressorWrapper(units=64, epochs=30, batch_size=32)
    if name == "TCN":
        return TCNRegressorWrapper(filters=64, kernel_size=3, epochs=30, batch_size=32)
    raise ValueError(f"Unknown model: {name}")


def calculate_mape(y_true, y_pred):
    values = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    mask = np.abs(values["y_true"]) > 1e-10
    if not mask.any():
        return np.nan
    return float(
        np.mean(
            np.abs(
                (values.loc[mask, "y_true"] - values.loc[mask, "y_pred"])
                / values.loc[mask, "y_true"]
            )
        )
        * 100
    )


def calculate_metrics(y_true, y_pred):
    values = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    return {
        "rmse": float(np.sqrt(mean_squared_error(values["y_true"], values["y_pred"]))),
        "mae": float(mean_absolute_error(values["y_true"], values["y_pred"])),
        "r2": float(r2_score(values["y_true"], values["y_pred"])),
        "mape": calculate_mape(values["y_true"], values["y_pred"]),
    }


def load_real_data(path):
    with np.load(path, allow_pickle=True) as data:
        return {
            "data": data["data"].astype(np.float32),
            "norm_params": data["norm_params"].item(),
        }


def load_synthetic_data(path):
    with np.load(path, allow_pickle=True) as data:
        return data["synthetic_samples"].astype(np.float32)


def denormalize_data(data, norm_params, feature_name="total_demand_clipped"):
    norm_info = norm_params[feature_name]
    return data * (norm_info["max"] - norm_info["min"] + 1e-7) + norm_info["min"]


def create_features_and_targets(time_series_data):
    if time_series_data.ndim == 3:
        flattened = time_series_data[:, :, 0].flatten()
    else:
        flattened = time_series_data.flatten()

    time_index = pd.date_range(start="2020-01-01", periods=len(flattened), freq="h")
    demand = pd.DataFrame({"demand": flattened}, index=time_index)
    features = pd.DataFrame(index=time_index)

    for lag in [1, 2, 3, 24, 25, 48, 168]:
        features[f"lag_{lag}"] = demand["demand"].shift(lag)

    features["roll_24h_mean"] = demand["demand"].shift(1).rolling(24, min_periods=12).mean()
    features["roll_24h_std"] = demand["demand"].shift(1).rolling(24, min_periods=12).std()
    features["roll_168h_mean"] = demand["demand"].shift(1).rolling(168, min_periods=24).mean()
    features["hour"] = features.index.hour
    features["dayofweek"] = features.index.dayofweek
    features["month"] = features.index.month
    features["hour_sin"] = np.sin(2 * np.pi * features.index.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features.index.hour / 24)
    features["dayofweek_sin"] = np.sin(2 * np.pi * features.index.dayofweek / 7)
    features["dayofweek_cos"] = np.cos(2 * np.pi * features.index.dayofweek / 7)

    combined = pd.concat([features, demand["demand"].rename("target")], axis=1).dropna()
    return combined.drop(columns="target"), combined["target"]


def load_reference_results():
    missing = [p for p in [CLASSICAL_REFERENCE, DEEP_REFERENCE] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing reference results:\n" + "\n".join(map(str, missing)))

    reference = pd.concat(
        [pd.read_csv(CLASSICAL_REFERENCE), pd.read_csv(DEEP_REFERENCE)],
        ignore_index=True,
    )
    reference = reference[
        reference["model"].isin(MODELS)
        & reference["training_type"].isin(["baseline", "augmented"])
    ]
    wide = reference.pivot_table(
        index=["trial", "model"],
        columns="training_type",
        values="result_rmse",
        aggfunc="first",
    ).dropna(subset=["baseline", "augmented"])
    wide["improvement_pct"] = (wide["baseline"] - wide["augmented"]) / wide["baseline"] * 100

    complete = wide.reset_index().pivot_table(
        index="trial",
        columns="model",
        values="improvement_pct",
        aggfunc="first",
    ).dropna(subset=MODELS)
    trial_average = complete[MODELS].mean(axis=1)
    strong_trials = sorted(trial_average[trial_average >= STRONG_THRESHOLD].index.astype(int))

    baseline_lookup = wide["baseline"].to_dict()
    return strong_trials, baseline_lookup, trial_average


def synthetic_path(condition_folder, trial):
    return (
        ABLATION_BASE
        / condition_folder
        / "synthetic_data"
        / f"synthetic_lvl{LEVEL}_trial{trial}.npz"
    )


def validate_inputs(selected_trials, baseline_lookup):
    missing = []
    for trial in selected_trials:
        train_path = PROCESSED_DIR / "train" / f"train_preprocessed_level_{LEVEL}_trial_{trial}.npz"
        test_path = PROCESSED_DIR / "test" / f"test_preprocessed_level_{LEVEL}_trial_{trial}.npz"
        for path in [train_path, test_path]:
            if not path.exists():
                missing.append(str(path))
        for folder, _ in CONDITIONS:
            path = synthetic_path(folder, trial)
            if not path.exists():
                missing.append(str(path))
        for model in MODELS:
            if (trial, model) not in baseline_lookup:
                missing.append(f"baseline: trial={trial}, model={model}")

    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"Missing required inputs:\n{preview}{suffix}")


def read_checkpoint():
    if CHECKPOINT_FILE.exists():
        return pd.read_csv(CHECKPOINT_FILE)
    return pd.DataFrame()


def save_checkpoint(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(results).drop_duplicates(
        subset=["level", "trial", "condition_id", "model"],
        keep="last",
    )
    temp_path = CHECKPOINT_FILE.with_suffix(".tmp")
    output.to_csv(temp_path, index=False)
    os.replace(temp_path, CHECKPOINT_FILE)
    return output


def prepare_real_trial(trial):
    train_path = PROCESSED_DIR / "train" / f"train_preprocessed_level_{LEVEL}_trial_{trial}.npz"
    test_path = PROCESSED_DIR / "test" / f"test_preprocessed_level_{LEVEL}_trial_{trial}.npz"
    train_data = load_real_data(train_path)
    test_data = load_real_data(test_path)

    train_real = denormalize_data(train_data["data"][:, :, 0], train_data["norm_params"])
    test_real = denormalize_data(test_data["data"][:, :, 0], test_data["norm_params"])
    X_train, y_train = create_features_and_targets(train_real)
    X_test, y_test = create_features_and_targets(test_real)
    return X_train, y_train, X_test, y_test, float(np.mean(train_real))


def run_augmented_model(name, X_aug, y_aug, X_test, y_test):
    if name in {"LSTM", "TCN"}:
        tf.random.set_seed(RANDOM_STATE)
    model = create_model(name)
    model.fit(X_aug, y_aug)
    metrics = calculate_metrics(y_test, model.predict(X_test))
    del model
    if name in {"LSTM", "TCN"}:
        tf.keras.backend.clear_session()
    gc.collect()
    return metrics


def build_summaries(results, selected_trials):
    selected = results[results["trial"].isin(selected_trials)].copy()
    selected["model"] = pd.Categorical(selected["model"], categories=MODELS, ordered=True)
    condition_order = [label for _, label in CONDITIONS]
    selected["condition"] = pd.Categorical(
        selected["condition"], categories=condition_order, ordered=True
    )

    summary = (
        selected.groupby(["condition", "model"], observed=False)
        .agg(
            n_trials=("trial", "nunique"),
            baseline_nrmse_mean=("baseline_nrmse", "mean"),
            baseline_nrmse_std=("baseline_nrmse", "std"),
            augmented_nrmse_mean=("augmented_nrmse", "mean"),
            augmented_nrmse_std=("augmented_nrmse", "std"),
            avg_improvement_pct=("improvement_pct", "mean"),
            improvement_std=("improvement_pct", "std"),
            success_rate_pct=("improvement_pct", lambda x: float((x > 0).mean() * 100)),
        )
        .reset_index()
    )
    summary_path = RESULTS_DIR / f"strong_trial_ablation_level700_summary_n{len(selected_trials)}.csv"
    summary.to_csv(summary_path, index=False)
    return selected, summary, summary_path


def create_plots(selected, summary, n_trials):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    condition_order = [label for _, label in CONDITIONS]

    overall = (
        selected.groupby("condition", observed=False)["improvement_pct"]
        .mean()
        .reindex(condition_order)
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = ["0.25"] + ["0.65"] * 5 + ["0.55", "0.85"]
    bars = ax.bar(overall.index, overall.values, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Average NRMSE improvement (%)")
    ax.set_xlabel("Synthetic-data condition")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    avg_png = PLOTS_DIR / f"strong_trial_ablation_average_n{n_trials}.png"
    avg_tiff = PLOTS_DIR / f"strong_trial_ablation_average_n{n_trials}.tiff"
    fig.savefig(avg_png, dpi=400, bbox_inches="tight")
    fig.savefig(avg_tiff, dpi=400, format="tiff", bbox_inches="tight")
    plt.close(fig)

    pivot = summary.pivot(index="model", columns="condition", values="avg_improvement_pct")
    pivot = pivot.reindex(index=MODELS, columns=condition_order)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(MODELS))
    width = 0.1
    for index, condition in enumerate(condition_order):
        offset = (index - (len(condition_order) - 1) / 2) * width
        ax.bar(x + offset, pivot[condition], width, label=condition, edgecolor="black", linewidth=0.35)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=25, ha="right")
    ax.set_ylabel("Average NRMSE improvement (%)")
    ax.set_xlabel("Forecasting model")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    model_png = PLOTS_DIR / f"strong_trial_ablation_modelwise_n{n_trials}.png"
    model_tiff = PLOTS_DIR / f"strong_trial_ablation_modelwise_n{n_trials}.tiff"
    fig.savefig(model_png, dpi=400, bbox_inches="tight")
    fig.savefig(model_tiff, dpi=400, format="tiff", bbox_inches="tight")
    plt.close(fig)
    return [avg_png, avg_tiff, model_png, model_tiff]


def check_inputs():
    strong_trials, baseline_lookup, _ = load_reference_results()
    validate_inputs(strong_trials, baseline_lookup)
    sample_trial = strong_trials[0]
    X_train, _, X_test, _, _ = prepare_real_trial(sample_trial)
    for condition_id, condition_label in CONDITIONS:
        X_syn, _ = create_features_and_targets(
            load_synthetic_data(synthetic_path(condition_id, sample_trial))
        )
        if list(X_syn.columns) != list(X_train.columns):
            raise ValueError(f"Feature mismatch for {condition_label} at trial {sample_trial}.")
    if list(X_test.columns) != list(X_train.columns):
        raise ValueError(f"Real train/test feature mismatch at trial {sample_trial}.")
    print(f"Input check passed: {len(strong_trials)} shared strong trials are complete.")
    print(f"Feature check passed: all eight conditions use {X_train.shape[1]} engineered features.")
    print(f"Expected augmented fits: {len(strong_trials) * len(CONDITIONS) * len(MODELS)}")


def run_experiment():
    total_start = time.time()
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    strong_trials, baseline_lookup, trial_average = load_reference_results()
    print("=" * 80)
    print("LEVEL 700 STRONG-TRIAL ABLATION FORECASTING")
    print("=" * 80)
    print(
        f"Found {len(strong_trials)} shared strong trials using the full-conditioning "
        f">= {STRONG_THRESHOLD:.0f}% average-improvement rule."
    )
    print(f"Strong trial IDs: {strong_trials}")

    while True:
        try:
            requested = int(input(f"How many strong trials do you want to run (1-{len(strong_trials)}): "))
            if 1 <= requested <= len(strong_trials):
                break
        except ValueError:
            pass
        print("Please enter a valid whole number.")

    selected_trials = strong_trials[:requested]
    validate_inputs(selected_trials, baseline_lookup)
    print(f"Selected trial IDs: {selected_trials}")
    print(f"Planned augmented fits: {requested * len(CONDITIONS) * len(MODELS)}")

    checkpoint = read_checkpoint()
    results = checkpoint.to_dict("records") if not checkpoint.empty else []
    completed = {
        (int(row["trial"]), row["condition_id"], row["model"])
        for row in results
    }

    for trial_index, trial in enumerate(selected_trials, start=1):
        print(f"\nTrial {trial} ({trial_index}/{requested})")
        X_train, y_train, X_test, y_test, mean_demand = prepare_real_trial(trial)

        for condition_id, condition_label in CONDITIONS:
            pending_models = [
                model for model in MODELS if (trial, condition_id, model) not in completed
            ]
            if not pending_models:
                print(f"  {condition_label}: already complete")
                continue

            synthetic = load_synthetic_data(synthetic_path(condition_id, trial))
            X_syn, y_syn = create_features_and_targets(synthetic)
            common_features = X_train.columns.intersection(X_syn.columns)
            X_aug = pd.concat(
                [X_train[common_features], X_syn[common_features]], ignore_index=True
            )
            y_aug = pd.concat([y_train, y_syn], ignore_index=True)
            X_test_common = X_test[common_features]

            print(f"  {condition_label}")
            for model_name in pending_models:
                model_start = time.time()
                metrics = run_augmented_model(
                    model_name, X_aug, y_aug, X_test_common, y_test
                )
                baseline_rmse = float(baseline_lookup[(trial, model_name)])
                baseline_nrmse = baseline_rmse / mean_demand
                augmented_nrmse = metrics["rmse"] / mean_demand
                improvement = (baseline_rmse - metrics["rmse"]) / baseline_rmse * 100

                results.append({
                    "level": LEVEL,
                    "trial": trial,
                    "reference_avg_improvement_pct": float(trial_average.loc[trial]),
                    "condition_id": condition_id,
                    "condition": condition_label,
                    "model": model_name,
                    "training_type": "augmented",
                    "baseline_rmse": baseline_rmse,
                    "augmented_rmse": metrics["rmse"],
                    "mean_demand": mean_demand,
                    "baseline_nrmse": baseline_nrmse,
                    "augmented_nrmse": augmented_nrmse,
                    "improvement_pct": improvement,
                    "result_mae": metrics["mae"],
                    "result_r2": metrics["r2"],
                    "result_mape": metrics["mape"],
                    "runtime_seconds": time.time() - model_start,
                })
                completed.add((trial, condition_id, model_name))
                checkpoint = save_checkpoint(results)
                print(
                    f"    {model_name:18} Aug RMSE={metrics['rmse']:.3f} "
                    f"Improvement={improvement:+.2f}%"
                )

            del synthetic, X_syn, y_syn, X_aug, y_aug
            gc.collect()

        if trial_index % BATCH_SIZE == 0 and trial_index < requested:
            tf.keras.backend.clear_session()
            gc.collect()
            print(f"\nCompleted {BATCH_SIZE} trials. Resting for {REST_SECONDS} seconds.")
            time.sleep(REST_SECONDS)

    final_results = read_checkpoint()
    selected, summary, summary_path = build_summaries(final_results, selected_trials)
    plot_paths = create_plots(selected, summary, requested)

    elapsed_hours = (time.time() - total_start) / 3600
    print("\nExperiment complete.")
    print(f"Detailed checkpoint: {CHECKPOINT_FILE}")
    print(f"Summary: {summary_path}")
    for path in plot_paths:
        print(f"Plot: {path}")
    print(f"Total experiment time: {elapsed_hours:.2f} hours")


def main():
    parser = argparse.ArgumentParser(description="Run Level 700 strong-trial ablation forecasting.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate all required inputs without training any forecasting models.",
    )
    args = parser.parse_args()
    if args.check:
        check_inputs()
    else:
        run_experiment()


if __name__ == "__main__":
    main()
