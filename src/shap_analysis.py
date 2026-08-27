import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/processed/aquasensor_final.csv"
PERFORMANCE_PATH = "data/processed/do_prediction_model_performance.csv"
MODELS_DIR = "models"

OUTPUT_DIR = "data/processed"
FIGURES_DIR = "figures/shap"

SAMPLE_SIZE = 500
HORIZON = "60min"

FEATURES = [
    "temperature",
    "air_temperature_c",
    "sunshine_wm2",
    "hour",
    "dissolved_oxygen_mgl",
    "dissolved_oxygen_pct",
    "pollution_alert",
    "anomaly_type",
    "season_proxy",
    "sensor_id",
    "sensor_name",
]


def safe_filename_text(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("%", "pct")
        .replace("/", "_per_")
    )


def load_best_model_name(metrics, target_type, horizon):
    selected = metrics[
        (metrics["target_type"] == target_type) &
        (metrics["horizon"] == horizon)
    ]

    best_row = selected.sort_values("RMSE").iloc[0]
    return best_row["model"]


def load_model(model_name, target_type, horizon):
    model_file = (
        f"{safe_filename_text(model_name)}_"
        f"{safe_filename_text(target_type)}_"
        f"{horizon}.pkl"
    )

    model_path = os.path.join(MODELS_DIR, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


def prepare_shap_data(model, X_sample):
    preprocessor = model.named_steps["preprocessor"]
    trained_model = model.named_steps["model"]

    X_transformed = preprocessor.transform(X_sample)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    return trained_model, X_transformed, feature_names


def run_shap_for_target(df, metrics, target_type, target_col, horizon):
    print("\n" + "=" * 70)
    print(f"SHAP EXPLAINABILITY: {target_type} | {horizon}")
    print("=" * 70)

    model_name = load_best_model_name(metrics, target_type, horizon)

    print(f"Best model selected: {model_name}")

    model = load_model(model_name, target_type, horizon)

    shap_df = df.dropna(subset=[target_col]).copy()
    shap_df = shap_df.tail(SAMPLE_SIZE)

    X_sample = shap_df[FEATURES].copy()

    trained_model, X_transformed, feature_names = prepare_shap_data(
        model,
        X_sample,
    )

    print("Calculating SHAP values...")
    print("Sample size:", X_transformed.shape[0])
    print("Number of features after encoding:", X_transformed.shape[1])

    explainer = shap.Explainer(trained_model, X_transformed)
    shap_values = explainer(X_transformed)

    safe_target = (
        target_type.lower()
        .replace(" ", "_")
        .replace("%", "pct")
        .replace("/", "_")
    )

    summary_plot_path = os.path.join(
        FIGURES_DIR,
        f"shap_summary_{safe_target}_{horizon}.png",
    )

    bar_plot_path = os.path.join(
        FIGURES_DIR,
        f"shap_bar_{safe_target}_{horizon}.png",
    )

    table_path = os.path.join(
        OUTPUT_DIR,
        f"shap_feature_importance_{safe_target}_{horizon}.csv",
    )

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
    )

    plt.title(f"SHAP Summary Plot - {target_type} {horizon}")
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )

    plt.title(f"SHAP Feature Importance - {target_type} {horizon}")
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap_value": mean_abs_shap,
        }
    ).sort_values("mean_absolute_shap_value", ascending=False)

    importance_df.to_csv(table_path, index=False)

    print("Saved SHAP summary plot:", summary_plot_path)
    print("Saved SHAP bar plot:", bar_plot_path)
    print("Saved SHAP importance table:", table_path)

    print("\nTop 10 important features:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def main():
    print("=" * 70)
    print("SHAP Explainability Analysis")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"], low_memory=False)
    metrics = pd.read_csv(PERFORMANCE_PATH)

    df["sensor_id"] = df["sensor_id"].astype(str)
    df["sensor_name"] = df["sensor_name"].astype(str)

    print("Dataset loaded:", df.shape)
    print("Performance table loaded:", metrics.shape)

    run_shap_for_target(
        df=df,
        metrics=metrics,
        target_type="DO mg/L",
        target_col=f"do_mgl_next_{HORIZON}",
        horizon=HORIZON,
    )

    run_shap_for_target(
        df=df,
        metrics=metrics,
        target_type="DO %",
        target_col=f"do_pct_next_{HORIZON}",
        horizon=HORIZON,
    )

    print("\nSHAP explainability complete.")


if __name__ == "__main__":
    main()