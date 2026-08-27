import os
import math
import joblib
import requests
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


AQUASENSOR_FINAL = "data/processed/aquasensor_final.csv"
PERFORMANCE_FILE = "data/processed/do_prediction_model_performance.csv"
LIVE_FORECAST_FILE = "data/processed/live_river_do_forecasts.csv"
MODELS_DIR = "models"

OUTPUT_FILE = "data/processed/live_shap_feature_importance.csv"
FIGURES_DIR = "figures/shap"

LATITUDE = 53.33
LONGITUDE = -1.65

TARGET_TYPE = "DO mg/L"
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


def get_best_model_name(target_type, horizon):
    metrics = pd.read_csv(PERFORMANCE_FILE)

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


def get_live_weather():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        "&hourly=temperature_2m,shortwave_radiation,cloud_cover"
        "&forecast_days=1"
        "&timezone=Europe/London"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()
    hourly = data["hourly"]

    weather = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "air_temperature_c": hourly["temperature_2m"],
            "sunshine_wm2": hourly["shortwave_radiation"],
            "cloud_cover_pct": hourly["cloud_cover"],
        }
    )

    now = pd.Timestamp.now()
    weather["time_diff"] = (weather["timestamp"] - now).abs()
    latest_weather = weather.sort_values("time_diff").iloc[0]

    return {
        "air_temperature_c": latest_weather["air_temperature_c"],
        "sunshine_wm2": latest_weather["sunshine_wm2"],
        "cloud_cover_pct": latest_weather["cloud_cover_pct"],
    }


def add_live_time_features(row):
    timestamp = pd.to_datetime(row["timestamp"])

    row["hour"] = timestamp.hour
    row["month"] = timestamp.month
    row["day_of_year"] = timestamp.dayofyear

    angle = 2 * math.pi * (row["day_of_year"] / 365.25)
    seasonal_signal = (math.cos(angle - math.pi) + 1) / 2
    clear_sky_signal = 1 - (row["cloud_cover_pct"] / 100)

    row["season_proxy"] = round(
        0.70 * seasonal_signal + 0.30 * clear_sky_signal,
        4
    )

    return row


def load_latest_live_input():
    df = pd.read_csv(AQUASENSOR_FINAL, parse_dates=["timestamp"], low_memory=False)

    df["sensor_id"] = df["sensor_id"].astype(str)
    df["sensor_name"] = df["sensor_name"].astype(str)

    latest_rows = (
        df.sort_values("timestamp")
        .groupby("sensor_id")
        .tail(1)
        .reset_index(drop=True)
    )

    live_weather = get_live_weather()

    rows = []

    for _, row in latest_rows.iterrows():
        row = row.copy()

        row["air_temperature_c"] = live_weather["air_temperature_c"]
        row["sunshine_wm2"] = live_weather["sunshine_wm2"]
        row["cloud_cover_pct"] = live_weather["cloud_cover_pct"]

        row = add_live_time_features(row)

        rows.append(row)

    live_input = pd.DataFrame(rows)

    return live_input


def run_live_shap():
    print("=" * 70)
    print("LIVE SHAP EXPLAINABILITY")
    print("=" * 70)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    model_name = get_best_model_name(TARGET_TYPE, HORIZON)
    model = load_model(model_name, TARGET_TYPE, HORIZON)

    print("Target:", TARGET_TYPE)
    print("Horizon:", HORIZON)
    print("Model:", model_name)

    live_input = load_latest_live_input()

    preprocessor = model.named_steps["preprocessor"]
    trained_model = model.named_steps["model"]

    X_live = live_input[FEATURES].copy()
    X_transformed = preprocessor.transform(X_live)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    explainer = shap.Explainer(trained_model, X_transformed)
    shap_values = explainer(X_transformed)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap_value": mean_abs_shap,
        }
    ).sort_values("mean_absolute_shap_value", ascending=False)

    importance_df["target_type"] = TARGET_TYPE
    importance_df["horizon"] = HORIZON
    importance_df["model_used"] = model_name
    importance_df["shap_run_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    importance_df.to_csv(OUTPUT_FILE, index=False)

    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(10).sort_values("mean_absolute_shap_value")

    plt.barh(
        top_features["feature"],
        top_features["mean_absolute_shap_value"]
    )

    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title(f"Live SHAP Feature Importance - {TARGET_TYPE} {HORIZON}")
    plt.tight_layout()

    plot_path = os.path.join(FIGURES_DIR, "live_shap_bar.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nTop live SHAP features:")
    print(importance_df.head(10).to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(plot_path)

    print("\nLive SHAP explainability complete.")


if __name__ == "__main__":
    run_live_shap()