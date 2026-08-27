import os
import time
import math
import joblib
import requests
import pandas as pd
from datetime import datetime


AQUASENSOR_FINAL = "data/processed/aquasensor_final.csv"
PERFORMANCE_FILE = "data/processed/do_prediction_model_performance.csv"
OUTPUT_FILE = "data/processed/live_river_do_forecasts.csv"
MODELS_DIR = "models"

INTERVAL_SECONDS = 15 * 60

LATITUDE = 53.33
LONGITUDE = -1.65

LOW_DO_THRESHOLD = 4.0
RAPID_DO_DROP_THRESHOLD = 2.0
TEMP_JUMP_THRESHOLD = 5.0
GAP_THRESHOLD_MINUTES = 30

HORIZONS = [
    "15min", "30min", "45min", "60min",
    "75min", "90min", "105min", "120min"
]

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

    season_map = {
        12: 0, 1: 0, 2: 0,
        3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3,
    }

    row["season"] = season_map[row["month"]]

    angle = 2 * math.pi * (row["day_of_year"] / 365.25)
    seasonal_signal = (math.cos(angle - math.pi) + 1) / 2
    clear_sky_signal = 1 - (row["cloud_cover_pct"] / 100)

    row["season_proxy"] = round(
        0.70 * seasonal_signal + 0.30 * clear_sky_signal,
        4
    )

    return row


def load_sensor_data():
    df = pd.read_csv(AQUASENSOR_FINAL, parse_dates=["timestamp"], low_memory=False)

    df["sensor_id"] = df["sensor_id"].astype(str)
    df["sensor_name"] = df["sensor_name"].astype(str)

    df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    return df


def load_latest_sensor_rows(sensor_df):
    latest_rows = (
        sensor_df.sort_values("timestamp")
        .groupby("sensor_id")
        .tail(1)
        .reset_index(drop=True)
    )

    return latest_rows


def get_previous_sensor_row(sensor_df, sensor_id, latest_timestamp):
    sensor_history = sensor_df[
        (sensor_df["sensor_id"] == sensor_id) &
        (sensor_df["timestamp"] < latest_timestamp)
    ].sort_values("timestamp")

    if sensor_history.empty:
        return None

    return sensor_history.tail(1).iloc[0]


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


def detect_live_anomaly(row, previous_row, result):
    reasons = []
    level = "Normal"
    status = "Normal"

    current_do = row["dissolved_oxygen_mgl"]
    current_temp = row["temperature"]
    current_time = pd.to_datetime(row["timestamp"])

    if current_do < LOW_DO_THRESHOLD:
        status = "Anomaly Detected"
        level = "High"
        reasons.append(f"Current DO is below {LOW_DO_THRESHOLD} mg/L")

    if previous_row is not None:
        previous_do = previous_row["dissolved_oxygen_mgl"]
        previous_temp = previous_row["temperature"]
        previous_time = pd.to_datetime(previous_row["timestamp"])

        do_drop = previous_do - current_do
        temp_jump = abs(current_temp - previous_temp)
        gap_minutes = (current_time - previous_time).total_seconds() / 60

        if do_drop >= RAPID_DO_DROP_THRESHOLD:
            status = "Anomaly Detected"
            level = "Medium" if level != "High" else level
            reasons.append(f"Rapid DO drop of {do_drop:.2f} mg/L since previous reading")

        if temp_jump >= TEMP_JUMP_THRESHOLD:
            status = "Anomaly Detected"
            level = "Medium" if level != "High" else level
            reasons.append(f"Sudden temperature change of {temp_jump:.2f} °C")

        if gap_minutes >= GAP_THRESHOLD_MINUTES:
            status = "Anomaly Detected"
            level = "Low" if level == "Normal" else level
            reasons.append(f"Sensor communication gap of {gap_minutes:.1f} minutes")

    predicted_risk_status = "No Predicted Risk"
    predicted_risk_reason = "Predicted DO remains above threshold"

    predicted_do_values = [
        result.get(f"predicted_do_mgl_{horizon}")
        for horizon in HORIZONS
        if result.get(f"predicted_do_mgl_{horizon}") is not None
    ]

    risky_predictions = [
        value for value in predicted_do_values
        if value < LOW_DO_THRESHOLD
    ]

    if risky_predictions:
        predicted_risk_status = "Future DO Risk"
        predicted_risk_reason = (
            f"At least one future predicted DO value is below {LOW_DO_THRESHOLD} mg/L"
        )
        if level == "Normal":
            level = "Medium"
            status = "Anomaly Risk Detected"

    if not reasons:
        reasons.append("No live anomaly detected")

    return {
        "live_anomaly_status": status,
        "live_anomaly_level": level,
        "live_anomaly_reason": "; ".join(reasons),
        "predicted_risk_status": predicted_risk_status,
        "predicted_risk_reason": predicted_risk_reason,
    }


def make_live_prediction():
    live_weather = get_live_weather()
    sensor_df = load_sensor_data()
    latest_rows = load_latest_sensor_rows(sensor_df)

    prediction_run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    forecast_rows = []

    print("\n" + "=" * 80)
    print("LIVE RIVER DO FORECAST WITH LIVE ANOMALY DETECTION")
    print("=" * 80)
    print(f"Prediction run time: {prediction_run_time}")
    print("Weather source: Open-Meteo Forecast API")
    print(f"Weather location: lat={LATITUDE}, lon={LONGITUDE}")
    print(f"Latest sensor rows used: {len(latest_rows)}")

    for _, row in latest_rows.iterrows():
        row = row.copy()

        row["air_temperature_c"] = live_weather["air_temperature_c"]
        row["sunshine_wm2"] = live_weather["sunshine_wm2"]
        row["cloud_cover_pct"] = live_weather["cloud_cover_pct"]

        row = add_live_time_features(row)

        input_row = pd.DataFrame([row[FEATURES]])

        result = {
            "prediction_run_time": prediction_run_time,
            "latest_sensor_timestamp": row["timestamp"],
            "sensor_id": row["sensor_id"],
            "sensor_name": row["sensor_name"],
            "current_do_mgl": row["dissolved_oxygen_mgl"],
            "current_do_pct": row["dissolved_oxygen_pct"],
            "pollution_alert": row["pollution_alert"],
            "anomaly_type": row["anomaly_type"],
        }

        for horizon in HORIZONS:
            mgl_model_name = get_best_model_name("DO mg/L", horizon)
            pct_model_name = get_best_model_name("DO %", horizon)

            mgl_model = load_model(mgl_model_name, "DO mg/L", horizon)
            pct_model = load_model(pct_model_name, "DO %", horizon)

            result[f"predicted_do_mgl_{horizon}"] = mgl_model.predict(input_row)[0]
            result[f"predicted_do_pct_{horizon}"] = pct_model.predict(input_row)[0]

        result["air_temperature_c"] = row["air_temperature_c"]
        result["sunshine_wm2"] = row["sunshine_wm2"]
        result["cloud_cover_pct"] = row["cloud_cover_pct"]

        previous_row = get_previous_sensor_row(
            sensor_df=sensor_df,
            sensor_id=row["sensor_id"],
            latest_timestamp=row["timestamp"],
        )

        anomaly_result = detect_live_anomaly(row, previous_row, result)
        result.update(anomaly_result)

        forecast_rows.append(result)

    forecast_df = pd.DataFrame(forecast_rows)

    final_columns = [
        "prediction_run_time",
        "latest_sensor_timestamp",
        "sensor_id",
        "sensor_name",
        "current_do_mgl",
        "current_do_pct",
        "pollution_alert",
        "anomaly_type",
        "predicted_do_mgl_15min",
        "predicted_do_pct_15min",
        "predicted_do_mgl_30min",
        "predicted_do_pct_30min",
        "predicted_do_mgl_45min",
        "predicted_do_pct_45min",
        "predicted_do_mgl_60min",
        "predicted_do_pct_60min",
        "predicted_do_mgl_75min",
        "predicted_do_pct_75min",
        "predicted_do_mgl_90min",
        "predicted_do_pct_90min",
        "predicted_do_mgl_105min",
        "predicted_do_pct_105min",
        "predicted_do_mgl_120min",
        "predicted_do_pct_120min",
        "air_temperature_c",
        "sunshine_wm2",
        "cloud_cover_pct",
        "live_anomaly_status",
        "live_anomaly_level",
        "live_anomaly_reason",
        "predicted_risk_status",
        "predicted_risk_reason",
    ]

    forecast_df = forecast_df[final_columns]

    if os.path.exists(OUTPUT_FILE):
        old_df = pd.read_csv(OUTPUT_FILE)

        for col in final_columns:
            if col not in old_df.columns:
                old_df[col] = pd.NA

        for col in old_df.columns:
            if col not in forecast_df.columns:
                forecast_df[col] = pd.NA

        final_df = pd.concat([old_df, forecast_df], ignore_index=True)
    else:
        final_df = forecast_df

    final_df.to_csv(OUTPUT_FILE, index=False)

    display_cols = [
        "sensor_name",
        "current_do_mgl",
        "current_do_pct",
        "predicted_do_mgl_15min",
        "predicted_do_mgl_60min",
        "predicted_do_mgl_120min",
        "live_anomaly_status",
        "live_anomaly_level",
        "predicted_risk_status",
    ]

    print("\nLive predictions with anomaly detection:")
    print(forecast_df[display_cols].to_string(index=False))

    print(f"\nSaved live forecasts to: {OUTPUT_FILE}")


def main():
    print("=" * 80)
    print("AquaSensor Live DO Prediction + Live Anomaly Detection System")
    print("Predicts DO mg/L and DO % every 15 minutes")
    print("Horizons: 15, 30, 45, 60, 75, 90, 105, 120 minutes")
    print("=" * 80)

    while True:
        make_live_prediction()

        print("\nWaiting 15 minutes for next prediction...")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()