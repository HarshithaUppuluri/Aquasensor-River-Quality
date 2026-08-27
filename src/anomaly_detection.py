import os
import time
import pandas as pd
from datetime import datetime


LIVE_FORECAST_FILE = "data/processed/live_river_do_forecasts.csv"

LIVE_ANOMALY_OUTPUT = "data/processed/live_anomaly_detection_results.csv"
LIVE_ANOMALY_SUMMARY = "data/processed/live_anomaly_summary.csv"

INTERVAL_SECONDS = 15 * 60

LOW_DO_THRESHOLD = 4.0
WARNING_DO_THRESHOLD = 5.0
RAPID_DROP_THRESHOLD = 2.0


def detect_live_anomaly(row):
    reasons = []
    status = "Normal"
    level = "Normal"

    current_do = row["current_do_mgl"]

    future_cols = [
        "predicted_do_mgl_15min",
        "predicted_do_mgl_30min",
        "predicted_do_mgl_45min",
        "predicted_do_mgl_60min",
        "predicted_do_mgl_75min",
        "predicted_do_mgl_90min",
        "predicted_do_mgl_105min",
        "predicted_do_mgl_120min",
    ]

    future_values = [
        row[col] for col in future_cols
        if col in row.index and pd.notna(row[col])
    ]

    if current_do < LOW_DO_THRESHOLD:
        status = "Anomaly Detected"
        level = "High"
        reasons.append(f"Current DO is below {LOW_DO_THRESHOLD} mg/L")

    elif current_do < WARNING_DO_THRESHOLD:
        status = "Warning"
        level = "Medium"
        reasons.append(f"Current DO is below warning threshold of {WARNING_DO_THRESHOLD} mg/L")

    if future_values:
        min_future_do = min(future_values)

        if min_future_do < LOW_DO_THRESHOLD:
            status = "Future Risk Detected"
            level = "High"
            reasons.append(
                f"Predicted future DO falls below {LOW_DO_THRESHOLD} mg/L"
            )

        elif min_future_do < WARNING_DO_THRESHOLD and level == "Normal":
            status = "Future Warning"
            level = "Medium"
            reasons.append(
                f"Predicted future DO falls below warning threshold of {WARNING_DO_THRESHOLD} mg/L"
            )

        if current_do - min_future_do >= RAPID_DROP_THRESHOLD:
            if level != "High":
                level = "Medium"
            status = "Future Drop Risk"
            reasons.append(
                f"Predicted DO drop of {current_do - min_future_do:.2f} mg/L"
            )

    if not reasons:
        reasons.append("No live anomaly detected")

    return status, level, "; ".join(reasons)


def run_live_anomaly_detection():
    if not os.path.exists(LIVE_FORECAST_FILE):
        raise FileNotFoundError(
            f"{LIVE_FORECAST_FILE} not found. Run live_predict.py first."
        )

    df = pd.read_csv(LIVE_FORECAST_FILE)

    if df.empty:
        raise ValueError("live_river_do_forecasts.csv is empty.")

    latest_run_time = df["prediction_run_time"].max()

    latest_df = df[df["prediction_run_time"] == latest_run_time].copy()

    results = []

    for _, row in latest_df.iterrows():
        status, level, reason = detect_live_anomaly(row)

        result = {
            "anomaly_check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction_run_time": row["prediction_run_time"],
            "latest_sensor_timestamp": row["latest_sensor_timestamp"],
            "sensor_id": row["sensor_id"],
            "sensor_name": row["sensor_name"],
            "current_do_mgl": row["current_do_mgl"],
            "current_do_pct": row["current_do_pct"],
            "predicted_do_mgl_15min": row.get("predicted_do_mgl_15min"),
            "predicted_do_mgl_30min": row.get("predicted_do_mgl_30min"),
            "predicted_do_mgl_45min": row.get("predicted_do_mgl_45min"),
            "predicted_do_mgl_60min": row.get("predicted_do_mgl_60min"),
            "predicted_do_mgl_75min": row.get("predicted_do_mgl_75min"),
            "predicted_do_mgl_90min": row.get("predicted_do_mgl_90min"),
            "predicted_do_mgl_105min": row.get("predicted_do_mgl_105min"),
            "predicted_do_mgl_120min": row.get("predicted_do_mgl_120min"),
            "air_temperature_c": row.get("air_temperature_c"),
            "sunshine_wm2": row.get("sunshine_wm2"),
            "cloud_cover_pct": row.get("cloud_cover_pct"),
            "live_anomaly_status": status,
            "live_anomaly_level": level,
            "live_anomaly_reason": reason,
        }

        results.append(result)

    results_df = pd.DataFrame(results)

    if os.path.exists(LIVE_ANOMALY_OUTPUT):
        old_df = pd.read_csv(LIVE_ANOMALY_OUTPUT)
        final_df = pd.concat([old_df, results_df], ignore_index=True)
    else:
        final_df = results_df

    final_df.to_csv(LIVE_ANOMALY_OUTPUT, index=False)

    summary_df = (
        final_df.groupby(["sensor_name", "live_anomaly_status", "live_anomaly_level"])
        .size()
        .reset_index(name="count")
    )

    summary_df.to_csv(LIVE_ANOMALY_SUMMARY, index=False)

    print("\n" + "=" * 80)
    print("LIVE ANOMALY DETECTION")
    print("=" * 80)
    print(f"Latest prediction run checked: {latest_run_time}")
    print("\nLive anomaly results:")
    print(
        results_df[
            [
                "sensor_name",
                "current_do_mgl",
                "predicted_do_mgl_15min",
                "predicted_do_mgl_60min",
                "predicted_do_mgl_120min",
                "live_anomaly_status",
                "live_anomaly_level",
                "live_anomaly_reason",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved live anomaly results to: {LIVE_ANOMALY_OUTPUT}")
    print(f"Saved live anomaly summary to: {LIVE_ANOMALY_SUMMARY}")


def main():
    print("=" * 80)
    print("AquaSensor Live Anomaly Detection System")
    print("Checks latest live DO predictions every 15 minutes")
    print("=" * 80)

    while True:
        run_live_anomaly_detection()

        print("\nWaiting 15 minutes for next anomaly check...")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()