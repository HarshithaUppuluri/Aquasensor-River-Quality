import os
import glob
import joblib
import requests
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RECENT_DIR = "data/raw/recent_aquasensor"
WEATHER_CLEANED = "data/processed/weather_cleaned.csv"
PERFORMANCE_FILE = "data/processed/do_prediction_model_performance.csv"
MODELS_DIR = "models"

RECENT_CLEANED_OUTPUT = "data/processed/recent_aquasensor_cleaned.csv"
RECENT_FINAL_OUTPUT = "data/processed/recent_aquasensor_final.csv"
RECENT_PREDICTIONS_OUTPUT = "data/processed/recent_data_predictions.csv"
RECENT_METRICS_OUTPUT = "data/processed/recent_data_model_performance.csv"
RECENT_FORECASTS_OUTPUT = "data/processed/recent_river_do_forecasts.csv"

LATITUDE = 53.33
LONGITUDE = -1.65

HORIZONS = {
    "15min": 1,
    "30min": 2,
    "45min": 3,
    "60min": 4,
    "75min": 5,
    "90min": 6,
    "105min": 7,
    "120min": 8,
}

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


def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("%", "pct", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def infer_sensor_from_filename(file_path):
    name = os.path.basename(file_path).lower()

    if "13-50" in name or "13_50" in name:
        return "941115", "Derwent 13-50"

    if "21" in name:
        return "941205", "Derwent 21"

    if "13" in name:
        return "sensor022", "Derwent 13"

    return "unknown", "unknown"


def clean_recent_aquasensor():
    files = glob.glob(os.path.join(RECENT_DIR, "*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No recent AquaSensor CSV files found in {RECENT_DIR}"
        )

    frames = []

    for file in files:
        print(f"Loading recent AquaSensor file: {file}")

        df = pd.read_csv(file, low_memory=False)
        df = clean_column_names(df)

        sensor_id, sensor_name = infer_sensor_from_filename(file)

        rename_map = {
            "name": "raw_name",
            "mg_l": "dissolved_oxygen_mgl",
            "percent": "dissolved_oxygen_pct",
        }

        df = df.rename(columns=rename_map)

        required = [
            "date",
            "time",
            "temperature",
            "dissolved_oxygen_mgl",
            "dissolved_oxygen_pct",
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{file} is missing columns: {missing}")

        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
            dayfirst=True,
        )

        df["sensor_id"] = sensor_id
        df["sensor_name"] = sensor_name

        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        df["dissolved_oxygen_mgl"] = pd.to_numeric(
            df["dissolved_oxygen_mgl"], errors="coerce"
        )
        df["dissolved_oxygen_pct"] = pd.to_numeric(
            df["dissolved_oxygen_pct"], errors="coerce"
        )

        df = df.dropna(
            subset=[
                "timestamp",
                "temperature",
                "dissolved_oxygen_mgl",
                "dissolved_oxygen_pct",
            ]
        )

        df = df.drop_duplicates(subset=["sensor_id", "timestamp"])

        df = df[
            (df["temperature"].between(-5, 40)) &
            (df["dissolved_oxygen_mgl"].between(0, 25)) &
            (df["dissolved_oxygen_pct"].between(0, 250))
        ]

        df = df[
            [
                "timestamp",
                "sensor_id",
                "sensor_name",
                "temperature",
                "dissolved_oxygen_mgl",
                "dissolved_oxygen_pct",
            ]
        ]

        frames.append(df)

    final_df = pd.concat(frames, ignore_index=True)
    final_df = final_df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    final_df.to_csv(RECENT_CLEANED_OUTPUT, index=False)

    print(f"\nRecent AquaSensor cleaned rows: {len(final_df)}")
    print(f"Start: {final_df['timestamp'].min()}")
    print(f"End:   {final_df['timestamp'].max()}")
    print(f"Saved: {RECENT_CLEANED_OUTPUT}")

    return final_df


def download_weather_for_recent(start_date, end_date):
    print("\nDownloading weather for recent evaluation from Open-Meteo...")

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly=temperature_2m,shortwave_radiation,cloud_cover"
        "&timezone=Europe/London"
    )

    try:
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

        print("Open-Meteo weather downloaded successfully.")
        return weather

    except Exception as e:
        print(f"WARNING: Could not download Open-Meteo weather: {e}")
        print("Using existing weather_cleaned.csv instead.")
        return None


def merge_weather(df):
    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d")

    weather = download_weather_for_recent(start_date, end_date)

    if weather is None:
        if not os.path.exists(WEATHER_CLEANED):
            raise FileNotFoundError("weather_cleaned.csv not found.")

        weather = pd.read_csv(
            WEATHER_CLEANED,
            parse_dates=["timestamp"],
            low_memory=False,
        )

    df["ts_hr"] = df["timestamp"].dt.floor("h")
    weather["ts_hr"] = weather["timestamp"].dt.floor("h")

    df = df.merge(
        weather[
            [
                "ts_hr",
                "air_temperature_c",
                "sunshine_wm2",
                "cloud_cover_pct",
            ]
        ],
        on="ts_hr",
        how="left",
    )

    df = df.drop(columns=["ts_hr"])

    df["air_temperature_c"] = df["air_temperature_c"].ffill().bfill()
    df["sunshine_wm2"] = df["sunshine_wm2"].ffill().bfill()
    df["cloud_cover_pct"] = df["cloud_cover_pct"].ffill().bfill()

    print("Weather merged into recent data")
    return df


def add_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    df["season"] = df["month"].map(
        {
            12: 0,
            1: 0,
            2: 0,
            3: 1,
            4: 1,
            5: 1,
            6: 2,
            7: 2,
            8: 2,
            9: 3,
            10: 3,
            11: 3,
        }
    )

    df["cloud_cover_pct"] = df["cloud_cover_pct"].fillna(50)

    angle = 2 * np.pi * (df["day_of_year"] / 365.25)
    seasonal_signal = (np.cos(angle - np.pi) + 1) / 2
    clear_sky_signal = 1 - (df["cloud_cover_pct"] / 100)

    df["season_proxy"] = (
        0.70 * seasonal_signal + 0.30 * clear_sky_signal
    ).round(4)

    df["pollution_alert"] = (df["dissolved_oxygen_mgl"] < 4.0).astype(int)

    df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    df["anomaly_type"] = 0

    df["gap_min"] = (
        df.groupby("sensor_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    df.loc[df["gap_min"] > 24, "anomaly_type"] = 2
    df = df.drop(columns=["gap_min"])

    df["time_30min"] = df["timestamp"].dt.floor("30min")
    low_do = df[df["dissolved_oxygen_mgl"] < 4.0]

    if len(low_do) > 0:
        affected = (
            low_do.groupby("time_30min")["sensor_id"]
            .nunique()
            .reset_index()
        )

        affected.columns = ["time_30min", "sensors_affected"]

        df = df.merge(affected, on="time_30min", how="left")
        df["sensors_affected"] = df["sensors_affected"].fillna(0)

        low_mask = df["dissolved_oxygen_mgl"] < 4.0

        df.loc[
            low_mask & (df["sensors_affected"] == 1),
            "anomaly_type",
        ] = 1

        df.loc[
            low_mask & (df["sensors_affected"] >= 2),
            "anomaly_type",
        ] = 3

        df = df.drop(columns=["sensors_affected"])

    df = df.drop(columns=["time_30min"])

    print("Recent data feature engineering complete")
    return df


def add_targets(df):
    for horizon, step in HORIZONS.items():
        df[f"do_mgl_next_{horizon}"] = (
            df.groupby("sensor_id")["dissolved_oxygen_mgl"].shift(-step)
        )

        df[f"do_pct_next_{horizon}"] = (
            df.groupby("sensor_id")["dissolved_oxygen_pct"].shift(-step)
        )

    print("Recent data prediction targets created")
    return df


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


def evaluate_recent_data(df):
    metrics_rows = []
    prediction_rows = []

    for horizon in HORIZONS:
        target_pairs = [
            ("DO mg/L", f"do_mgl_next_{horizon}"),
            ("DO %", f"do_pct_next_{horizon}"),
        ]

        for target_type, target_col in target_pairs:
            model_df = df.dropna(subset=[target_col]).copy()

            if model_df.empty:
                continue

            X = model_df[FEATURES]
            y_true = model_df[target_col]

            model_name = get_best_model_name(target_type, horizon)
            model = load_model(model_name, target_type, horizon)

            y_pred = model.predict(X)

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            metrics_rows.append(
                {
                    "target_type": target_type,
                    "horizon": horizon,
                    "model_used": model_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "rows_evaluated": len(model_df),
                }
            )

            pred_df = model_df[
                [
                    "timestamp",
                    "sensor_id",
                    "sensor_name",
                    "dissolved_oxygen_mgl",
                    "dissolved_oxygen_pct",
                    "pollution_alert",
                    "anomaly_type",
                    "air_temperature_c",
                    "sunshine_wm2",
                    "cloud_cover_pct",
                ]
            ].copy()

            pred_df["target_type"] = target_type
            pred_df["horizon"] = horizon
            pred_df["model_used"] = model_name
            pred_df["actual_value"] = y_true.values
            pred_df["predicted_value"] = y_pred
            pred_df["absolute_error"] = np.abs(y_true.values - y_pred)

            prediction_rows.append(pred_df)

            print(
                f"{target_type:7s} {horizon:6s} {model_name:18s} "
                f"MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}"
            )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)

    metrics_df.to_csv(RECENT_METRICS_OUTPUT, index=False)
    predictions_df.to_csv(RECENT_PREDICTIONS_OUTPUT, index=False)

    print(f"\nSaved recent metrics: {RECENT_METRICS_OUTPUT}")
    print(f"Saved recent predictions: {RECENT_PREDICTIONS_OUTPUT}")

    return metrics_df, predictions_df


def create_recent_forecast_table(predictions_df):
    forecast_tables = []

    for target_type in ["DO mg/L", "DO %"]:
        subset = predictions_df[
            predictions_df["target_type"] == target_type
        ].copy()

        pivot = subset.pivot_table(
            index=[
                "timestamp",
                "sensor_id",
                "sensor_name",
                "dissolved_oxygen_mgl",
                "dissolved_oxygen_pct",
                "pollution_alert",
                "anomaly_type",
                "air_temperature_c",
                "sunshine_wm2",
                "cloud_cover_pct",
            ],
            columns="horizon",
            values="predicted_value",
            aggfunc="first",
        ).reset_index()

        if target_type == "DO mg/L":
            pivot = pivot.rename(
                columns={
                    "15min": "predicted_do_mgl_15min",
                    "30min": "predicted_do_mgl_30min",
                    "45min": "predicted_do_mgl_45min",
                    "60min": "predicted_do_mgl_60min",
                    "75min": "predicted_do_mgl_75min",
                    "90min": "predicted_do_mgl_90min",
                    "105min": "predicted_do_mgl_105min",
                    "120min": "predicted_do_mgl_120min",
                }
            )

        else:
            pivot = pivot.rename(
                columns={
                    "15min": "predicted_do_pct_15min",
                    "30min": "predicted_do_pct_30min",
                    "45min": "predicted_do_pct_45min",
                    "60min": "predicted_do_pct_60min",
                    "75min": "predicted_do_pct_75min",
                    "90min": "predicted_do_pct_90min",
                    "105min": "predicted_do_pct_105min",
                    "120min": "predicted_do_pct_120min",
                }
            )

        forecast_tables.append(pivot)

    final_forecast = forecast_tables[0].merge(
        forecast_tables[1],
        on=[
            "timestamp",
            "sensor_id",
            "sensor_name",
            "dissolved_oxygen_mgl",
            "dissolved_oxygen_pct",
            "pollution_alert",
            "anomaly_type",
            "air_temperature_c",
            "sunshine_wm2",
            "cloud_cover_pct",
        ],
        how="outer",
    )

    final_forecast = final_forecast.rename(
        columns={
            "timestamp": "latest_sensor_timestamp",
            "dissolved_oxygen_mgl": "current_do_mgl",
            "dissolved_oxygen_pct": "current_do_pct",
        }
    )

    final_forecast.insert(
        0,
        "prediction_run_time",
        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

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
    ]

    final_forecast = final_forecast[final_columns]
    final_forecast.to_csv(RECENT_FORECASTS_OUTPUT, index=False)

    print(f"Saved recent forecast table: {RECENT_FORECASTS_OUTPUT}")
    return final_forecast


def main():
    print("=" * 70)
    print("RECENT UNUSED AQUASENSOR DATA EVALUATION")
    print("=" * 70)

    df = clean_recent_aquasensor()
    df = merge_weather(df)
    df = add_features(df)
    df = add_targets(df)

    df.to_csv(RECENT_FINAL_OUTPUT, index=False)
    print(f"Saved recent final dataset: {RECENT_FINAL_OUTPUT}")

    print("\nEvaluating saved trained models on recent unused data...\n")
    metrics_df, predictions_df = evaluate_recent_data(df)

    create_recent_forecast_table(predictions_df)

    print("\nRecent unused-data evaluation complete.")


if __name__ == "__main__":
    main()