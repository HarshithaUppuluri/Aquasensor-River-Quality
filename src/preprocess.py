import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


AQUASENSOR_CLEANED = "data/processed/aquasensor_cleaned.csv"
WEATHER_CLEANED = "data/processed/weather_cleaned.csv"
GOVT_CLEANED = "data/processed/govt_cleaned.csv"

OUTPUT_FILE = "data/processed/aquasensor_final.csv"

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


def load_cleaned_aquasensor():
    if not os.path.exists(AQUASENSOR_CLEANED):
        raise FileNotFoundError("Run python src/data_cleaning.py first.")

    df = pd.read_csv(AQUASENSOR_CLEANED, parse_dates=["timestamp"], low_memory=False)

    df["sensor_id"] = df["sensor_id"].astype(str)
    df["sensor_name"] = df["sensor_name"].astype(str)

    df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    print(f"AquaSensor cleaned data loaded: {len(df)} rows")
    return df


def merge_weather(df):
    if not os.path.exists(WEATHER_CLEANED):
        raise FileNotFoundError("weather_cleaned.csv not found. Run data_cleaning.py first.")

    weather = pd.read_csv(WEATHER_CLEANED, parse_dates=["timestamp"], low_memory=False)

    df["ts_hr"] = df["timestamp"].dt.floor("h")
    weather["ts_hr"] = weather["timestamp"].dt.floor("h")

    df = df.merge(
        weather[["ts_hr", "air_temperature_c", "sunshine_wm2", "cloud_cover_pct"]],
        on="ts_hr",
        how="left",
    )

    df = df.drop(columns=["ts_hr"])

    print("Weather data merged")
    return df


def load_govt_for_context():
    if not os.path.exists(GOVT_CLEANED):
        print("Govt cleaned file not found. Skipping Govt context.")
        return None

    govt = pd.read_csv(GOVT_CLEANED, parse_dates=["timestamp"], low_memory=False)

    print(f"Government cleaned data available for comparison: {len(govt)} rows")
    return govt


def add_time_features(df):
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

    print("Time features added")
    return df


def add_season_proxy(df):
    df["cloud_cover_pct"] = df["cloud_cover_pct"].fillna(50)

    angle = 2 * np.pi * (df["day_of_year"] / 365.25)
    seasonal_signal = (np.cos(angle - np.pi) + 1) / 2
    clear_sky_signal = 1 - (df["cloud_cover_pct"] / 100)

    df["season_proxy"] = (
        0.70 * seasonal_signal + 0.30 * clear_sky_signal
    ).round(4)

    print("Season proxy created")
    return df


def add_pollution_alert(df, threshold=4.0):
    df["pollution_alert"] = (
        df["dissolved_oxygen_mgl"] < threshold
    ).astype(int)

    print(f"Pollution alerts created: {df['pollution_alert'].sum()}")
    return df


def add_anomaly_type(df, threshold=4.0):
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
    low_do = df[df["dissolved_oxygen_mgl"] < threshold]

    if len(low_do) > 0:
        affected = (
            low_do.groupby("time_30min")["sensor_id"]
            .nunique()
            .reset_index()
        )

        affected.columns = ["time_30min", "sensors_affected"]

        df = df.merge(affected, on="time_30min", how="left")
        df["sensors_affected"] = df["sensors_affected"].fillna(0)

        low_mask = df["dissolved_oxygen_mgl"] < threshold

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

    print("Anomaly type created")
    return df


def add_prediction_targets(df):
    for horizon, step in HORIZONS.items():
        df[f"do_mgl_next_{horizon}"] = (
            df.groupby("sensor_id")["dissolved_oxygen_mgl"].shift(-step)
        )

        df[f"do_pct_next_{horizon}"] = (
            df.groupby("sensor_id")["dissolved_oxygen_pct"].shift(-step)
        )

    print("DO mg/L and DO % prediction targets created")

    for horizon in HORIZONS:
        print(
            f"{horizon}: "
            f"mg/L usable={df[f'do_mgl_next_{horizon}'].notna().sum()}, "
            f"% usable={df[f'do_pct_next_{horizon}'].notna().sum()}"
        )

    return df


def impute_feature_columns(df):
    feature_cols = [
        "temperature",
        "air_temperature_c",
        "sunshine_wm2",
        "hour",
        "season_proxy",
    ]

    imputer = KNNImputer(n_neighbors=5)
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    print("Missing feature values imputed")
    return df


def save_final_dataset(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Final modelling dataset saved: {OUTPUT_FILE}")


def main():
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("Using cleaned AquaSensor + cleaned weather + Govt context")
    print("=" * 60)

    df = load_cleaned_aquasensor()
    govt = load_govt_for_context()

    df = merge_weather(df)
    df = add_time_features(df)
    df = add_season_proxy(df)
    df = add_pollution_alert(df)
    df = add_anomaly_type(df)
    df = add_prediction_targets(df)
    df = impute_feature_columns(df)

    save_final_dataset(df)

    print("\nFinal dataset summary:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Start: {df['timestamp'].min()}")
    print(f"End:   {df['timestamp'].max()}")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()