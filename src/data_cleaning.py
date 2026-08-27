import os
import glob
import pandas as pd
import numpy as np


AQUASENSOR_DIR = "data/raw/Aquasensor Data 1yr"
GOVT_DIR = "data/raw/Govt.Data"
WEATHER_FILE = "data/raw/weather_data.csv"

OUTPUT_DIR = "data/processed"

AQUASENSOR_OUTPUT = "data/processed/aquasensor_cleaned.csv"
GOVT_OUTPUT = "data/processed/govt_cleaned.csv"
WEATHER_OUTPUT = "data/processed/weather_cleaned.csv"
REPORT_OUTPUT = "data/processed/data_cleaning_report.csv"


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


def clean_aquasensor():
    files = glob.glob(os.path.join(AQUASENSOR_DIR, "*.csv"))

    if not files:
        raise FileNotFoundError(f"No AquaSensor files found in {AQUASENSOR_DIR}")

    frames = []
    report = []

    sensor_name_map = {
        "sensor022": "Derwent 13",
        "941115": "Derwent 13-50",
        "941205": "Derwent 21",
    }

    for file in files:
        print(f"Cleaning AquaSensor file: {file}")

        df = pd.read_csv(file, low_memory=False)
        original_rows = len(df)

        df = clean_column_names(df)

        rename_map = {
            "name": "sensor_id",
            "mg_l": "dissolved_oxygen_mgl",
            "percent": "dissolved_oxygen_pct",
        }

        df = df.rename(columns=rename_map)

        if "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                errors="coerce",
                dayfirst=True,
            )
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        required = [
            "timestamp",
            "sensor_id",
            "temperature",
            "dissolved_oxygen_mgl",
            "dissolved_oxygen_pct",
        ]

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{file} missing columns: {missing}")

        df["sensor_id"] = df["sensor_id"].astype(str).str.strip()
        df["sensor_name"] = df["sensor_id"].map(sensor_name_map).fillna(df["sensor_id"])

        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        df["dissolved_oxygen_mgl"] = pd.to_numeric(df["dissolved_oxygen_mgl"], errors="coerce")
        df["dissolved_oxygen_pct"] = pd.to_numeric(df["dissolved_oxygen_pct"], errors="coerce")

        before_missing = len(df)
        df = df.dropna(
            subset=[
                "timestamp",
                "sensor_id",
                "temperature",
                "dissolved_oxygen_mgl",
                "dissolved_oxygen_pct",
            ]
        )
        removed_missing = before_missing - len(df)

        before_duplicates = len(df)
        df = df.drop_duplicates(subset=["sensor_id", "timestamp"])
        removed_duplicates = before_duplicates - len(df)

        before_invalid = len(df)
        df = df[
            (df["temperature"].between(-5, 40)) &
            (df["dissolved_oxygen_mgl"].between(0, 25)) &
            (df["dissolved_oxygen_pct"].between(0, 250))
        ]
        removed_invalid = before_invalid - len(df)

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

        report.append({
            "dataset": "AquaSensor",
            "file": os.path.basename(file),
            "original_rows": original_rows,
            "cleaned_rows": len(df),
            "removed_missing": removed_missing,
            "removed_duplicates": removed_duplicates,
            "removed_invalid_values": removed_invalid,
        })

    final_df = pd.concat(frames, ignore_index=True)
    final_df = final_df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    return final_df, report


def clean_govt():
    files = glob.glob(os.path.join(GOVT_DIR, "*.csv"))

    if not files:
        print(f"No Government files found in {GOVT_DIR}")
        return pd.DataFrame(), []

    frames = []
    report = []

    for file in files:
        print(f"Cleaning Government file: {file}")

        df = pd.read_csv(file, low_memory=False)
        original_rows = len(df)

        df = clean_column_names(df)

        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        elif "date_time" in df.columns:
            df = df.rename(columns={"date_time": "timestamp"})
        elif "date" in df.columns and "time" in df.columns:
            df["timestamp"] = df["date"].astype(str) + " " + df["time"].astype(str)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        before_missing = len(df)
        if "timestamp" in df.columns:
            df = df.dropna(subset=["timestamp"])
        if "value" in df.columns:
            df = df.dropna(subset=["value"])
        removed_missing = before_missing - len(df)

        before_duplicates = len(df)
        df = df.drop_duplicates()
        removed_duplicates = before_duplicates - len(df)

        before_invalid = len(df)
        if "value" in df.columns:
            df = df[df["value"].between(-1000, 10000)]
        removed_invalid = before_invalid - len(df)

        df["source_file"] = os.path.basename(file)

        frames.append(df)

        report.append({
            "dataset": "Government",
            "file": os.path.basename(file),
            "original_rows": original_rows,
            "cleaned_rows": len(df),
            "removed_missing": removed_missing,
            "removed_duplicates": removed_duplicates,
            "removed_invalid_values": removed_invalid,
        })

    final_df = pd.concat(frames, ignore_index=True)
    return final_df, report


def clean_weather():
    if not os.path.exists(WEATHER_FILE):
        print(f"Weather file not found: {WEATHER_FILE}")
        return pd.DataFrame(), []

    print(f"Cleaning weather file: {WEATHER_FILE}")

    df = pd.read_csv(WEATHER_FILE, low_memory=False)
    original_rows = len(df)

    df = clean_column_names(df)

    required = [
        "timestamp",
        "air_temperature_c",
        "sunshine_wm2",
        "cloud_cover_pct",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Weather file missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["air_temperature_c"] = pd.to_numeric(df["air_temperature_c"], errors="coerce")
    df["sunshine_wm2"] = pd.to_numeric(df["sunshine_wm2"], errors="coerce")
    df["cloud_cover_pct"] = pd.to_numeric(df["cloud_cover_pct"], errors="coerce")

    before_missing = len(df)
    df = df.dropna(
        subset=[
            "timestamp",
            "air_temperature_c",
            "sunshine_wm2",
            "cloud_cover_pct",
        ]
    )
    removed_missing = before_missing - len(df)

    before_duplicates = len(df)
    df = df.drop_duplicates(subset=["timestamp"])
    removed_duplicates = before_duplicates - len(df)

    before_invalid = len(df)
    df = df[
        (df["air_temperature_c"].between(-20, 45)) &
        (df["sunshine_wm2"].between(0, 1400)) &
        (df["cloud_cover_pct"].between(0, 100))
    ]
    removed_invalid = before_invalid - len(df)

    df = df[
        [
            "timestamp",
            "air_temperature_c",
            "sunshine_wm2",
            "cloud_cover_pct",
        ]
    ]

    df = df.sort_values("timestamp").reset_index(drop=True)

    report = [{
        "dataset": "Weather",
        "file": os.path.basename(WEATHER_FILE),
        "original_rows": original_rows,
        "cleaned_rows": len(df),
        "removed_missing": removed_missing,
        "removed_duplicates": removed_duplicates,
        "removed_invalid_values": removed_invalid,
    }]

    return df, report


def main():
    print("=" * 60)
    print("DATA CLEANING PIPELINE")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    aquasensor_df, aquasensor_report = clean_aquasensor()
    govt_df, govt_report = clean_govt()
    weather_df, weather_report = clean_weather()

    aquasensor_df.to_csv(AQUASENSOR_OUTPUT, index=False)
    print(f"\nSaved: {AQUASENSOR_OUTPUT}")

    if not govt_df.empty:
        govt_df.to_csv(GOVT_OUTPUT, index=False)
        print(f"Saved: {GOVT_OUTPUT}")

    if not weather_df.empty:
        weather_df.to_csv(WEATHER_OUTPUT, index=False)
        print(f"Saved: {WEATHER_OUTPUT}")

    report_df = pd.DataFrame(
        aquasensor_report + govt_report + weather_report
    )
    report_df.to_csv(REPORT_OUTPUT, index=False)
    print(f"Saved: {REPORT_OUTPUT}")

    print("\nAquaSensor cleaned:")
    print(f"Rows: {len(aquasensor_df)}")
    print(f"Start: {aquasensor_df['timestamp'].min()}")
    print(f"End:   {aquasensor_df['timestamp'].max()}")

    if not govt_df.empty and "timestamp" in govt_df.columns:
        print("\nGovernment cleaned:")
        print(f"Rows: {len(govt_df)}")
        print(f"Start: {govt_df['timestamp'].min()}")
        print(f"End:   {govt_df['timestamp'].max()}")

    if not weather_df.empty:
        print("\nWeather cleaned:")
        print(f"Rows: {len(weather_df)}")
        print(f"Start: {weather_df['timestamp'].min()}")
        print(f"End:   {weather_df['timestamp'].max()}")

    print("\nData cleaning complete.")


if __name__ == "__main__":
    main()