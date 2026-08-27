import requests
import pandas as pd
import os
from io import StringIO
from datetime import datetime

# ── Station IDs ───────────────────────────────────────────
# E01786A  = your AquaSensor (water QUALITY: temp, pH, DO etc.)
# L2604    = EA gauging station, River Derwent at Mytham Bridge
#            (nearest level/flow station to Hathersage, ~2km away)

QUALITY_STATION = "E01786A"
LEVEL_STATION   = "L2604"
BASE_HYDRO      = "https://environment.data.gov.uk/hydrology"
BASE_FLOOD      = "https://environment.data.gov.uk/flood-monitoring"
FROM = "2024-01-01"
TO   = datetime.today().strftime("%Y-%m-%d")


def download_level_and_flow():
    """Download river level + flow from the Mytham Bridge gauging station."""
    measures = {
        "river_level_m":   f"{BASE_FLOOD}/id/measures/L2604-level-stage-i-15_min-m/readings.csv",
        "river_flow_m3s":  f"{BASE_FLOOD}/id/measures/L2604-flow-i-15_min-m3_s/readings.csv",
    }

    frames = []
    for col_name, url in measures.items():
        full_url = f"{url}?startdate={FROM}&enddate={TO}&_limit=100000"
        print(f"Downloading {col_name}...")
        try:
            r = requests.get(full_url, timeout=60)
            if r.status_code != 200:
                print(f"  ERROR: {r.status_code} — {url}")
                continue
            df = pd.read_csv(StringIO(r.text))
            df = df.rename(columns={"dateTime": "timestamp", "value": col_name})
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[["timestamp", col_name]]
            print(f"  Got {len(df)} rows")
            frames.append(df)
        except Exception as e:
            print(f"  Failed: {e}")

    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    return pd.merge(frames[0], frames[1], on="timestamp", how="outer")


def download_quality():
    """Download water quality measures from your AquaSensor station E01786A."""
    measure_ids = {
        "temp_C":     "E01786A-temp-i-subdaily-C",
        "turbidity":  "E01786A-turb-i-subdaily-ntu",
        "do_mgL":     "E01786A-do-i-subdaily-mgL",
        "ph":         "E01786A-ph-i-subdaily",
        "do_pct":     "E01786A-do-i-subdaily-pct",
        "cond_uS":    "E01786A-cond-i-subdaily-uS",
        "ammonia":    "E01786A-amm-i-subdaily-mgL",
    }

    frames = []
    for col_name, measure_id in measure_ids.items():
        url = (
            f"{BASE_HYDRO}/id/measures/{measure_id}/readings.csv"
            f"?mineq-date={FROM}&maxeq-date={TO}&_limit=100000"
        )
        print(f"Downloading {col_name}...")
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                print(f"  ERROR: {r.status_code}")
                continue
            df = pd.read_csv(StringIO(r.text))
            df = df.rename(columns={"dateTime": "timestamp", "value": col_name})
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[["timestamp", col_name]]
            print(f"  Got {len(df)} rows")
            frames.append(df)
        except Exception as e:
            print(f"  Failed: {e}")

    if not frames:
        return None
    combined = frames[0]
    for f in frames[1:]:
        combined = pd.merge(combined, f, on="timestamp", how="outer")
    return combined


def main():
    print("=" * 50)
    print("EA Government River Data Download")
    print("Level/Flow: Derwent at Mytham Bridge (L2604)")
    print("Quality:    AquaSensor E01786A, Hathersage")
    print("=" * 50)

    os.makedirs("data/raw", exist_ok=True)

    # ── Level + Flow ──────────────────────────────────────
    print("\n── River Level & Flow ──")
    hydro = download_level_and_flow()
    if hydro is not None:
        hydro = hydro.sort_values("timestamp")
        hydro.to_csv("data/raw/govt_river_derwent.csv", index=False)
        print(f"\nSaved: data/raw/govt_river_derwent.csv  ({len(hydro)} rows)")
        print(hydro.head())
    else:
        print("ERROR: No level/flow data downloaded")

    # ── Water Quality ─────────────────────────────────────
    print("\n── Water Quality ──")
    quality = download_quality()
    if quality is not None:
        quality = quality.sort_values("timestamp")
        quality.to_csv("data/raw/govt_water_quality.csv", index=False)
        print(f"\nSaved: data/raw/govt_water_quality.csv  ({len(quality)} rows)")
        print(quality.head())
    else:
        print("ERROR: No quality data downloaded")


main()