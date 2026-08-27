import requests
import pandas as pd
import os
from io import StringIO

# ── API login details ────────────────────────────────────
API_URL  = "https://api.aquasensor.co.uk/aq.php"
USERNAME = "shu"
TOKEN    = "aebbf6305f9fce1d5591ee05a3448eff"

# ── The 3 sensors access ─────────────────────────
SENSORS = {
    "sensor022": "Derwent_13",
    "sensor044": "Derwent_13_50",
    "941205":    "Derwent_21",
}

# ── Date range — everything from start of 2024 ───────────
FROM_DATE = "01-01-24"
TO_DATE   = "31-12-25"

# ─────────────────────────────────────────────────────────────

def download_one_sensor(sensor_id, sensor_name):
    """
    Downloads all readings for one sensor.
    Returns a pandas dataframe.
    """
    print(f"Downloading {sensor_name}...")

    # Build the URL
    url = (
        f"{API_URL}"
        f"?op=readings"
        f"&username={USERNAME}"
        f"&token={TOKEN}"
        f"&sensorid={sensor_id}"
        f"&fromdate={FROM_DATE}"
        f"&todate={TO_DATE}"
    )

    # Make the request
    response = requests.get(url, timeout=60)

    # Check it worked
    if response.status_code != 200:
        print(f"  ERROR: Could not connect. Status = {response.status_code}")
        return None

    if len(response.text.strip()) == 0:
        print(f"  No data returned for {sensor_name}")
        return None

    # Turn the text response into a dataframe
    df = pd.read_csv(StringIO(response.text))

    # Add columns to identify which sensor this is
    df['sensor_id']   = sensor_id
    df['sensor_name'] = sensor_name

    print(f"  Got {len(df)} rows")
    return df


def save_data(df, filepath):
    """
    Saves the combined dataframe to a CSV file.
    """
    # Make sure the folder exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"\nSaved to: {filepath}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())


def main():
    print("=" * 50)
    print("AquaSensor Data Download")
    print("=" * 50)

    # Step 1 — Download each sensor one by one
    all_data = []

    for sensor_id, sensor_name in SENSORS.items():
        df = download_one_sensor(sensor_id, sensor_name)
        if df is not None:
            all_data.append(df)

    # Step 2 — Check if got something
    if len(all_data) == 0:
        print("ERROR: No data downloaded at all. Check your internet connection.")
        return

    # Step 3 — Combine all sensors into one big dataframe
    combined = pd.concat(all_data, ignore_index=True)

    # Step 4 — Clean up column names
    combined.columns = combined.columns.str.strip().str.lower()
    combined = combined.rename(columns={
        'mg/l':    'dissolved_oxygen_mgl',
        'percent': 'dissolved_oxygen_pct',
    })

    # Step 5 — Create a proper timestamp column
    combined['timestamp'] = pd.to_datetime(
        combined['date'] + ' ' + combined['time'],
        dayfirst=True
    )

    # Step 6 — Sort by sensor and time
    combined = combined.sort_values(
        ['sensor_id', 'timestamp']
    ).reset_index(drop=True)

    # Step 7 — Save everything to one CSV file
    save_data(combined, "data/raw/aquasensor_all_sensors.csv")

    # Step 8 — Also save each sensor separately
    for sensor_id, sensor_name in SENSORS.items():
        one_sensor = combined[combined['sensor_id'] == sensor_id]
        if len(one_sensor) > 0:
            path = f"data/raw/aquasensor_{sensor_name}.csv"
            one_sensor.to_csv(path, index=False)
            print(f"Saved {sensor_name}: {len(one_sensor)} rows")

    print("\nDone! Your data is in the data/raw/ folder.")


# Run the script
main()