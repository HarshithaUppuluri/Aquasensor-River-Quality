import requests
import pandas as pd
import os
from datetime import datetime

FROM = "2024-01-01"
TO   = datetime.today().strftime("%Y-%m-%d")


def main():
    print("=" * 50)
    print("Downloading Weather Data from Open-Meteo")
    print("=" * 50)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   53.33,
        "longitude":  -1.65,
        "start_date": FROM,
        "end_date":   TO,
        "hourly": [
            "temperature_2m",
            "shortwave_radiation",
            "cloudcover",
        ]
    }

    print("Connecting to Open-Meteo API...")
    try:
        r = requests.get(url, params=params, timeout=60)
        data = r.json()

        df = pd.DataFrame({
            'timestamp':       pd.to_datetime(
                                   data['hourly']['time']),
            'air_temperature_c': data['hourly']['temperature_2m'],
            'sunshine_wm2':      data['hourly'][
                                     'shortwave_radiation'],
            'cloud_cover_pct':   data['hourly']['cloudcover'],
        })

        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/weather_data.csv", index=False)

        print(f"Done!")
        print(f"Rows saved:  {len(df)}")
        print(f"File saved:  data/raw/weather_data.csv")
        print(df.head(5))

    except Exception as e:
        print(f"ERROR: {e}")
        print("Check your internet connection and try again.")


main()