"""
AquaPulse Scheduled Data Collector

This script is designed to be executed automatically by Render.

Each execution:
1. Initialises the PostgreSQL database tables.
2. Fetches AquaSensor data once.
3. Stores actual AquaSensor readings in PostgreSQL.
4. Generates DO forecasts for all three monitoring stations.
5. Stores the forecasts in PostgreSQL.
6. Exits cleanly.

The Render Cron Job will execute this script every 15 minutes.
"""

import sys
import traceback

from utils.aquasensor_api import fetch_aquasensor_live
from utils.database import initialise_database
from utils.live_service import refresh_live_system


# ============================================================
# COLLECTOR
# ============================================================

def main():
    """
    Run one complete AquaPulse collection and prediction cycle.
    """

    print("=" * 65)
    print("AquaPulse scheduled collector starting")
    print("=" * 65)

    # ========================================================
    # STEP 1 — INITIALISE POSTGRESQL
    # ========================================================

    print(
        "Initialising PostgreSQL database..."
    )

    initialise_database()

    print(
        "PostgreSQL database ready."
    )

    # ========================================================
    # STEP 2 — FETCH AQUASENSOR DATA
    # ========================================================
    #
    # fetch_aquasensor_live() already:
    #
    # - contacts the AquaSensor API
    # - processes all returned readings
    # - saves ALL valid readings into PostgreSQL
    # - keeps the CSV fallback
    # - returns the newest reading for each station
    #
    # Therefore we do NOT call the API again later.
    # ========================================================

    print(
        "Fetching AquaSensor readings..."
    )

    sensors = fetch_aquasensor_live()

    if sensors is None or sensors.empty:

        raise RuntimeError(
            "AquaSensor returned no usable sensor readings."
        )

    print(
        f"Latest station readings received: {len(sensors)}"
    )

    # ========================================================
    # DISPLAY SENSOR INFORMATION IN RENDER LOG
    # ========================================================

    for _, sensor in sensors.iterrows():

        sensor_name = str(
            sensor.get(
                "sensor_name",
                "Unknown"
            )
        )

        timestamp = sensor.get(
            "timestamp"
        )

        do_mgl = sensor.get(
            "dissolved_oxygen_mgl"
        )

        temperature = sensor.get(
            "temperature"
        )

        print(
            f"Station: {sensor_name} | "
            f"Timestamp: {timestamp} | "
            f"DO: {do_mgl} mg/L | "
            f"Temperature: {temperature} C"
        )

    # ========================================================
    # STEP 3 — GENERATE FORECASTS
    # ========================================================
    #
    # We pass the sensors dataframe directly.
    #
    # This is important because refresh_live_system() will NOT
    # make another AquaSensor API request when sensors are
    # supplied.
    # ========================================================

    print(
        "Generating DO forecasts..."
    )

    _, forecasts = refresh_live_system(
        sensors=sensors,
        save_to_database=True,
    )

    if (
        forecasts is None
        or forecasts.empty
    ):

        raise RuntimeError(
            "No DO forecasts were generated."
        )

    # ========================================================
    # STEP 4 — CONFIRM RESULTS
    # ========================================================

    print(
        f"Forecast rows generated and stored: "
        f"{len(forecasts)}"
    )

    for _, forecast in forecasts.iterrows():

        station = str(
            forecast.get(
                "sensor_name",
                "Unknown"
            )
        )

        current_do = forecast.get(
            "current_do_mgl"
        )

        prediction_15 = forecast.get(
            "predicted_do_mgl_15min"
        )

        prediction_60 = forecast.get(
            "predicted_do_mgl_60min"
        )

        prediction_120 = forecast.get(
            "predicted_do_mgl_120min"
        )

        print(
            f"{station} | "
            f"Current DO: {current_do} | "
            f"+15m: {prediction_15} | "
            f"+60m: {prediction_60} | "
            f"+120m: {prediction_120}"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("=" * 65)
    print(
        "AquaPulse collection completed successfully."
    )
    print("=" * 65)

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        exit_code = main()

        sys.exit(
            exit_code
        )

    except Exception as exc:

        print("=" * 65)
        print("AquaPulse collector FAILED")
        print("=" * 65)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        sys.exit(1)