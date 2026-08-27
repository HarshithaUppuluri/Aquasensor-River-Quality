from datetime import timedelta

import pandas as pd

from utils.anomaly import assess_live_anomaly
from utils.aquasensor_api import fetch_aquasensor_live
from utils.config import (
    HORIZONS,
    HORIZON_MINUTES,
    LIVE_FORECAST_FILE,
)
from utils.database import save_forecasts
from utils.model_runtime import predict_sensor


def refresh_live_system(
    sensors=None,
    save_to_database=True,
):
    """
    Refresh the AquaPulse live prediction system.

    Parameters
    ----------
    sensors:
        Optional dataframe containing the latest AquaSensor
        reading for each station.

        If sensors is not supplied, the AquaSensor API is
        called automatically.

    save_to_database:
        When True, generated forecasts are stored in
        PostgreSQL.

    Returns
    -------
    sensors:
        Latest actual reading for every monitoring station.

    current_forecasts:
        Current 15-120 minute DO forecasts for every station.
    """

    # ========================================================
    # FETCH SENSOR DATA ONLY WHEN NECESSARY
    # ========================================================

    if sensors is None:

        sensors = fetch_aquasensor_live()

    if sensors is None or sensors.empty:

        raise RuntimeError(
            "No AquaSensor readings are available."
        )

    # Work on a copy so the original dataframe is not changed.
    sensors = sensors.copy()

    # ========================================================
    # NORMALISE SENSOR TIMESTAMPS
    # ========================================================

    sensors["timestamp"] = pd.to_datetime(
        sensors["timestamp"],
        errors="coerce",
    )

    sensors = sensors.dropna(
        subset=[
            "timestamp",
            "sensor_id",
            "sensor_name",
            "dissolved_oxygen_mgl",
            "temperature",
        ]
    )

    if sensors.empty:

        raise RuntimeError(
            "No valid AquaSensor readings are available."
        )

    # ========================================================
    # GENERATE FORECASTS
    # ========================================================

    rows = []

    for _, sensor in sensors.iterrows():

        # ----------------------------------------------------
        # ML PREDICTIONS
        # ----------------------------------------------------

        predictions, model_meta = predict_sensor(
            sensor
        )

        # ----------------------------------------------------
        # ANOMALY ASSESSMENT
        # ----------------------------------------------------

        anomaly = assess_live_anomaly(
            str(
                sensor["sensor_name"]
            ),
            float(
                sensor[
                    "dissolved_oxygen_mgl"
                ]
            ),
            float(
                sensor[
                    "temperature"
                ]
            ),
            predictions,
        )

        # ----------------------------------------------------
        # TIMES
        # ----------------------------------------------------

        prediction_run_time = (
            pd.Timestamp.now(
                tz="Europe/London"
            )
        )

        sensor_timestamp = pd.to_datetime(
            sensor["timestamp"],
            errors="coerce",
        )

        if pd.isna(sensor_timestamp):
            continue

        # ----------------------------------------------------
        # BASE FORECAST ROW
        # ----------------------------------------------------

        row = {
            "prediction_run_time":
                prediction_run_time,

            "latest_sensor_timestamp":
                sensor_timestamp,

            "sensor_id":
                str(
                    sensor[
                        "sensor_id"
                    ]
                ),

            "sensor_name":
                str(
                    sensor[
                        "sensor_name"
                    ]
                ),

            "temperature":
                pd.to_numeric(
                    sensor[
                        "temperature"
                    ],
                    errors="coerce",
                ),

            "current_do_mgl":
                pd.to_numeric(
                    sensor[
                        "dissolved_oxygen_mgl"
                    ],
                    errors="coerce",
                ),

            "current_do_pct":
                pd.to_numeric(
                    sensor.get(
                        "dissolved_oxygen_pct"
                    ),
                    errors="coerce",
                ),

            "anomaly_level":
                anomaly[
                    "level"
                ],

            "anomaly_reason":
                anomaly[
                    "reason"
                ],

            "do_change":
                anomaly[
                    "do_change"
                ],

            "forecast_min_do":
                anomaly[
                    "forecast_min"
                ],
        }

        # ====================================================
        # ALL 8 FORECAST HORIZONS
        # ====================================================

        for horizon in HORIZONS:

            minutes = HORIZON_MINUTES[
                horizon
            ]

            prediction_value = (
                predictions[
                    horizon
                ]
            )

            row[
                f"predicted_do_mgl_{horizon}"
            ] = float(
                prediction_value
            )

            row[
                f"forecast_time_{horizon}"
            ] = (
                sensor_timestamp
                + timedelta(
                    minutes=minutes
                )
            )

            row[
                f"model_{horizon}"
            ] = str(
                model_meta[
                    horizon
                ]["model"]
            )

        rows.append(
            row
        )

    # ========================================================
    # CURRENT FORECAST DATAFRAME
    # ========================================================

    current_forecasts = pd.DataFrame(
        rows
    )

    if current_forecasts.empty:

        raise RuntimeError(
            "No live forecasts could be generated."
        )

    # ========================================================
    # POSTGRESQL FORECAST STORAGE
    # ========================================================

    if save_to_database:

        save_forecasts(
            current_forecasts
        )

    # ========================================================
    # CSV FALLBACK / LOCAL HISTORY
    # ========================================================
    #
    # PostgreSQL is now the production persistent store.
    # This CSV remains useful locally and as a fallback.
    # ========================================================

    LIVE_FORECAST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if LIVE_FORECAST_FILE.exists():

        try:

            previous = pd.read_csv(
                LIVE_FORECAST_FILE,
                low_memory=False,
            )

            combined = pd.concat(
                [
                    previous,
                    current_forecasts,
                ],
                ignore_index=True,
            )

        except Exception:

            combined = (
                current_forecasts.copy()
            )

    else:

        combined = (
            current_forecasts.copy()
        )

    # ========================================================
    # NORMALISE CSV DATETIMES
    # ========================================================

    if (
        "latest_sensor_timestamp"
        in combined.columns
    ):

        combined[
            "latest_sensor_timestamp"
        ] = pd.to_datetime(
            combined[
                "latest_sensor_timestamp"
            ],
            errors="coerce",
        )

    if (
        "prediction_run_time"
        in combined.columns
    ):

        combined[
            "prediction_run_time"
        ] = pd.to_datetime(
            combined[
                "prediction_run_time"
            ],
            errors="coerce",
        )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    combined = (
        combined
        .drop_duplicates(
            subset=[
                "sensor_id",
                "latest_sensor_timestamp",
            ],
            keep="last",
        )
        .sort_values(
            [
                "sensor_name",
                "latest_sensor_timestamp",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SAVE LOCAL CSV FALLBACK
    # ========================================================

    combined.to_csv(
        LIVE_FORECAST_FILE,
        index=False,
    )

    return (
        sensors,
        current_forecasts,
    )