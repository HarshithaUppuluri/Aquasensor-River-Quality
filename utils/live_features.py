import numpy as np
import pandas as pd

from utils.aquasensor_api import load_live_history
from utils.config import MODEL_FEATURE_FALLBACK, PROCESSED_FEATURE_FILE


def expected_model_features(model):
    """
    Return the exact feature names expected by the saved sklearn pipeline.
    """
    names = getattr(model, "feature_names_in_", None)

    if names is not None:
        return [str(x) for x in names]

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            names = getattr(step, "feature_names_in_", None)

            if names is not None:
                return [str(x) for x in names]

    return MODEL_FEATURE_FALLBACK.copy()


def _template(sensor_id, sensor_name):
    """
    Load the latest historical processed row for the same sensor.

    This is used only as a compatibility fallback if the saved model expects
    a feature that is not being created from the live API.
    """
    if not PROCESSED_FEATURE_FILE.exists():
        return None

    try:
        df = pd.read_csv(
            PROCESSED_FEATURE_FILE,
            low_memory=False,
        )
    except Exception:
        return None

    if "sensor_id" in df.columns:
        by_id = df[
            df["sensor_id"].astype(str) == str(sensor_id)
        ]

        if not by_id.empty:
            return by_id.iloc[-1]

    if "sensor_name" in df.columns:
        by_name = df[
            df["sensor_name"].astype(str) == str(sensor_name)
        ]

        if not by_name.empty:
            return by_name.iloc[-1]

    return None


def _previous_do(sensor_name, ts):
    """
    Return the previous live DO measurement for anomaly feature engineering.
    """
    history = load_live_history(sensor_name)

    if history.empty:
        return None

    history = history.copy()

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="coerce",
    )

    history = history.dropna(
        subset=["timestamp"]
    )

    if history.empty:
        return None

    current_ts = pd.to_datetime(ts)

    # Make timezone handling consistent
    if current_ts.tzinfo is not None:
        try:
            history["timestamp"] = history[
                "timestamp"
            ].dt.tz_convert(current_ts.tzinfo)
        except TypeError:
            history["timestamp"] = history[
                "timestamp"
            ].dt.tz_localize(current_ts.tzinfo)

    history = history[
        history["timestamp"] < current_ts
    ]

    if history.empty:
        return None

    value = pd.to_numeric(
        history.iloc[-1]["dissolved_oxygen_mgl"],
        errors="coerce",
    )

    if pd.isna(value):
        return None

    return float(value)


def anomaly_feature(
    sensor_name,
    ts,
    do,
    temp,
):
    """
    Recreate the numeric anomaly_type feature expected by the trained model.

    0 = normal
    1 = low dissolved oxygen
    2 = sudden DO decrease
    3 = physically implausible/sensor-range reading
    """

    if (
        not np.isfinite(do)
        or not np.isfinite(temp)
        or do < 0
        or do > 25
        or temp < -5
        or temp > 40
    ):
        return 3

    if do < 4:
        return 1

    previous_do = _previous_do(
        sensor_name,
        ts,
    )

    if previous_do is not None:
        do_change = do - previous_do

        if do_change <= -1:
            return 2

    return 0


def create_season_proxy(
    timestamp,
    cloud_cover_pct,
):
    """
    Recreate season_proxy EXACTLY as it was created in preprocess.py.

    Original formula:

        angle = 2*pi*(day_of_year / 365.25)

        seasonal_signal =
            (cos(angle - pi) + 1) / 2

        clear_sky_signal =
            1 - cloud_cover_pct / 100

        season_proxy =
            0.70 * seasonal_signal
            + 0.30 * clear_sky_signal

    The trained model expects season_proxy as a numeric feature.
    """

    timestamp = pd.to_datetime(
        timestamp
    )

    day_of_year = int(
        timestamp.dayofyear
    )

    cloud_cover_pct = pd.to_numeric(
        cloud_cover_pct,
        errors="coerce",
    )

    # Same behaviour as preprocess.py
    if pd.isna(cloud_cover_pct):
        cloud_cover_pct = 50.0

    cloud_cover_pct = float(
        cloud_cover_pct
    )

    angle = (
        2
        * np.pi
        * (
            day_of_year
            / 365.25
        )
    )

    seasonal_signal = (
        np.cos(
            angle - np.pi
        )
        + 1
    ) / 2

    clear_sky_signal = (
        1
        - (
            cloud_cover_pct
            / 100
        )
    )

    season_proxy = (
        0.70
        * seasonal_signal
        + 0.30
        * clear_sky_signal
    )

    return round(
        float(season_proxy),
        4,
    )


def build_live_feature_row(
    sensor_row,
    weather,
    model,
):
    """
    Convert one live AquaSensor API reading plus live Open-Meteo data into
    the exact feature schema expected by the trained sklearn pipeline.
    """

    timestamp = pd.to_datetime(
        sensor_row["timestamp"]
    )

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "Europe/London"
        )
    else:
        timestamp = timestamp.tz_convert(
            "Europe/London"
        )

    sensor_id = str(
        sensor_row["sensor_id"]
    )

    sensor_name = str(
        sensor_row["sensor_name"]
    )

    dissolved_oxygen_mgl = float(
        sensor_row[
            "dissolved_oxygen_mgl"
        ]
    )

    dissolved_oxygen_pct = float(
        sensor_row[
            "dissolved_oxygen_pct"
        ]
    )

    water_temperature = float(
        sensor_row[
            "temperature"
        ]
    )

    air_temperature = pd.to_numeric(
        weather.get(
            "air_temperature_c"
        ),
        errors="coerce",
    )

    sunshine = pd.to_numeric(
        weather.get(
            "sunshine_wm2"
        ),
        errors="coerce",
    )

    cloud_cover = pd.to_numeric(
        weather.get(
            "cloud_cover_pct"
        ),
        errors="coerce",
    )

    if pd.isna(air_temperature):
        air_temperature = 0.0

    if pd.isna(sunshine):
        sunshine = 0.0

    if pd.isna(cloud_cover):
        cloud_cover = 50.0

    # ---------------------------------------------------------
    # Exact model feature names
    # ---------------------------------------------------------

    feature_names = expected_model_features(
        model
    )

    # Historical row used only as fallback
    template = _template(
        sensor_id,
        sensor_name,
    )

    values = {}

    for feature in feature_names:

        if (
            template is not None
            and feature in template.index
        ):
            values[feature] = template[
                feature
            ]

        else:
            values[feature] = 0.0

    # ---------------------------------------------------------
    # Live feature engineering
    # ---------------------------------------------------------

    live_values = {
        "temperature":
            water_temperature,

        "air_temperature_c":
            float(
                air_temperature
            ),

        "sunshine_wm2":
            float(
                sunshine
            ),

        "hour":
            int(
                timestamp.hour
            ),

        "dissolved_oxygen_mgl":
            dissolved_oxygen_mgl,

        "dissolved_oxygen_pct":
            dissolved_oxygen_pct,

        "pollution_alert":
            int(
                dissolved_oxygen_mgl
                < 4
            ),

        "anomaly_type":
            anomaly_feature(
                sensor_name,
                timestamp,
                dissolved_oxygen_mgl,
                water_temperature,
            ),

        # IMPORTANT:
        # Numeric feature — NOT string
        "season_proxy":
            create_season_proxy(
                timestamp,
                cloud_cover,
            ),

        "sensor_id":
            sensor_id,

        "sensor_name":
            sensor_name,
    }

    # Only insert fields expected by the saved model
    for key, value in live_values.items():

        if key in values:
            values[key] = value

    # ---------------------------------------------------------
    # Ensure correct data types
    # ---------------------------------------------------------

    numeric_features = [
        "temperature",
        "air_temperature_c",
        "sunshine_wm2",
        "hour",
        "dissolved_oxygen_mgl",
        "dissolved_oxygen_pct",
        "pollution_alert",
        "anomaly_type",
        "season_proxy",
    ]

    categorical_features = [
        "sensor_id",
        "sensor_name",
    ]

    for column in numeric_features:

        if column in values:
            values[column] = pd.to_numeric(
                values[column],
                errors="coerce",
            )

    for column in categorical_features:

        if column in values:
            values[column] = str(
                values[column]
            )

    # Replace any accidental missing numeric values
    for column in numeric_features:

        if (
            column in values
            and pd.isna(
                values[column]
            )
        ):
            values[column] = 0.0

    live_df = pd.DataFrame(
        [values],
        columns=feature_names,
    )

    return live_df