import os
import tomllib
from pathlib import Path

import pandas as pd
import requests

from utils.config import (
    AQUASENSOR_BASE_URL,
    LIVE_API_HISTORY_FILE,
    SENSOR_NAME_MAP,
)


# ============================================================
# CUSTOM ERROR
# ============================================================

class AquaSensorAPIError(RuntimeError):
    pass


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# AQUASENSOR CREDENTIALS
# ============================================================

def _credentials():
    """
    Load AquaSensor credentials.

    Priority:
    1. Environment variables - suitable for deployment.
    2. config/secrets.toml - preferred Flask local setup.
    3. .streamlit/secrets.toml - temporary legacy fallback.

    This allows AquaPulse to run locally and after deployment
    without depending on Streamlit.
    """

    # --------------------------------------------------------
    # 1. DEPLOYMENT ENVIRONMENT VARIABLES
    # --------------------------------------------------------

    user = os.getenv(
        "AQUASENSOR_USER",
        "",
    ).strip()

    token = os.getenv(
        "AQUASENSOR_TOKEN",
        "",
    ).strip()

    catchment = os.getenv(
        "AQUASENSOR_CATCHMENT",
        "",
    ).strip()

    if user and token and catchment:
        return user, token, catchment

    # --------------------------------------------------------
    # 2. LOCAL SECRETS FILES
    # --------------------------------------------------------

    possible_files = [
        PROJECT_ROOT
        / "config"
        / "secrets.toml",

        PROJECT_ROOT
        / ".streamlit"
        / "secrets.toml",
    ]

    for secrets_file in possible_files:

        if not secrets_file.exists():
            continue

        try:

            with open(
                secrets_file,
                "rb",
            ) as file:

                secrets = tomllib.load(
                    file
                )

            user = str(
                secrets.get(
                    "AQUASENSOR_USER",
                    "",
                )
            ).strip()

            token = str(
                secrets.get(
                    "AQUASENSOR_TOKEN",
                    "",
                )
            ).strip()

            catchment = str(
                secrets.get(
                    "AQUASENSOR_CATCHMENT",
                    "",
                )
            ).strip()

            if (
                user
                and token
                and catchment
            ):
                return (
                    user,
                    token,
                    catchment,
                )

        except Exception as exc:

            raise AquaSensorAPIError(
                f"Could not read AquaSensor "
                f"credentials from {secrets_file}"
            ) from exc

    # --------------------------------------------------------
    # NOTHING FOUND
    # --------------------------------------------------------

    raise AquaSensorAPIError(
        "AquaSensor credentials are missing. "
        "Provide AQUASENSOR_USER, AQUASENSOR_TOKEN "
        "and AQUASENSOR_CATCHMENT as environment "
        "variables or store them in "
        "config/secrets.toml."
    )


# ============================================================
# NORMALISE API RESPONSE
# ============================================================

def _normalise_payload(payload):
    """
    Extract the list of sensors from the different response
    structures returned by the AquaSensor API.
    """

    if (
        isinstance(payload, list)
        and payload
        and isinstance(
            payload[0],
            dict,
        )
        and "sensors" in payload[0]
    ):
        return payload[0]["sensors"]

    if (
        isinstance(payload, dict)
        and "sensors" in payload
    ):
        return payload["sensors"]

    raise AquaSensorAPIError(
        "Unexpected AquaSensor API response structure."
    )


# ============================================================
# PARSE SENSOR
# ============================================================

def _parse_sensor(sensor):
    """
    Convert one AquaSensor API sensor response into
    tabular reading records.
    """

    sensor_id = str(
        sensor.get(
            "uid",
            "",
        )
    ).strip()

    description = str(
        sensor.get(
            "description",
            "",
        )
    ).strip()

    sensor_name = SENSOR_NAME_MAP.get(
        sensor_id,
        description or sensor_id,
    )

    latitude = pd.to_numeric(
        sensor.get("lat"),
        errors="coerce",
    )

    longitude = pd.to_numeric(
        sensor.get("lon"),
        errors="coerce",
    )

    location = str(
        sensor.get(
            "location",
            "",
        )
    ).strip()

    rows = []

    for reading in (
        sensor.get(
            "data",
            [],
        )
        or []
    ):

        ts = pd.to_numeric(
            reading.get("ts"),
            errors="coerce",
        )

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        if pd.notna(ts):

            timestamp = (
                pd.to_datetime(
                    int(ts),
                    unit="s",
                    utc=True,
                )
                .tz_convert(
                    "Europe/London"
                )
            )

        else:

            timestamp = pd.to_datetime(
                (
                    f"{reading.get('date', '')} "
                    f"{reading.get('time', '')}"
                ),
                errors="coerce",
                dayfirst=True,
            )

        # ----------------------------------------------------
        # READING
        # ----------------------------------------------------

        rows.append(
            {
                "timestamp":
                    timestamp,

                "sensor_id":
                    sensor_id,

                "sensor_name":
                    sensor_name,

                "location":
                    location,

                "lat":
                    (
                        float(latitude)
                        if pd.notna(latitude)
                        else None
                    ),

                "lon":
                    (
                        float(longitude)
                        if pd.notna(longitude)
                        else None
                    ),

                "temperature":
                    pd.to_numeric(
                        reading.get("tmp"),
                        errors="coerce",
                    ),

                "dissolved_oxygen_pct":
                    pd.to_numeric(
                        reading.get("pcent"),
                        errors="coerce",
                    ),

                "dissolved_oxygen_mgl":
                    pd.to_numeric(
                        reading.get("mgl"),
                        errors="coerce",
                    ),

                "count":
                    pd.to_numeric(
                        reading.get("cnt"),
                        errors="coerce",
                    ),
            }
        )

    return rows


# ============================================================
# SAVE LIVE HISTORY
# ============================================================

def append_live_history(df):
    """
    Append live AquaSensor readings to the historical CSV
    while preventing duplicate sensor/timestamp records.
    """

    if df is None or df.empty:
        return

    LIVE_API_HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    incoming = df.copy()

    incoming["timestamp"] = (
        incoming["timestamp"]
        .astype(str)
    )

    # --------------------------------------------------------
    # ADD EXISTING HISTORY
    # --------------------------------------------------------

    if LIVE_API_HISTORY_FILE.exists():

        try:

            old = pd.read_csv(
                LIVE_API_HISTORY_FILE,
                low_memory=False,
            )

            incoming = pd.concat(
                [
                    old,
                    incoming,
                ],
                ignore_index=True,
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    incoming = (
        incoming
        .drop_duplicates(
            subset=[
                "sensor_id",
                "timestamp",
            ],
            keep="last",
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    incoming.to_csv(
        LIVE_API_HISTORY_FILE,
        index=False,
    )


# ============================================================
# LOAD LIVE HISTORY
# ============================================================

def load_live_history(
    sensor_name=None,
):
    """
    Load stored AquaSensor live readings.

    A station name can optionally be supplied to return
    readings for only one monitoring station.
    """

    if not LIVE_API_HISTORY_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        LIVE_API_HISTORY_FILE,
        low_memory=False,
    )

    if df.empty:
        return df

    if "timestamp" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "timestamp",
        ]
    )

    if (
        sensor_name is not None
        and "sensor_name" in df.columns
    ):

        df = df[
            df["sensor_name"]
            .astype(str)
            == str(sensor_name)
        ]

    return (
        df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# LOAD FALLBACK READINGS
# ============================================================

def load_latest_saved_readings():
    """
    Return the newest successfully stored reading for each
    AquaSensor monitoring station.

    This is used if the live AquaSensor API is temporarily
    unavailable.
    """

    history = load_live_history()

    if history.empty:

        raise AquaSensorAPIError(
            "No saved AquaSensor readings are available."
        )

    latest = (
        history
        .sort_values(
            "timestamp"
        )
        .groupby(
            "sensor_name",
            as_index=False,
            dropna=False,
        )
        .tail(1)
        .reset_index(
            drop=True
        )
    )

    return latest


# ============================================================
# FETCH LIVE AQUASENSOR DATA
# ============================================================

def fetch_aquasensor_live():
    """
    Fetch the newest readings from the official AquaSensor
    API.

    If the API is unavailable, AquaPulse automatically uses
    the latest successfully stored readings instead.
    """

    user, token, catchment = (
        _credentials()
    )

    try:

        # ----------------------------------------------------
        # API REQUEST
        # ----------------------------------------------------

        response = requests.get(
            AQUASENSOR_BASE_URL,
            params={
                "user":
                    user,

                "token":
                    token,

                "catchment":
                    catchment,
            },
            headers={
                "Accept":
                    "application/json",
            },
            timeout=12,
        )

        response.raise_for_status()

        # ----------------------------------------------------
        # API JSON
        # ----------------------------------------------------

        payload = response.json()

        sensors = _normalise_payload(
            payload
        )

        # ----------------------------------------------------
        # PARSE READINGS
        # ----------------------------------------------------

        rows = []

        for sensor in sensors:

            rows.extend(
                _parse_sensor(
                    sensor
                )
            )

        df = pd.DataFrame(
            rows
        )

        if df.empty:

            raise AquaSensorAPIError(
                "AquaSensor returned no readings."
            )

        # ----------------------------------------------------
        # CLEAN
        # ----------------------------------------------------

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "timestamp",
                "dissolved_oxygen_mgl",
                "temperature",
            ]
        )

        # ----------------------------------------------------
        # LATEST READING PER STATION
        # ----------------------------------------------------

        df = (
            df
            .sort_values(
                [
                    "sensor_name",
                    "timestamp",
                ]
            )
            .groupby(
                "sensor_name",
                as_index=False,
                dropna=False,
            )
            .tail(1)
            .reset_index(
                drop=True
            )
        )

        if df.empty:

            raise AquaSensorAPIError(
                "No valid AquaSensor readings "
                "were returned."
            )

        # ----------------------------------------------------
        # SAVE SUCCESSFUL LIVE RESULT
        # ----------------------------------------------------

        append_live_history(
            df
        )

        # ----------------------------------------------------
        # DATA SOURCE INFORMATION
        # ----------------------------------------------------

        df.attrs[
            "data_source"
        ] = "live_api"

        df.attrs[
            "api_error"
        ] = None

        return df

    # ========================================================
    # FALLBACK
    # ========================================================

    except Exception as api_error:

        try:

            fallback = (
                load_latest_saved_readings()
            )

            fallback.attrs[
                "data_source"
            ] = "saved_fallback"

            fallback.attrs[
                "api_error"
            ] = str(
                api_error
            )

            return fallback

        except Exception as fallback_error:

            raise AquaSensorAPIError(
                "The AquaSensor API is unavailable "
                "and no saved live readings could be loaded. "
                f"API error: {api_error}. "
                f"Fallback error: {fallback_error}"
            ) from fallback_error