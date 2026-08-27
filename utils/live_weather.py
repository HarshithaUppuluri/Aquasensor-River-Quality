import pandas as pd
import requests


# ============================================================
# LIVE WEATHER
# ============================================================

def fetch_live_weather(
    latitude,
    longitude,
):
    """
    Fetch current weather conditions from Open-Meteo for
    the supplied AquaSensor latitude and longitude.

    The weather variables are used as environmental
    features for live dissolved oxygen prediction.
    """

    try:

        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "current":
                    (
                        "temperature_2m,"
                        "cloud_cover,"
                        "shortwave_radiation"
                    ),

                "timezone":
                    "Europe/London",
            },
            timeout=30,
        )

        response.raise_for_status()

        current = (
            response
            .json()
            .get(
                "current",
                {},
            )
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            "Live weather data could not be retrieved "
            "from Open-Meteo."
        ) from exc


    # ========================================================
    # NUMERIC VALUE HELPER
    # ========================================================

    def numeric_value(key):
        """
        Convert an Open-Meteo value to a float.

        Missing or invalid values are replaced with 0.0.
        """

        value = pd.to_numeric(
            current.get(key),
            errors="coerce",
        )

        if pd.notna(value):
            return float(value)

        return 0.0


    # ========================================================
    # RETURN WEATHER FEATURES
    # ========================================================

    return {
        "air_temperature_c":
            numeric_value(
                "temperature_2m"
            ),

        "cloud_cover_pct":
            numeric_value(
                "cloud_cover"
            ),

        "sunshine_wm2":
            numeric_value(
                "shortwave_radiation"
            ),

        "weather_timestamp":
            current.get(
                "time"
            ),
    }