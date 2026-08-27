from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
)

import threading
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.live_service import refresh_live_system
from utils.config import HORIZONS, HORIZON_LABELS
from utils.database import load_sensor_history, load_forecast_history


app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

LIVE_HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_aquasensor_api_readings.csv"
)

LIVE_FORECAST_HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_river_do_forecasts.csv"
)


# ============================================================
# SETTINGS
# ============================================================

LIVE_CACHE_SECONDS = 300
FALLBACK_CACHE_SECONDS = 20

PREFERRED_STATIONS = [
    "Derwent 13",
    "Derwent 13-50",
    "Derwent 21",
]


_live_cache = {
    "time": 0.0,
    "sensors": None,
    "forecasts": None,
}

_live_cache_lock = threading.Lock()


# ============================================================
# HELPERS
# ============================================================

def station_location(station_name):

    locations = {
        "Derwent 13": "Downstream",
        "Derwent 13-50": "Midstream",
        "Derwent 21": "Upstream",
    }

    return locations.get(
        station_name,
        "Monitoring station",
    )


# ============================================================
# LIVE SYSTEM
# ============================================================

def get_live_system(force=False):

    now = time.time()

    with _live_cache_lock:

        sensors = _live_cache["sensors"]
        forecasts = _live_cache["forecasts"]

        if (
            not force
            and sensors is not None
            and forecasts is not None
        ):

            source = sensors.attrs.get(
                "data_source",
                "live_api",
            )

            max_age = (
                LIVE_CACHE_SECONDS
                if source == "live_api"
                else FALLBACK_CACHE_SECONDS
            )

            if (
                now - _live_cache["time"]
                < max_age
            ):

                return sensors, forecasts

    sensors, forecasts = refresh_live_system(
        save_to_database=False,
    )

    if sensors is None or sensors.empty:
        raise RuntimeError(
            "No AquaSensor readings are available."
        )

    if forecasts is None or forecasts.empty:
        raise RuntimeError(
            "No dissolved oxygen forecasts are available."
        )

    with _live_cache_lock:

        _live_cache["time"] = time.time()
        _live_cache["sensors"] = sensors
        _live_cache["forecasts"] = forecasts

    return sensors, forecasts


# ============================================================
# COMMON LIVE PAGE DATA
# ============================================================

def prepare_page_data(
    sensors,
    forecasts,
    selected_station=None,
):

    available_names = (
        sensors["sensor_name"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    station_names = [
        station
        for station in PREFERRED_STATIONS
        if station in available_names
    ]

    for station in available_names:

        if station not in station_names:
            station_names.append(station)

    if not station_names:
        raise RuntimeError(
            "No monitoring stations were found."
        )

    if selected_station not in station_names:
        selected_station = station_names[0]

    # --------------------------------------------------------
    # CURRENT SENSOR
    # --------------------------------------------------------

    sensor_rows = sensors[
        sensors["sensor_name"].astype(str)
        == selected_station
    ]

    if sensor_rows.empty:
        raise RuntimeError(
            f"No reading found for {selected_station}."
        )

    sensor = sensor_rows.iloc[-1]

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast_rows_for_station = forecasts[
        forecasts["sensor_name"].astype(str)
        == selected_station
    ]

    if forecast_rows_for_station.empty:
        raise RuntimeError(
            f"No forecast found for {selected_station}."
        )

    forecast = forecast_rows_for_station.iloc[-1]

    # --------------------------------------------------------
    # STATION CARDS
    # --------------------------------------------------------

    stations = []

    for station_name in station_names:

        rows = sensors[
            sensors["sensor_name"].astype(str)
            == station_name
        ]

        if rows.empty:
            continue

        row = rows.iloc[-1]

        stations.append(
            {
                "sensor_name": station_name,
                "location": station_location(
                    station_name
                ),
                "do_mgl": float(
                    row["dissolved_oxygen_mgl"]
                ),
                "do_pct": float(
                    row["dissolved_oxygen_pct"]
                ),
                "temperature": float(
                    row["temperature"]
                ),
            }
        )

    # --------------------------------------------------------
    # FORECAST CARDS
    # --------------------------------------------------------

    forecast_rows = []

    for horizon in HORIZONS:

        column = (
            f"predicted_do_mgl_{horizon}"
        )

        if column not in forecast.index:
            continue

        value = pd.to_numeric(
            forecast[column],
            errors="coerce",
        )

        if pd.isna(value):
            continue

        forecast_rows.append(
            {
                "horizon": horizon,
                "label": HORIZON_LABELS.get(
                    horizon,
                    horizon,
                ),
                "value": float(value),
            }
        )

    timestamp = pd.to_datetime(
        sensor["timestamp"],
        errors="coerce",
    )

    if pd.isna(timestamp):
        timestamp_text = str(
            sensor["timestamp"]
        )
    else:
        timestamp_text = timestamp.strftime(
            "%d %b %Y, %H:%M"
        )

    data_source = sensors.attrs.get(
        "data_source",
        "live_api",
    )

    return {
        "stations": stations,
        "station_names": station_names,
        "selected_station": selected_station,
        "sensor": sensor,
        "forecast": forecast,
        "forecast_rows": forecast_rows,
        "timestamp_text": timestamp_text,
        "data_source": data_source,
    }


# ============================================================
# HISTORICAL DATA
# ============================================================

def _clean_sensor_history(history):
    """Normalise actual sensor history from PostgreSQL or CSV."""

    if history is None or history.empty:
        return pd.DataFrame()

    history = history.copy()

    required_columns = [
        "timestamp",
        "sensor_name",
        "dissolved_oxygen_mgl",
        "dissolved_oxygen_pct",
        "temperature",
    ]

    if not all(
        column in history.columns
        for column in required_columns
    ):
        return pd.DataFrame()

    history["sensor_name"] = (
        history["sensor_name"].astype(str)
    )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="coerce",
    )

    for column in (
        "dissolved_oxygen_mgl",
        "dissolved_oxygen_pct",
        "temperature",
    ):
        history[column] = pd.to_numeric(
            history[column],
            errors="coerce",
        )

    return (
        history
        .dropna(
            subset=[
                "timestamp",
                "sensor_name",
                "dissolved_oxygen_mgl",
                "dissolved_oxygen_pct",
                "temperature",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=[
                "timestamp",
                "sensor_name",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )


def load_history():
    """
    Load all accumulated actual AquaSensor readings.

    Production priority:
    1. PostgreSQL persistent history.
    2. CSV fallback for local use or temporary DB failure.
    """

    try:
        history = load_sensor_history()
        history = _clean_sensor_history(history)

        if not history.empty:
            return history

    except Exception as exc:
        print(
            "Warning: PostgreSQL sensor history unavailable: "
            f"{exc}"
        )

    if not LIVE_HISTORY_FILE.exists():
        return pd.DataFrame()

    try:
        history = pd.read_csv(
            LIVE_HISTORY_FILE,
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()

    return _clean_sensor_history(history)


def load_station_history(
    station_name,
    max_rows=5000,
):
    """Return actual DO and temperature history for one station."""

    # Query PostgreSQL directly first so long-running production
    # history does not require loading every station into memory.
    try:
        history = load_sensor_history(
            sensor_name=station_name,
        )
        history = _clean_sensor_history(history)

        if not history.empty:
            return (
                history
                .sort_values("timestamp")
                .tail(max_rows)
                .reset_index(drop=True)
            )

    except Exception as exc:
        print(
            f"Warning: PostgreSQL history unavailable for "
            f"{station_name}: {exc}"
        )

    history = load_history()

    if history.empty:
        return history

    history = history[
        history["sensor_name"].astype(str)
        == str(station_name)
    ].copy()

    return (
        history
        .sort_values("timestamp")
        .tail(max_rows)
        .reset_index(drop=True)
    )


def load_station_forecast_history(
    station_name,
    max_rows=5000,
):
    """
    Load accumulated historical 15-minute DO predictions.

    PostgreSQL is the production source. The existing CSV is
    retained as a fallback for local development.
    """

    history = pd.DataFrame()

    try:
        history = load_forecast_history(
            sensor_name=station_name,
        )
    except Exception as exc:
        print(
            f"Warning: PostgreSQL forecast history unavailable "
            f"for {station_name}: {exc}"
        )

    if history is None or history.empty:

        if not LIVE_FORECAST_HISTORY_FILE.exists():
            return pd.DataFrame()

        try:
            history = pd.read_csv(
                LIVE_FORECAST_HISTORY_FILE,
                low_memory=False,
            )
        except Exception:
            return pd.DataFrame()

        if "sensor_name" in history.columns:
            history = history[
                history["sensor_name"].astype(str)
                == str(station_name)
            ].copy()

    required_columns = [
        "sensor_name",
        "forecast_time_15min",
        "predicted_do_mgl_15min",
    ]

    if not all(
        column in history.columns
        for column in required_columns
    ):
        return pd.DataFrame()

    history["forecast_time_15min"] = pd.to_datetime(
        history["forecast_time_15min"],
        errors="coerce",
    )

    history["predicted_do_mgl_15min"] = pd.to_numeric(
        history["predicted_do_mgl_15min"],
        errors="coerce",
    )

    return (
        history
        .dropna(
            subset=[
                "forecast_time_15min",
                "predicted_do_mgl_15min",
            ]
        )
        .sort_values("forecast_time_15min")
        .drop_duplicates(
            subset=["forecast_time_15min"],
            keep="last",
        )
        .tail(max_rows)
        .reset_index(drop=True)
    )


# ============================================================
# INSIGHTS GRAPH
# ============================================================

def create_station_plot(
    station_name,
    sensors,
    forecasts,
):
    """
    Create one live Insights graph for a monitoring station.

    The graph always contains three time-series running together:
    - Actual DO: solid dark blue
    - Water Temperature: solid orange, right y-axis
    - Predicted DO: dotted blue

    The same function is called for Derwent 13, Derwent 13-50
    and Derwent 21.
    """

    # --------------------------------------------------------
    # CURRENT SENSOR READING
    # --------------------------------------------------------

    sensor_rows = sensors[
        sensors["sensor_name"].astype(str)
        == str(station_name)
    ]

    if sensor_rows.empty:
        return None

    sensor = sensor_rows.iloc[-1]

    current_time = pd.to_datetime(
        sensor["timestamp"],
        errors="coerce",
    )

    current_do = pd.to_numeric(
        sensor.get("dissolved_oxygen_mgl"),
        errors="coerce",
    )

    current_temperature = pd.to_numeric(
        sensor.get("temperature"),
        errors="coerce",
    )

    current_do_pct = pd.to_numeric(
        sensor.get("dissolved_oxygen_pct"),
        errors="coerce",
    )

    if (
        pd.isna(current_time)
        or pd.isna(current_do)
        or pd.isna(current_temperature)
    ):
        return None

    current_do = float(current_do)
    current_temperature = float(current_temperature)

    # --------------------------------------------------------
    # ACTUAL DO + TEMPERATURE HISTORY
    # --------------------------------------------------------

    history = load_station_history(
        station_name,
        max_rows=5000,
    )

    latest_actual = pd.DataFrame(
        {
            "timestamp": [current_time],
            "sensor_name": [station_name],
            "dissolved_oxygen_mgl": [current_do],
            "dissolved_oxygen_pct": [current_do_pct],
            "temperature": [current_temperature],
        }
    )

    if history.empty:
        history = latest_actual
    else:
        history = pd.concat(
            [history, latest_actual],
            ignore_index=True,
        )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="coerce",
    )
    history["dissolved_oxygen_mgl"] = pd.to_numeric(
        history["dissolved_oxygen_mgl"],
        errors="coerce",
    )
    history["temperature"] = pd.to_numeric(
        history["temperature"],
        errors="coerce",
    )

    history = (
        history
        .dropna(
            subset=[
                "timestamp",
                "dissolved_oxygen_mgl",
                "temperature",
            ]
        )
        .drop_duplicates(
            subset=["timestamp", "sensor_name"],
            keep="last",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # HISTORICAL PREDICTED DO
    # --------------------------------------------------------

    forecast_history = load_station_forecast_history(
        station_name,
        max_rows=5000,
    )

    predicted_points = []

    if not forecast_history.empty:
        for _, row in forecast_history.iterrows():
            predicted_points.append(
                {
                    "timestamp": row["forecast_time_15min"],
                    "predicted_do": float(
                        row["predicted_do_mgl_15min"]
                    ),
                }
            )

    # --------------------------------------------------------
    # LATEST FUTURE 15-120 MINUTE FORECAST
    # --------------------------------------------------------

    station_forecasts = forecasts[
        forecasts["sensor_name"].astype(str)
        == str(station_name)
    ]

    if not station_forecasts.empty:
        forecast = station_forecasts.iloc[-1]

        # Connect the newest predicted line to the latest actual DO.
        predicted_points.append(
            {
                "timestamp": current_time,
                "predicted_do": current_do,
            }
        )

        for horizon in HORIZONS:
            prediction_column = f"predicted_do_mgl_{horizon}"

            if prediction_column not in forecast.index:
                continue

            value = pd.to_numeric(
                forecast[prediction_column],
                errors="coerce",
            )

            if pd.isna(value):
                continue

            minutes = int(
                str(horizon).replace("min", "")
            )

            predicted_points.append(
                {
                    "timestamp": (
                        current_time
                        + pd.Timedelta(minutes=minutes)
                    ),
                    "predicted_do": float(value),
                }
            )

    predicted_history = pd.DataFrame(predicted_points)

    if not predicted_history.empty:
        predicted_history["timestamp"] = pd.to_datetime(
            predicted_history["timestamp"],
            errors="coerce",
        )
        predicted_history["predicted_do"] = pd.to_numeric(
            predicted_history["predicted_do"],
            errors="coerce",
        )
        predicted_history = (
            predicted_history
            .dropna(
                subset=["timestamp", "predicted_do"]
            )
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    figure = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    # 1. ACTUAL DO - DARK BLUE SOLID
    figure.add_trace(
        go.Scatter(
            x=history["timestamp"],
            y=history["dissolved_oxygen_mgl"],
            name="Actual DO",
            mode="lines",
            connectgaps=False,
            line=dict(
                color="#3155A5",
                width=3,
            ),
            hovertemplate=(
                "<b>Actual DO</b>"
                "<br>%{x|%d %b %Y %H:%M}"
                "<br>%{y:.2f} mg/L"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # 2. WATER TEMPERATURE - ORANGE SOLID
    figure.add_trace(
        go.Scatter(
            x=history["timestamp"],
            y=history["temperature"],
            name="Water Temperature",
            mode="lines",
            connectgaps=False,
            line=dict(
                color="#F28C28",
                width=3,
            ),
            hovertemplate=(
                "<b>Water Temperature</b>"
                "<br>%{x|%d %b %Y %H:%M}"
                "<br>%{y:.2f} °C"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    # 3. PREDICTED DO - BLUE DOTTED
    if not predicted_history.empty:
        figure.add_trace(
            go.Scatter(
                x=predicted_history["timestamp"],
                y=predicted_history["predicted_do"],
                name="Predicted DO",
                mode="lines",
                connectgaps=False,
                line=dict(
                    color="#7698C8",
                    width=3,
                    dash="dot",
                ),
                hovertemplate=(
                    "<b>Predicted DO</b>"
                    "<br>%{x|%d %b %Y %H:%M}"
                    "<br>%{y:.2f} mg/L"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    figure.update_layout(
        title=dict(
            text=f"{station_name} — Live River Trend",
            x=0.02,
            font=dict(
                size=19,
                color="#334A56",
            ),
        ),
        height=540,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="x unified",
        dragmode="pan",
        margin=dict(
            l=70,
            r=75,
            t=90,
            b=75,
        ),
        legend=dict(
            orientation="h",
            y=1.03,
            x=0,
        ),
    )

    figure.update_xaxes(
        title_text="Date and time",
        showgrid=True,
        gridcolor="#E7EEF2",
        rangeslider=dict(
            visible=True,
            thickness=0.08,
        ),
        rangeselector=dict(
            buttons=[
                dict(
                    count=1,
                    label="1h",
                    step="hour",
                    stepmode="backward",
                ),
                dict(
                    count=6,
                    label="6h",
                    step="hour",
                    stepmode="backward",
                ),
                dict(
                    count=1,
                    label="24h",
                    step="day",
                    stepmode="backward",
                ),
                dict(
                    count=7,
                    label="7d",
                    step="day",
                    stepmode="backward",
                ),
                dict(
                    step="all",
                    label="All",
                ),
            ]
        ),
    )

    figure.update_yaxes(
        title_text="Dissolved Oxygen (mg/L)",
        secondary_y=False,
        showgrid=True,
        gridcolor="#E7EEF2",
    )

    figure.update_yaxes(
        title_text="Water Temperature (°C)",
        secondary_y=True,
        showgrid=False,
    )

    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d",
            ],
        },
    )


# ============================================================
# DATA RECORD FILTER
# ============================================================

def filter_history(
    history,
    station_filter,
    range_filter,
):

    if history.empty:
        return history

    filtered = history.copy()

    if (
        station_filter
        != "All Stations"
    ):

        filtered = filtered[
            filtered["sensor_name"]
            == station_filter
        ]

    if filtered.empty:
        return filtered

    latest_time = (
        filtered["timestamp"].max()
    )

    ranges = {
        "1h": pd.Timedelta(hours=1),
        "6h": pd.Timedelta(hours=6),
        "24h": pd.Timedelta(hours=24),
        "7d": pd.Timedelta(days=7),
    }

    if range_filter in ranges:

        start_time = (
            latest_time
            - ranges[range_filter]
        )

        filtered = filtered[
            filtered["timestamp"]
            >= start_time
        ]

    return filtered.sort_values(
        "timestamp"
    )


# ============================================================
# DATA RECORDS GRAPH
# ============================================================

def create_records_plot(
    filtered_history,
):

    if filtered_history.empty:
        return None

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Dissolved Oxygen (mg/L)",
            "Dissolved Oxygen Percentage (%)",
            "Water Temperature (°C)",
        ),
    )

    colours = {
        "Derwent 13": "#73AFC8",
        "Derwent 13-50": "#9887C9",
        "Derwent 21": "#DDA17A",
    }

    station_names = (
        filtered_history[
            "sensor_name"
        ]
        .drop_duplicates()
        .tolist()
    )

    for station_name in station_names:

        station_data = filtered_history[
            filtered_history[
                "sensor_name"
            ]
            == station_name
        ]

        colour = colours.get(
            station_name,
            "#79B29F",
        )

        # DO MG/L
        figure.add_trace(
            go.Scatter(
                x=station_data[
                    "timestamp"
                ],
                y=station_data[
                    "dissolved_oxygen_mgl"
                ],
                name=station_name,
                legendgroup=station_name,
                mode="lines+markers",
                line=dict(
                    color=colour,
                    width=2.5,
                ),
                marker=dict(
                    size=4
                ),
                hovertemplate=(
                    f"<b>{station_name}</b>"
                    "<br>%{x|%d %b %Y %H:%M}"
                    "<br>DO: %{y:.2f} mg/L"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # DO %
        figure.add_trace(
            go.Scatter(
                x=station_data[
                    "timestamp"
                ],
                y=station_data[
                    "dissolved_oxygen_pct"
                ],
                legendgroup=station_name,
                showlegend=False,
                mode="lines+markers",
                line=dict(
                    color=colour,
                    width=2.2,
                ),
                marker=dict(
                    size=3
                ),
                hovertemplate=(
                    f"<b>{station_name}</b>"
                    "<br>%{x|%d %b %Y %H:%M}"
                    "<br>DO: %{y:.2f}%"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        # TEMPERATURE
        figure.add_trace(
            go.Scatter(
                x=station_data[
                    "timestamp"
                ],
                y=station_data[
                    "temperature"
                ],
                legendgroup=station_name,
                showlegend=False,
                mode="lines+markers",
                line=dict(
                    color=colour,
                    width=2.2,
                ),
                marker=dict(
                    size=3
                ),
                hovertemplate=(
                    f"<b>{station_name}</b>"
                    "<br>%{x|%d %b %Y %H:%M}"
                    "<br>%{y:.2f} °C"
                    "<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )

    figure.update_layout(
        title=dict(
            text=(
                "Historical AquaSensor Measurements"
            ),
            x=0.02,
        ),
        height=800,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="x unified",
        dragmode="pan",
        margin=dict(
            l=70,
            r=35,
            t=100,
            b=90,
        ),
        legend=dict(
            orientation="h",
            y=1.03,
            x=0,
        ),
    )

    figure.update_xaxes(
        showgrid=True,
        gridcolor="#EDF1F2",
    )

    figure.update_yaxes(
        showgrid=True,
        gridcolor="#EDF1F2",
    )

    figure.update_xaxes(
        title_text="Date and time",
        rangeslider=dict(
            visible=True,
            thickness=0.08,
        ),
        row=3,
        col=1,
    )

    figure.update_yaxes(
        title_text="DO mg/L",
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="DO %",
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text="Temperature °C",
        row=3,
        col=1,
    )

    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
        },
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    selected_station = request.args.get(
        "station",
        "Derwent 13",
    )

    try:

        sensors, forecasts = get_live_system()

        data = prepare_page_data(
            sensors,
            forecasts,
            selected_station,
        )

        return render_template(
            "index.html",
            active_page="home",
            error=None,
            **data,
        )

    except Exception as exc:

        return render_template(
            "index.html",
            active_page="home",
            error=str(exc),
            stations=[],
            station_names=[],
            selected_station=None,
            sensor=None,
            forecast=None,
            forecast_rows=[],
            timestamp_text="Unavailable",
            data_source=None,
        )


# ============================================================
# INSIGHTS
# ============================================================

@app.route("/insights")
def insights():

    selected_station = request.args.get(
        "station",
        "Derwent 13",
    )

    try:

        sensors, forecasts = get_live_system()

        data = prepare_page_data(
            sensors,
            forecasts,
            selected_station,
        )

        station_statuses = []

        for station_name in data[
            "station_names"
        ]:

            sensor_rows = sensors[
                sensors[
                    "sensor_name"
                ].astype(str)
                == station_name
            ]

            forecast_rows = forecasts[
                forecasts[
                    "sensor_name"
                ].astype(str)
                == station_name
            ]

            if (
                sensor_rows.empty
                or forecast_rows.empty
            ):
                continue

            sensor_row = (
                sensor_rows.iloc[-1]
            )

            forecast_row = (
                forecast_rows.iloc[-1]
            )

            level = str(
                forecast_row.get(
                    "anomaly_level",
                    "Green",
                )
            ).title()

            reason = str(
                forecast_row.get(
                    "anomaly_reason",
                    "No warning detected.",
                )
            )

            station_statuses.append(
                {
                    "name": station_name,
                    "location": station_location(
                        station_name
                    ),
                    "do_mgl": float(
                        sensor_row[
                            "dissolved_oxygen_mgl"
                        ]
                    ),
                    "do_pct": float(
                        sensor_row[
                            "dissolved_oxygen_pct"
                        ]
                    ),
                    "temperature": float(
                        sensor_row[
                            "temperature"
                        ]
                    ),
                    "level": level,
                    "reason": reason,
                }
            )

        graphs = []

        for station_name in data[
            "station_names"
        ]:

            graphs.append(
                {
                    "station_name":
                        station_name,
                    "location":
                        station_location(
                            station_name
                        ),
                    "html":
                        create_station_plot(
                            station_name,
                            sensors,
                            forecasts,
                        ),
                }
            )

        return render_template(
            "insights.html",
            active_page="insights",
            station_statuses=station_statuses,
            graphs=graphs,
            error=None,
            **data,
        )

    except Exception as exc:

        return render_template(
            "insights.html",
            active_page="insights",
            error=str(exc),
            station_statuses=[],
            graphs=[],
            stations=[],
            station_names=[],
            selected_station=None,
            sensor=None,
            forecast=None,
            forecast_rows=[],
            timestamp_text="Unavailable",
            data_source=None,
        )


# ============================================================
# DATA RECORDS
# ============================================================

@app.route("/data-records")
def data_records():

    station_filter = request.args.get(
        "station",
        "All Stations",
    )

    range_filter = request.args.get(
        "range",
        "24h",
    )

    if station_filter not in (
        ["All Stations"]
        + PREFERRED_STATIONS
    ):
        station_filter = (
            "All Stations"
        )

    if range_filter not in [
        "1h",
        "6h",
        "24h",
        "7d",
        "all",
    ]:
        range_filter = "24h"

    history = load_history()

    filtered = filter_history(
        history,
        station_filter,
        range_filter,
    )

    graph_html = (
        create_records_plot(
            filtered
        )
        if not filtered.empty
        else None
    )

    total_records = len(
        filtered
    )

    station_count = (
        filtered[
            "sensor_name"
        ].nunique()
        if not filtered.empty
        else 0
    )

    if filtered.empty:

        earliest_record = "No records"
        latest_record = "No records"

    else:

        earliest_record = (
            filtered[
                "timestamp"
            ]
            .min()
            .strftime(
                "%d %b %Y, %H:%M"
            )
        )

        latest_record = (
            filtered[
                "timestamp"
            ]
            .max()
            .strftime(
                "%d %b %Y, %H:%M"
            )
        )

    records = []

    if not filtered.empty:

        table_data = (
            filtered
            .sort_values(
                "timestamp",
                ascending=False,
            )
            .head(1000)
        )

        for _, row in table_data.iterrows():

            records.append(
                {
                    "timestamp":
                        row["timestamp"]
                        .strftime(
                            "%d %b %Y, %H:%M:%S"
                        ),
                    "sensor_name":
                        row["sensor_name"],
                    "do_mgl":
                        float(
                            row[
                                "dissolved_oxygen_mgl"
                            ]
                        ),
                    "do_pct":
                        float(
                            row[
                                "dissolved_oxygen_pct"
                            ]
                        ),
                    "temperature":
                        float(
                            row[
                                "temperature"
                            ]
                        ),
                }
            )

    return render_template(
        "data_records.html",
        active_page="data_records",
        station_filter=station_filter,
        range_filter=range_filter,
        station_names=PREFERRED_STATIONS,
        records=records,
        graph_html=graph_html,
        total_records=total_records,
        earliest_record=earliest_record,
        latest_record=latest_record,
        station_count=station_count,
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html",
        active_page="about",
    )


# ============================================================
# REFRESH
# ============================================================

@app.route("/refresh")
def refresh():

    page = request.args.get(
        "page",
        "home",
    )

    station = request.args.get(
        "station",
        "Derwent 13",
    )

    try:
        get_live_system(
            force=True
        )
    except Exception:
        pass

    if page == "insights":

        return redirect(
            url_for(
                "insights",
                station=station,
            )
        )

    if page == "data_records":

        return redirect(
            url_for(
                "data_records"
            )
        )

    return redirect(
        url_for(
            "home",
            station=station,
        )
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
    )