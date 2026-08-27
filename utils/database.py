import os

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


class DatabaseError(RuntimeError):
    pass


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_database_url():
    """
    Return the PostgreSQL connection URL supplied by Render.
    """

    database_url = os.getenv(
        "DATABASE_URL",
        "",
    ).strip()

    if not database_url:
        raise DatabaseError(
            "DATABASE_URL environment variable is missing."
        )

    return database_url


def get_connection():
    """
    Open a PostgreSQL connection.
    """

    try:

        return psycopg2.connect(
            get_database_url()
        )

    except Exception as exc:

        raise DatabaseError(
            f"Could not connect to PostgreSQL: {exc}"
        ) from exc


# ============================================================
# CREATE TABLES
# ============================================================

def initialise_database():
    """
    Create the AquaPulse production tables if they do not
    already exist.

    sensor_readings:
        Stores actual AquaSensor measurements.

    forecasts:
        Stores every 15-120 minute DO prediction run.
    """

    connection = get_connection()

    try:

        with connection.cursor() as cursor:

            # ------------------------------------------------
            # ACTUAL SENSOR READINGS
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id BIGSERIAL PRIMARY KEY,

                    sensor_id TEXT NOT NULL,
                    sensor_name TEXT NOT NULL,

                    reading_timestamp TIMESTAMPTZ NOT NULL,

                    location TEXT,

                    latitude DOUBLE PRECISION,
                    longitude DOUBLE PRECISION,

                    temperature DOUBLE PRECISION,

                    dissolved_oxygen_mgl DOUBLE PRECISION,
                    dissolved_oxygen_pct DOUBLE PRECISION,

                    reading_count DOUBLE PRECISION,

                    created_at TIMESTAMPTZ
                        NOT NULL
                        DEFAULT NOW(),

                    UNIQUE (
                        sensor_id,
                        reading_timestamp
                    )
                );
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_sensor_readings_station_time
                ON sensor_readings (
                    sensor_name,
                    reading_timestamp
                );
                """
            )

            # ------------------------------------------------
            # FORECAST HISTORY
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    id BIGSERIAL PRIMARY KEY,

                    sensor_id TEXT NOT NULL,
                    sensor_name TEXT NOT NULL,

                    prediction_run_time TIMESTAMPTZ NOT NULL,

                    latest_sensor_timestamp
                        TIMESTAMPTZ NOT NULL,

                    temperature DOUBLE PRECISION,

                    current_do_mgl DOUBLE PRECISION,
                    current_do_pct DOUBLE PRECISION,

                    anomaly_level TEXT,
                    anomaly_reason TEXT,

                    do_change DOUBLE PRECISION,
                    forecast_min_do DOUBLE PRECISION,

                    predicted_do_mgl_15min
                        DOUBLE PRECISION,

                    predicted_do_mgl_30min
                        DOUBLE PRECISION,

                    predicted_do_mgl_45min
                        DOUBLE PRECISION,

                    predicted_do_mgl_60min
                        DOUBLE PRECISION,

                    predicted_do_mgl_75min
                        DOUBLE PRECISION,

                    predicted_do_mgl_90min
                        DOUBLE PRECISION,

                    predicted_do_mgl_105min
                        DOUBLE PRECISION,

                    predicted_do_mgl_120min
                        DOUBLE PRECISION,

                    forecast_time_15min
                        TIMESTAMPTZ,

                    forecast_time_30min
                        TIMESTAMPTZ,

                    forecast_time_45min
                        TIMESTAMPTZ,

                    forecast_time_60min
                        TIMESTAMPTZ,

                    forecast_time_75min
                        TIMESTAMPTZ,

                    forecast_time_90min
                        TIMESTAMPTZ,

                    forecast_time_105min
                        TIMESTAMPTZ,

                    forecast_time_120min
                        TIMESTAMPTZ,

                    model_15min TEXT,
                    model_30min TEXT,
                    model_45min TEXT,
                    model_60min TEXT,
                    model_75min TEXT,
                    model_90min TEXT,
                    model_105min TEXT,
                    model_120min TEXT,

                    created_at TIMESTAMPTZ
                        NOT NULL
                        DEFAULT NOW(),

                    UNIQUE (
                        sensor_id,
                        latest_sensor_timestamp
                    )
                );
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_forecasts_station_time
                ON forecasts (
                    sensor_name,
                    latest_sensor_timestamp
                );
                """
            )

        connection.commit()

    except Exception as exc:

        connection.rollback()

        raise DatabaseError(
            f"Could not initialise database: {exc}"
        ) from exc

    finally:

        connection.close()


# ============================================================
# SAVE ACTUAL AQUASENSOR READINGS
# ============================================================

def save_sensor_readings(df):
    """
    Save AquaSensor readings into PostgreSQL.

    Existing sensor/timestamp combinations are updated rather
    than duplicated.
    """

    if df is None or df.empty:
        return 0

    required_columns = [
        "sensor_id",
        "sensor_name",
        "timestamp",
        "temperature",
        "dissolved_oxygen_mgl",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise DatabaseError(
            "Sensor dataframe is missing columns: "
            + ", ".join(missing)
        )

    working = df.copy()

    working["timestamp"] = pd.to_datetime(
        working["timestamp"],
        errors="coerce",
        utc=True,
    )

    working = working.dropna(
        subset=[
            "sensor_id",
            "sensor_name",
            "timestamp",
            "temperature",
            "dissolved_oxygen_mgl",
        ]
    )

    if working.empty:
        return 0

    records = []

    for _, row in working.iterrows():

        records.append(
            (
                str(row["sensor_id"]),
                str(row["sensor_name"]),

                row["timestamp"].to_pydatetime(),

                (
                    str(row.get("location", ""))
                    if pd.notna(
                        row.get("location")
                    )
                    else None
                ),

                _number_or_none(
                    row.get("lat")
                ),

                _number_or_none(
                    row.get("lon")
                ),

                _number_or_none(
                    row.get("temperature")
                ),

                _number_or_none(
                    row.get(
                        "dissolved_oxygen_mgl"
                    )
                ),

                _number_or_none(
                    row.get(
                        "dissolved_oxygen_pct"
                    )
                ),

                _number_or_none(
                    row.get("count")
                ),
            )
        )

    sql = """
        INSERT INTO sensor_readings (
            sensor_id,
            sensor_name,
            reading_timestamp,
            location,
            latitude,
            longitude,
            temperature,
            dissolved_oxygen_mgl,
            dissolved_oxygen_pct,
            reading_count
        )
        VALUES %s

        ON CONFLICT (
            sensor_id,
            reading_timestamp
        )

        DO UPDATE SET
            sensor_name =
                EXCLUDED.sensor_name,

            location =
                EXCLUDED.location,

            latitude =
                EXCLUDED.latitude,

            longitude =
                EXCLUDED.longitude,

            temperature =
                EXCLUDED.temperature,

            dissolved_oxygen_mgl =
                EXCLUDED.dissolved_oxygen_mgl,

            dissolved_oxygen_pct =
                EXCLUDED.dissolved_oxygen_pct,

            reading_count =
                EXCLUDED.reading_count;
    """

    connection = get_connection()

    try:

        with connection.cursor() as cursor:

            execute_values(
                cursor,
                sql,
                records,
                page_size=500,
            )

        connection.commit()

        return len(records)

    except Exception as exc:

        connection.rollback()

        raise DatabaseError(
            f"Could not save sensor readings: {exc}"
        ) from exc

    finally:

        connection.close()


# ============================================================
# SAVE FORECASTS
# ============================================================

def save_forecasts(df):
    """
    Save AquaPulse live prediction runs into PostgreSQL.
    """

    if df is None or df.empty:
        return 0

    records = []

    for _, row in df.iterrows():

        sensor_timestamp = pd.to_datetime(
            row.get(
                "latest_sensor_timestamp"
            ),
            errors="coerce",
            utc=True,
        )

        prediction_time = pd.to_datetime(
            row.get(
                "prediction_run_time"
            ),
            errors="coerce",
            utc=True,
        )

        if (
            pd.isna(sensor_timestamp)
            or pd.isna(prediction_time)
        ):
            continue

        records.append(
            (
                str(row["sensor_id"]),
                str(row["sensor_name"]),

                prediction_time.to_pydatetime(),

                sensor_timestamp.to_pydatetime(),

                _number_or_none(
                    row.get("temperature")
                ),

                _number_or_none(
                    row.get("current_do_mgl")
                ),

                _number_or_none(
                    row.get("current_do_pct")
                ),

                _text_or_none(
                    row.get("anomaly_level")
                ),

                _text_or_none(
                    row.get("anomaly_reason")
                ),

                _number_or_none(
                    row.get("do_change")
                ),

                _number_or_none(
                    row.get("forecast_min_do")
                ),

                _number_or_none(
                    row.get(
                        "predicted_do_mgl_15min"
                    )
                ),

                _number_or_none(
                    row.get(
                        "predicted_do_mgl_30min"
                    )
                ),

                _number_or_none(
                    row.get(
                        "predicted_do_mgl_45min"
                    )
                ),

                _number_or_none(
                    row.get(
                        "predicted_do_mgl_60min"
                    )
                ),

                _number_or_none(
                    row.get(
                        "predicted_do_mgl_75min"
                    )
                ),

                _number_or_none(
                    row.get(
                        "predicted_do_mgl_90min"
                    )
                ),

                _number_or_none(
                    row.get(
                        "predicted_do_mgl_105min"
                    )
                ),

                _number_or_none(
                    row.get(
                        "predicted_do_mgl_120min"
                    )
                ),

                _datetime_or_none(
                    row.get(
                        "forecast_time_15min"
                    )
                ),

                _datetime_or_none(
                    row.get(
                        "forecast_time_30min"
                    )
                ),

                _datetime_or_none(
                    row.get(
                        "forecast_time_45min"
                    )
                ),

                _datetime_or_none(
                    row.get(
                        "forecast_time_60min"
                    )
                ),

                _datetime_or_none(
                    row.get(
                        "forecast_time_75min"
                    )
                ),

                _datetime_or_none(
                    row.get(
                        "forecast_time_90min"
                    )
                ),

                _datetime_or_none(
                    row.get(
                        "forecast_time_105min"
                    )
                ),

                _datetime_or_none(
                    row.get(
                        "forecast_time_120min"
                    )
                ),

                _text_or_none(
                    row.get("model_15min")
                ),

                _text_or_none(
                    row.get("model_30min")
                ),

                _text_or_none(
                    row.get("model_45min")
                ),

                _text_or_none(
                    row.get("model_60min")
                ),

                _text_or_none(
                    row.get("model_75min")
                ),

                _text_or_none(
                    row.get("model_90min")
                ),

                _text_or_none(
                    row.get("model_105min")
                ),

                _text_or_none(
                    row.get("model_120min")
                ),
            )
        )

    if not records:
        return 0

    sql = """
        INSERT INTO forecasts (
            sensor_id,
            sensor_name,

            prediction_run_time,
            latest_sensor_timestamp,

            temperature,
            current_do_mgl,
            current_do_pct,

            anomaly_level,
            anomaly_reason,
            do_change,
            forecast_min_do,

            predicted_do_mgl_15min,
            predicted_do_mgl_30min,
            predicted_do_mgl_45min,
            predicted_do_mgl_60min,
            predicted_do_mgl_75min,
            predicted_do_mgl_90min,
            predicted_do_mgl_105min,
            predicted_do_mgl_120min,

            forecast_time_15min,
            forecast_time_30min,
            forecast_time_45min,
            forecast_time_60min,
            forecast_time_75min,
            forecast_time_90min,
            forecast_time_105min,
            forecast_time_120min,

            model_15min,
            model_30min,
            model_45min,
            model_60min,
            model_75min,
            model_90min,
            model_105min,
            model_120min
        )

        VALUES %s

        ON CONFLICT (
            sensor_id,
            latest_sensor_timestamp
        )

        DO UPDATE SET

            prediction_run_time =
                EXCLUDED.prediction_run_time,

            temperature =
                EXCLUDED.temperature,

            current_do_mgl =
                EXCLUDED.current_do_mgl,

            current_do_pct =
                EXCLUDED.current_do_pct,

            anomaly_level =
                EXCLUDED.anomaly_level,

            anomaly_reason =
                EXCLUDED.anomaly_reason,

            do_change =
                EXCLUDED.do_change,

            forecast_min_do =
                EXCLUDED.forecast_min_do,

            predicted_do_mgl_15min =
                EXCLUDED.predicted_do_mgl_15min,

            predicted_do_mgl_30min =
                EXCLUDED.predicted_do_mgl_30min,

            predicted_do_mgl_45min =
                EXCLUDED.predicted_do_mgl_45min,

            predicted_do_mgl_60min =
                EXCLUDED.predicted_do_mgl_60min,

            predicted_do_mgl_75min =
                EXCLUDED.predicted_do_mgl_75min,

            predicted_do_mgl_90min =
                EXCLUDED.predicted_do_mgl_90min,

            predicted_do_mgl_105min =
                EXCLUDED.predicted_do_mgl_105min,

            predicted_do_mgl_120min =
                EXCLUDED.predicted_do_mgl_120min,

            forecast_time_15min =
                EXCLUDED.forecast_time_15min,

            forecast_time_30min =
                EXCLUDED.forecast_time_30min,

            forecast_time_45min =
                EXCLUDED.forecast_time_45min,

            forecast_time_60min =
                EXCLUDED.forecast_time_60min,

            forecast_time_75min =
                EXCLUDED.forecast_time_75min,

            forecast_time_90min =
                EXCLUDED.forecast_time_90min,

            forecast_time_105min =
                EXCLUDED.forecast_time_105min,

            forecast_time_120min =
                EXCLUDED.forecast_time_120min,

            model_15min =
                EXCLUDED.model_15min,

            model_30min =
                EXCLUDED.model_30min,

            model_45min =
                EXCLUDED.model_45min,

            model_60min =
                EXCLUDED.model_60min,

            model_75min =
                EXCLUDED.model_75min,

            model_90min =
                EXCLUDED.model_90min,

            model_105min =
                EXCLUDED.model_105min,

            model_120min =
                EXCLUDED.model_120min;
    """

    connection = get_connection()

    try:

        with connection.cursor() as cursor:

            execute_values(
                cursor,
                sql,
                records,
                page_size=100,
            )

        connection.commit()

        return len(records)

    except Exception as exc:

        connection.rollback()

        raise DatabaseError(
            f"Could not save forecasts: {exc}"
        ) from exc

    finally:

        connection.close()


# ============================================================
# READ SENSOR HISTORY
# ============================================================

def load_sensor_history(
    sensor_name=None,
    limit=None,
):
    """
    Read accumulated actual AquaSensor history from PostgreSQL.
    """

    sql = """
        SELECT
            sensor_id,
            sensor_name,

            reading_timestamp
                AS timestamp,

            location,

            latitude
                AS lat,

            longitude
                AS lon,

            temperature,

            dissolved_oxygen_mgl,
            dissolved_oxygen_pct,

            reading_count
                AS count

        FROM sensor_readings
    """

    parameters = []

    if sensor_name is not None:

        sql += """
            WHERE sensor_name = %s
        """

        parameters.append(
            str(sensor_name)
        )

    sql += """
        ORDER BY reading_timestamp ASC
    """

    if limit is not None:

        sql += """
            LIMIT %s
        """

        parameters.append(
            int(limit)
        )

    connection = get_connection()

    try:

        df = pd.read_sql_query(
            sql,
            connection,
            params=parameters,
        )

    finally:

        connection.close()

    if not df.empty:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

    return df


# ============================================================
# READ FORECAST HISTORY
# ============================================================

def load_forecast_history(
    sensor_name=None,
):
    """
    Read accumulated AquaPulse predictions from PostgreSQL.
    """

    sql = """
        SELECT *
        FROM forecasts
    """

    parameters = []

    if sensor_name is not None:

        sql += """
            WHERE sensor_name = %s
        """

        parameters.append(
            str(sensor_name)
        )

    sql += """
        ORDER BY latest_sensor_timestamp ASC
    """

    connection = get_connection()

    try:

        df = pd.read_sql_query(
            sql,
            connection,
            params=parameters,
        )

    finally:

        connection.close()

    datetime_columns = [
        "prediction_run_time",
        "latest_sensor_timestamp",

        "forecast_time_15min",
        "forecast_time_30min",
        "forecast_time_45min",
        "forecast_time_60min",
        "forecast_time_75min",
        "forecast_time_90min",
        "forecast_time_105min",
        "forecast_time_120min",
    ]

    for column in datetime_columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    return df


# ============================================================
# HELPERS
# ============================================================

def _number_or_none(value):

    number = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(number):
        return None

    return float(number)


def _text_or_none(value):

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except Exception:
        pass

    return str(value)


def _datetime_or_none(value):

    timestamp = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(timestamp):
        return None

    return timestamp.to_pydatetime()


# ============================================================
# DATABASE TEST
# ============================================================

def test_database_connection():
    """
    Simple connection check used during deployment testing.
    """

    connection = get_connection()

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                "SELECT 1;"
            )

            result = cursor.fetchone()

        return (
            result is not None
            and result[0] == 1
        )

    finally:

        connection.close()