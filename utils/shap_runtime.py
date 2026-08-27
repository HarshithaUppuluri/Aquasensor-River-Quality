import re

import numpy as np
import pandas as pd

from utils.config import (
    LIVE_SHAP_FILE,
    PROCESSED_FEATURE_FILE,
)
from utils.live_features import build_live_feature_row
from utils.live_weather import fetch_live_weather
from utils.model_runtime import load_horizon_model


class SHAPRuntimeError(RuntimeError):
    pass


def clean_feature_name(name):
    name = str(name)

    name = re.sub(
        r"^(num|cat)__",
        "",
        name,
    )

    exact_names = {
        "temperature":
            "Water temperature",

        "air_temperature_c":
            "Air temperature",

        "sunshine_wm2":
            "Sunshine radiation",

        "hour":
            "Hour of day",

        "dissolved_oxygen_mgl":
            "Current dissolved oxygen",

        "dissolved_oxygen_pct":
            "Current oxygen saturation",

        "pollution_alert":
            "Low oxygen alert",

        "anomaly_type":
            "Anomaly type",

        "season_proxy":
            "Season proxy",
    }

    if name in exact_names:
        return exact_names[name]

    if name.startswith("sensor_id_"):
        return (
            "Sensor ID "
            + name.replace(
                "sensor_id_",
                "",
                1,
            )
        )

    if name.startswith("sensor_name_"):
        return (
            "Sensor "
            + name.replace(
                "sensor_name_",
                "",
                1,
            )
        )

    return name.replace(
        "_",
        " ",
    ).strip()


def load_background_rows(
    feature_names,
    sensor_name=None,
    max_rows=300,
):
    """
    Load historical processed feature rows to use as the SHAP reference
    distribution.

    This is much better than using the live observation itself as the
    reference/background.
    """

    if not PROCESSED_FEATURE_FILE.exists():
        raise SHAPRuntimeError(
            "Historical processed dataset was not found."
        )

    try:
        data = pd.read_csv(
            PROCESSED_FEATURE_FILE,
            low_memory=False,
        )

    except Exception as exc:

        raise SHAPRuntimeError(
            f"Could not load historical dataset: {exc}"
        ) from exc

    missing = [
        feature
        for feature in feature_names
        if feature not in data.columns
    ]

    if missing:

        raise SHAPRuntimeError(
            "Historical dataset is missing model features: "
            + ", ".join(missing)
        )

    # Prefer data from the same sensor.
    if (
        sensor_name is not None
        and "sensor_name" in data.columns
    ):

        same_sensor = data[
            data["sensor_name"].astype(str)
            == str(sensor_name)
        ]

        if len(same_sensor) >= 30:
            data = same_sensor

    background = data[
        feature_names
    ].copy()

    # Ensure categorical columns are strings.
    for column in [
        "sensor_id",
        "sensor_name",
    ]:

        if column in background.columns:
            background[column] = (
                background[column]
                .astype(str)
            )

    # Convert numeric features correctly.
    numeric_columns = [
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

    for column in numeric_columns:

        if column in background.columns:

            background[column] = pd.to_numeric(
                background[column],
                errors="coerce",
            )

    background = background.dropna()

    if background.empty:

        raise SHAPRuntimeError(
            "No valid historical background rows are available."
        )

    # Use a representative sample rather than thousands of rows.
    if len(background) > max_rows:

        background = background.sample(
            n=max_rows,
            random_state=42,
        )

    return background


def explain_prediction(
    sensor_row,
    horizon="60min",
):
    """
    Explain one live DO prediction using SHAP.

    The historical processed dataset provides the SHAP background.
    """

    try:
        import shap

    except ImportError as exc:

        raise SHAPRuntimeError(
            "SHAP is not installed. "
            "Run: python -m pip install shap"
        ) from exc

    # ---------------------------------------------------------
    # Load correct model
    # ---------------------------------------------------------

    model, meta, model_path = (
        load_horizon_model(
            horizon
        )
    )

    # ---------------------------------------------------------
    # Live weather
    # ---------------------------------------------------------

    latitude = float(
        sensor_row["latitude"]
        if "latitude" in sensor_row.index
        else sensor_row["lat"]
    )

    longitude = float(
        sensor_row["longitude"]
        if "longitude" in sensor_row.index
        else sensor_row["lon"]
    )

    weather = fetch_live_weather(
        latitude,
        longitude,
    )

    # ---------------------------------------------------------
    # Build exact live model feature row
    # ---------------------------------------------------------

    X_live = build_live_feature_row(
        sensor_row,
        weather,
        model,
    )

    # ---------------------------------------------------------
    # Get pipeline components
    # ---------------------------------------------------------

    if not hasattr(
        model,
        "named_steps",
    ):

        raise SHAPRuntimeError(
            "Saved model is not a sklearn Pipeline."
        )

    if "preprocessor" not in model.named_steps:

        raise SHAPRuntimeError(
            "Pipeline does not contain 'preprocessor'."
        )

    if "model" not in model.named_steps:

        raise SHAPRuntimeError(
            "Pipeline does not contain 'model'."
        )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    estimator = model.named_steps[
        "model"
    ]

    # ---------------------------------------------------------
    # Historical SHAP background
    # ---------------------------------------------------------

    feature_names_raw = list(
        X_live.columns
    )

    background_raw = load_background_rows(
        feature_names_raw,
        sensor_name=str(
            sensor_row["sensor_name"]
        ),
        max_rows=300,
    )

    # ---------------------------------------------------------
    # Transform through EXACT trained preprocessor
    # ---------------------------------------------------------

    try:

        X_live_transformed = (
            preprocessor.transform(
                X_live
            )
        )

        background_transformed = (
            preprocessor.transform(
                background_raw
            )
        )

    except Exception as exc:

        raise SHAPRuntimeError(
            f"Preprocessing for SHAP failed: {exc}"
        ) from exc

    # ---------------------------------------------------------
    # Convert sparse matrices where needed
    # ---------------------------------------------------------

    if hasattr(
        X_live_transformed,
        "toarray",
    ):

        X_live_dense = (
            X_live_transformed.toarray()
        )

    else:

        X_live_dense = np.asarray(
            X_live_transformed
        )

    if hasattr(
        background_transformed,
        "toarray",
    ):

        background_dense = (
            background_transformed.toarray()
        )

    else:

        background_dense = np.asarray(
            background_transformed
        )

    # ---------------------------------------------------------
    # Get transformed feature names
    # ---------------------------------------------------------

    try:

        transformed_feature_names = (
            preprocessor.get_feature_names_out()
        )

    except Exception:

        transformed_feature_names = [
            f"feature_{i}"
            for i in range(
                X_live_dense.shape[1]
            )
        ]

    # ---------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------

    try:

        estimator_name = (
            estimator.__class__.__name__
            .lower()
        )

        if (
            "linear" in estimator_name
            or "ridge" in estimator_name
            or "lasso" in estimator_name
        ):

            explainer = (
                shap.LinearExplainer(
                    estimator,
                    background_dense,
                )
            )

            shap_values = (
                explainer.shap_values(
                    X_live_dense
                )
            )

        elif (
            "forest" in estimator_name
            or "xgb" in estimator_name
            or "boost" in estimator_name
            or "tree" in estimator_name
        ):

            explainer = (
                shap.TreeExplainer(
                    estimator,
                    data=background_dense,
                )
            )

            shap_values = (
                explainer.shap_values(
                    X_live_dense
                )
            )

        else:

            explainer = shap.Explainer(
                estimator.predict,
                background_dense,
            )

            explanation = explainer(
                X_live_dense
            )

            shap_values = (
                explanation.values
            )

    except Exception as exc:

        raise SHAPRuntimeError(
            f"SHAP explanation failed: {exc}"
        ) from exc

    # ---------------------------------------------------------
    # Normalise SHAP output shape
    # ---------------------------------------------------------

    values = np.asarray(
        shap_values
    )

    if values.ndim == 3:
        values = values[:, :, 0]

    if values.ndim == 2:
        values = values[0]

    values = values.reshape(-1)

    # ---------------------------------------------------------
    # Create output table
    # ---------------------------------------------------------

    if len(values) != len(
        transformed_feature_names
    ):

        raise SHAPRuntimeError(
            "SHAP feature-count mismatch."
        )

    result = pd.DataFrame(
        {
            "feature": [
                clean_feature_name(
                    name
                )
                for name
                in transformed_feature_names
            ],

            "shap_value":
                values,
        }
    )

    result[
        "absolute_shap_value"
    ] = result[
        "shap_value"
    ].abs()

    result = result.sort_values(
        "absolute_shap_value",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # Add metadata
    # ---------------------------------------------------------

    result.insert(
        0,
        "sensor_name",
        str(
            sensor_row[
                "sensor_name"
            ]
        ),
    )

    result.insert(
        1,
        "horizon",
        horizon,
    )

    result.insert(
        2,
        "model",
        str(
            meta["model"]
        ),
    )

    # ---------------------------------------------------------
    # Save latest SHAP output
    # ---------------------------------------------------------

    LIVE_SHAP_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        LIVE_SHAP_FILE,
        index=False,
    )

    return result