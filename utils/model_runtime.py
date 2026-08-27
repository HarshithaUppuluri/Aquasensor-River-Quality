import re
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from utils.config import (
    HORIZONS,
    MODELS_DIR,
    PERFORMANCE_FILE,
)

from utils.live_features import (
    build_live_feature_row,
)

from utils.live_weather import (
    fetch_live_weather,
)


# ============================================================
# CUSTOM ERROR
# ============================================================

class ModelRuntimeError(RuntimeError):
    pass


# ============================================================
# HELPERS
# ============================================================

def _slug(value):
    """
    Convert a value into a simple lowercase string containing
    only letters and numbers.

    This is used when matching model filenames with model
    metadata.
    """

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


# ============================================================
# MODEL PERFORMANCE TABLE
# ============================================================

@lru_cache(maxsize=1)
def performance():
    """
    Load and cache the model-performance table.

    The table is used to determine the best trained model
    for each dissolved oxygen forecast horizon.
    """

    if not PERFORMANCE_FILE.exists():

        raise ModelRuntimeError(
            f"Missing model performance file: "
            f"{PERFORMANCE_FILE}"
        )

    return pd.read_csv(
        PERFORMANCE_FILE,
        low_memory=False,
    )


# ============================================================
# BEST MODEL METADATA
# ============================================================

def best_meta(horizon):
    """
    Select the best DO mg/L model for a forecast horizon.

    Preference:
    1. Highest R² when available.
    2. Lowest RMSE when R² is unavailable.
    """

    df = performance()

    selected = df[
        (
            df["target_type"]
            .astype(str)
            .str.lower()
            == "do mg/l"
        )
        &
        (
            df["horizon"]
            .astype(str)
            == str(horizon)
        )
    ].copy()

    if selected.empty:

        raise ModelRuntimeError(
            f"No DO mg/L performance row "
            f"found for {horizon}"
        )

    if "R2" in selected.columns:

        selected["R2"] = pd.to_numeric(
            selected["R2"],
            errors="coerce",
        )

        selected = selected.sort_values(
            "R2",
            ascending=False,
        )

    elif "RMSE" in selected.columns:

        selected["RMSE"] = pd.to_numeric(
            selected["RMSE"],
            errors="coerce",
        )

        selected = selected.sort_values(
            "RMSE",
            ascending=True,
        )

    return selected.iloc[0]


# ============================================================
# MODEL FILE MATCHING
# ============================================================

def _score(
    path,
    target,
    horizon,
    model,
):
    """
    Score a model filename based on how closely it matches
    the target variable, forecast horizon and model name.
    """

    filename = _slug(
        path.stem
    )

    score = 0

    if _slug(target) in filename:
        score += 100

    if _slug(horizon) in filename:
        score += 40

    if _slug(model) in filename:
        score += 60

    model_name = model.lower()

    if "linear" in model_name:

        tokens = (
            "linear",
            "linreg",
            "lr",
        )

    elif "random" in model_name:

        tokens = (
            "randomforest",
            "rf",
        )

    else:

        tokens = (
            "xgboost",
            "xgb",
        )

    for token in tokens:

        if token in filename:
            score += 20

    return score


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@lru_cache(maxsize=None)
def load_horizon_model(horizon):
    """
    Load and cache the best trained model for a given
    dissolved oxygen forecast horizon.
    """

    meta = best_meta(
        horizon
    )

    target = str(
        meta["target_column"]
    )

    model_name = str(
        meta["model"]
    )

    model_files = []

    for pattern in (
        "*.joblib",
        "*.pkl",
        "*.pickle",
    ):

        model_files.extend(
            MODELS_DIR.rglob(
                pattern
            )
        )

    if not model_files:

        raise ModelRuntimeError(
            f"No model files found in "
            f"{MODELS_DIR}"
        )

    best_file = max(
        model_files,
        key=lambda path: _score(
            path,
            target,
            horizon,
            model_name,
        ),
    )

    best_score = _score(
        best_file,
        target,
        horizon,
        model_name,
    )

    if best_score <= 0:

        raise ModelRuntimeError(
            f"Cannot match a trained model "
            f"for horizon {horizon}"
        )

    try:

        model = joblib.load(
            best_file
        )

    except Exception as exc:

        raise ModelRuntimeError(
            f"Could not load model file: "
            f"{best_file}"
        ) from exc

    return (
        model,
        meta,
        best_file,
    )


# ============================================================
# LIVE PREDICTION
# ============================================================

def predict_sensor(sensor):
    """
    Generate dissolved oxygen predictions for one live
    AquaSensor reading across every configured horizon.
    """

    lat = float(
        sensor["lat"]
    )

    lon = float(
        sensor["lon"]
    )

    weather = fetch_live_weather(
        lat,
        lon,
    )

    predictions = {}

    metadata = {}

    for horizon in HORIZONS:

        model, meta, model_path = (
            load_horizon_model(
                horizon
            )
        )

        X_live = build_live_feature_row(
            sensor,
            weather,
            model,
        )

        prediction = model.predict(
            X_live
        )

        prediction_value = float(
            np.asarray(
                prediction
            )
            .reshape(-1)[0]
        )

        predictions[
            horizon
        ] = prediction_value

        metadata[
            horizon
        ] = {
            "model": str(
                meta["model"]
            ),
            "target_column": str(
                meta["target_column"]
            ),
            "model_path": str(
                model_path
            ),
            "X_live": X_live,
        }

    return (
        predictions,
        metadata,
    )