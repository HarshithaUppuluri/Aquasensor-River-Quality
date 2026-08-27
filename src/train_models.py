import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


DATA_PATH = "data/processed/aquasensor_final.csv"
MODELS_DIR = "models"
OUTPUT_DIR = "data/processed"

PERFORMANCE_PATH = "data/processed/do_prediction_model_performance.csv"
ALL_PREDICTIONS_PATH = "data/processed/all_model_predictions.csv"
FORECASTS_PATH = "data/processed/river_do_forecasts.csv"

HORIZONS = ["15min", "30min", "45min", "60min", "75min", "90min", "105min", "120min"]

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = ["sensor_id", "sensor_name"]


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run python src/preprocess.py first.")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"], low_memory=False)
    df["sensor_id"] = df["sensor_id"].astype(str)
    df["sensor_name"] = df["sensor_name"].astype(str)

    print(f"Loaded processed data: {len(df)} rows")
    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def safe_filename_text(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("%", "pct")
        .replace("/", "_per_")
    )


def train_single_target(df, target_col, target_type, horizon):
    model_df = df.dropna(subset=[target_col]).copy()

    X = model_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = model_df[target_col]

    meta = model_df[
        [
            "timestamp",
            "sensor_id",
            "sensor_name",
            "dissolved_oxygen_mgl",
            "dissolved_oxygen_pct",
            "pollution_alert",
            "anomaly_type",
        ]
    ].copy()

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        meta,
        test_size=0.2,
        shuffle=False,
    )

    metrics_rows = []
    prediction_rows = []

    print("\n" + "=" * 75)
    print(f"TARGET: {target_type} | HORIZON: {horizon}")
    print("=" * 75)
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")

    for model_name, model in get_models().items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae, rmse, r2 = calculate_metrics(y_test, y_pred)

        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R2   : {r2:.4f}")

        metrics_rows.append(
            {
                "target_type": target_type,
                "horizon": horizon,
                "target_column": target_col,
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )

        model_file = (
            f"{safe_filename_text(model_name)}_"
            f"{safe_filename_text(target_type)}_"
            f"{horizon}.pkl"
        )

        model_path = os.path.join(MODELS_DIR, model_file)
        joblib.dump(pipeline, model_path)

        pred_df = meta_test.copy()
        pred_df["target_type"] = target_type
        pred_df["horizon"] = horizon
        pred_df["model"] = model_name
        pred_df["actual_value"] = y_test.values
        pred_df["predicted_value"] = y_pred
        pred_df["absolute_error"] = np.abs(y_test.values - y_pred)

        prediction_rows.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)

    best_row = metrics_df.sort_values("RMSE").iloc[0]

    print("\nBest model:")
    print(f"  {best_row['model']} | RMSE = {best_row['RMSE']:.4f}")

    print("\nSample predictions:")
    sample = predictions_df[predictions_df["model"] == best_row["model"]].head(8)
    print(
        sample[
            [
                "timestamp",
                "sensor_id",
                "sensor_name",
                "actual_value",
                "predicted_value",
            ]
        ].to_string(index=False)
    )

    return metrics_df, predictions_df


def train_all_models(df):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = []
    all_predictions = []

    for horizon in HORIZONS:
        targets = [
            ("DO mg/L", f"do_mgl_next_{horizon}"),
            ("DO %", f"do_pct_next_{horizon}"),
        ]

        for target_type, target_col in targets:
            if target_col in df.columns:
                metrics_df, predictions_df = train_single_target(
                    df=df,
                    target_col=target_col,
                    target_type=target_type,
                    horizon=horizon,
                )
                all_metrics.append(metrics_df)
                all_predictions.append(predictions_df)

    final_metrics = pd.concat(all_metrics, ignore_index=True)
    final_predictions = pd.concat(all_predictions, ignore_index=True)

    final_metrics.to_csv(PERFORMANCE_PATH, index=False)
    final_predictions.to_csv(ALL_PREDICTIONS_PATH, index=False)

    print("\n" + "=" * 75)
    print("MODEL TRAINING COMPLETE")
    print("=" * 75)
    print(f"Model performance saved to: {PERFORMANCE_PATH}")
    print(f"All model predictions saved to: {ALL_PREDICTIONS_PATH}")
    print(f"Trained models saved in: {MODELS_DIR}")

    return final_metrics, final_predictions


def create_river_do_forecasts(predictions_df):
    forecast_tables = []

    for target_type in ["DO mg/L", "DO %"]:
        subset = predictions_df[predictions_df["target_type"] == target_type]

        best_model = (
            subset.groupby("model")["absolute_error"]
            .mean()
            .sort_values()
            .index[0]
        )

        best_subset = subset[subset["model"] == best_model].copy()

        pivot = best_subset.pivot_table(
            index=[
                "timestamp",
                "sensor_id",
                "sensor_name",
                "dissolved_oxygen_mgl",
                "dissolved_oxygen_pct",
                "pollution_alert",
                "anomaly_type",
            ],
            columns="horizon",
            values="predicted_value",
            aggfunc="first",
        ).reset_index()

        if target_type == "DO mg/L":
            pivot = pivot.rename(
                columns={
                    "15min": "predicted_do_mgl_15min",
                    "30min": "predicted_do_mgl_30min",
                    "45min": "predicted_do_mgl_45min",
                    "60min": "predicted_do_mgl_1hr",
                    "75min": "predicted_do_mgl_1hr15min",
                    "90min": "predicted_do_mgl_1hr30min",
                    "105min": "predicted_do_mgl_1hr45min",
                    "120min": "predicted_do_mgl_2hr",
                }
            )
            pivot["best_model_do_mgl"] = best_model
        else:
            pivot = pivot.rename(
                columns={
                    "15min": "predicted_do_pct_15min",
                    "30min": "predicted_do_pct_30min",
                    "45min": "predicted_do_pct_45min",
                    "60min": "predicted_do_pct_1hr",
                    "75min": "predicted_do_pct_1hr15min",
                    "90min": "predicted_do_pct_1hr30min",
                    "105min": "predicted_do_pct_1hr45min",
                    "120min": "predicted_do_pct_2hr",
                }
            )
            pivot["best_model_do_pct"] = best_model

        forecast_tables.append(pivot)

    river_do_forecasts = forecast_tables[0].merge(
        forecast_tables[1],
        on=[
            "timestamp",
            "sensor_id",
            "sensor_name",
            "dissolved_oxygen_mgl",
            "dissolved_oxygen_pct",
            "pollution_alert",
            "anomaly_type",
        ],
        how="outer",
    )

    river_do_forecasts = river_do_forecasts.rename(
        columns={
            "dissolved_oxygen_mgl": "current_do_mgl",
            "dissolved_oxygen_pct": "current_do_pct",
        }
    )

    river_do_forecasts.to_csv(FORECASTS_PATH, index=False)

    print("\n" + "=" * 75)
    print("RIVER DO FORECASTS CREATED")
    print("=" * 75)
    print(f"Saved to: {FORECASTS_PATH}")
    print("\nFirst 10 rows:")
    print(river_do_forecasts.head(10).to_string(index=False))

    return river_do_forecasts


def main():
    print("=" * 75)
    print("AquaSensor Multi-Horizon DO Model Training")
    print("Linear Regression | Random Forest | XGBoost")
    print("=" * 75)

    df = load_data()
    metrics_df, predictions_df = train_all_models(df)
    create_river_do_forecasts(predictions_df)

    print("\nBest models by target and horizon:")
    best_models = (
        metrics_df.sort_values("RMSE")
        .groupby(["target_type", "horizon"])
        .head(1)[["target_type", "horizon", "model", "MAE", "RMSE", "R2"]]
    )

    print(best_models.to_string(index=False))

    print("\nImportant output files:")
    print(f"  1. {PERFORMANCE_PATH}")
    print(f"  2. {ALL_PREDICTIONS_PATH}")
    print(f"  3. {FORECASTS_PATH}")


if __name__ == "__main__":
    main()