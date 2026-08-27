from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PERFORMANCE_FILE = PROCESSED_DIR / "do_prediction_model_performance.csv"
PROCESSED_FEATURE_FILE = PROCESSED_DIR / "aquasensor_final.csv"
LIVE_API_HISTORY_FILE = PROCESSED_DIR / "live_aquasensor_api_readings.csv"
LIVE_FORECAST_FILE = PROCESSED_DIR / "live_api_do_forecasts.csv"
LIVE_SHAP_FILE = PROCESSED_DIR / "live_api_shap.csv"
AQUASENSOR_BASE_URL = "https://api.aquasensor.co.uk/aqapi.php"
HORIZONS = ["15min","30min","45min","60min","75min","90min","105min","120min"]
HORIZON_LABELS = {"15min":"15 min","30min":"30 min","45min":"45 min","60min":"1 hour","75min":"1 h 15 min","90min":"1 h 30 min","105min":"1 h 45 min","120min":"2 hours"}
HORIZON_MINUTES = {"15min":15,"30min":30,"45min":45,"60min":60,"75min":75,"90min":90,"105min":105,"120min":120}
STATIONS = ["Derwent 13","Derwent 13-50","Derwent 21"]
SENSOR_NAME_MAP = {"sensor022":"Derwent 13","941115":"Derwent 13-50","94115":"Derwent 13-50","941205":"Derwent 21"}
MODEL_FEATURE_FALLBACK = ["temperature","air_temperature_c","sunshine_wm2","hour","dissolved_oxygen_mgl","dissolved_oxygen_pct","pollution_alert","anomaly_type","season_proxy","sensor_id","sensor_name"]
