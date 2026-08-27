from datetime import timedelta
import pandas as pd
from utils.anomaly import assess_live_anomaly
from utils.aquasensor_api import fetch_aquasensor_live, append_live_history
from utils.config import HORIZONS,HORIZON_MINUTES,LIVE_FORECAST_FILE
from utils.model_runtime import predict_sensor

def refresh_live_system():
    sensors=fetch_aquasensor_live(); append_live_history(sensors); rows=[]
    for _,s in sensors.iterrows():
        preds,meta=predict_sensor(s)
        an=assess_live_anomaly(str(s["sensor_name"]),float(s["dissolved_oxygen_mgl"]),float(s["temperature"]),preds)
        row={"prediction_run_time":pd.Timestamp.now(tz="Europe/London"),"latest_sensor_timestamp":s["timestamp"],
             "sensor_id":s["sensor_id"],"sensor_name":s["sensor_name"],"temperature":s["temperature"],
             "current_do_mgl":s["dissolved_oxygen_mgl"],"current_do_pct":s["dissolved_oxygen_pct"],
             "anomaly_level":an["level"],"anomaly_reason":an["reason"],"do_change":an["do_change"],"forecast_min_do":an["forecast_min"]}
        base=pd.to_datetime(s["timestamp"])
        for h in HORIZONS:
            row[f"predicted_do_mgl_{h}"]=preds[h]
            row[f"forecast_time_{h}"]=base+timedelta(minutes=HORIZON_MINUTES[h])
            row[f"model_{h}"]=meta[h]["model"]
        rows.append(row)
    df=pd.DataFrame(rows); LIVE_FORECAST_FILE.parent.mkdir(parents=True,exist_ok=True)
    if LIVE_FORECAST_FILE.exists():
        old=pd.read_csv(LIVE_FORECAST_FILE,low_memory=False); combo=pd.concat([old,df],ignore_index=True)
    else: combo=df
    combo=combo.drop_duplicates(subset=["sensor_id","latest_sensor_timestamp"],keep="last")
    combo.to_csv(LIVE_FORECAST_FILE,index=False)
    return sensors,df
