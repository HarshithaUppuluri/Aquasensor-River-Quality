import pandas as pd
from utils.aquasensor_api import load_live_history

def assess_live_anomaly(sensor_name,current_do,current_temp,predictions):
    vals=[float(v) for v in predictions.values() if v is not None and pd.notna(v)]
    fmin=min(vals) if vals else None
    h=load_live_history(sensor_name); prev=None
    if len(h)>=2:
        v=pd.to_numeric(h.iloc[-2]["dissolved_oxygen_mgl"],errors="coerce")
        if pd.notna(v): prev=float(v)
    change=None if prev is None else float(current_do)-prev
    if current_do<4: return {"level":"Red","reason":"Current dissolved oxygen is below 4 mg/L.","do_change":change,"forecast_min":fmin}
    if fmin is not None and fmin<4: return {"level":"Red","reason":"At least one 15–120 minute DO forecast is below 4 mg/L.","do_change":change,"forecast_min":fmin}
    if change is not None and change<=-2: return {"level":"Red","reason":f"DO fell rapidly by {abs(change):.2f} mg/L.","do_change":change,"forecast_min":fmin}
    if current_do<6: return {"level":"Orange","reason":"Current dissolved oxygen is between 4 and 6 mg/L.","do_change":change,"forecast_min":fmin}
    if fmin is not None and fmin<6: return {"level":"Orange","reason":"At least one 15–120 minute forecast drops below 6 mg/L.","do_change":change,"forecast_min":fmin}
    if change is not None and change<=-1: return {"level":"Orange","reason":f"DO fell by {abs(change):.2f} mg/L.","do_change":change,"forecast_min":fmin}
    if current_temp>=25: return {"level":"Orange","reason":"Water temperature is elevated; continue monitoring dissolved oxygen.","do_change":change,"forecast_min":fmin}
    return {"level":"Green","reason":"No configured live DO warning or anomaly was detected.","do_change":change,"forecast_min":fmin}
