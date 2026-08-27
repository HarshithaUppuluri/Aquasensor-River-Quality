import pandas as pd
import os
from datetime import datetime

OUTPUT = "data/processed/live_predictions.csv"

if not os.path.exists(OUTPUT):
    print("No live predictions found yet.")
    print("Run src/live_predict.py first!")
else:
    df = pd.read_csv(OUTPUT)

    if len(df) == 0:
        print("File exists but no predictions saved yet.")
    else:
        print("=" * 55)
        print("LIVE PREDICTION RESULTS")
        print(f"Total predictions made: {len(df)}")
        print(f"File: {OUTPUT}")
        print("=" * 55)

        # Show all predictions in a clean table
        display = df[[
            'predicted_at',
            'sensor_name',
            'current_do_mgl',
            'predicted_do_15min',
            'current_temperature',
        ]].copy()
        display.columns = [
            'Time',
            'Sensor',
            'Current DO (mg/L)',
            'Predicted DO +15min',
            'Water Temp (°C)',
        ]
        print(display.to_string(index=False))

        # Show latest per sensor
        print("\n" + "=" * 55)
        print("MOST RECENT PREDICTION PER SENSOR")
        print("=" * 55)
        latest = df.groupby('sensor_id').last().reset_index()
        for _, row in latest.iterrows():
            diff = (row['predicted_do_15min']
                    - row['current_do_mgl'])
            direction = "↑ rising" if diff > 0 else "↓ falling"
            status = ""
            if row['predicted_do_15min'] < 4.0:
                status = " ⚠ CRITICAL"
            elif row['predicted_do_15min'] < 6.0:
                status = " ⚠ WARNING"
            else:
                status = " ✓ Normal"

            print(f"\n{row['sensor_name']}")
            print(f"  Predicted at:     {row['predicted_at']}")
            print(f"  Water temp:       "
                  f"{row['current_temperature']:.1f} °C")
            print(f"  Current DO:       "
                  f"{row['current_do_mgl']:.2f} mg/L")
            print(f"  Predicted +15min: "
                  f"{row['predicted_do_15min']:.2f} mg/L "
                  f"({direction}){status}")