# AquaPulse V2

A clean Streamlit multipage application showing only:

- Live dissolved oxygen (mg/L)
- Live physical water temperature (°C)
- DO forecasts from 15 to 120 minutes
- Home, Alerts and Analysis navigation

## Install

```powershell
pip install -r requirements.txt
```

## Place in the dissertation project

Copy the complete `AquaPulse_V2` folder into the root of the existing
`aquasensor-river-quality` project, alongside the `data` folder.

Expected project paths:

```text
aquasensor-river-quality/
├── AquaPulse_V2/
├── data/
│   ├── raw/
│   │   └── aquasensor_all_sensors.csv
│   └── processed/
│       └── live_river_do_forecasts.csv
```

The loader also recognises individual exports such as:

- `022 sensorreadings*.csv`
- `941115 sensorreadings*.csv`
- `941205 sensorreadings*.csv`

## Run

```powershell
cd AquaPulse_V2
python -m streamlit run AquaPulse_dashboard.py
```

The application checks source files every 30 seconds. A dashboard cannot make
a historical file live: the AquaSensor collection/API script must keep the
physical sensor source updated.
