"""
DO 15-Minute Prediction Model
==============================
Trains 3 models and generates predictions.
This is done FIRST before anomaly detection
and dashboard — as confirmed by supervisor 19/06/2026.

5 model input features:
  temperature, air_temperature_c, sunshine_wm2,
  hour, season_proxy

Target:
  do_next_15min

3 models compared:
  1. Linear Regression (baseline)
  2. Random Forest
  3. XGBoost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (mean_absolute_error,
                              mean_squared_error,
                              r2_score)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# STEP 1 — Load preprocessed data
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("DO 15-Min Prediction — Model Training")
print("=" * 60)

path = "data/processed/aquasensor_final.csv"
if not os.path.exists(path):
    raise FileNotFoundError(
        "data/processed/aquasensor_final.csv not found.\n"
        "Run src/preprocess.py first!"
    )

df = pd.read_csv(path, parse_dates=['timestamp'])
print(f"Data loaded: {len(df)} rows")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Define features and target
# ─────────────────────────────────────────────────────────────
# These are the 5 NUMERIC inputs to the ML model
# (the other 5 of your 10 variables are
#  labels/identity columns, not model inputs)
FEATURES = [
    'temperature',           # variable 1 — water temperature
    'air_temperature_c',     # variable 2 — air temperature
    'sunshine_wm2',          # variable 3 — sunshine
    'hour',                  # variable 4 — hour of day
    'season_proxy',          # variable 10 — season proxy
]
TARGET = 'do_next_15min'

# Drop rows where target is missing
df_model = df[FEATURES + [TARGET,
                           'sensor_id',
                           'timestamp']].dropna()

X = df_model[FEATURES]
y = df_model[TARGET]

print(f"\nModel features:   {FEATURES}")
print(f"Target:           {TARGET}")
print(f"Total samples:    {len(X)}")
print(f"Target range:     {y.min():.2f} to {y.max():.2f} mg/L")
print(f"Target mean:      {y.mean():.2f} mg/L")

# ─────────────────────────────────────────────────────────────
# STEP 3 — Split into train and test
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain samples:    {len(X_train)} (80%)")
print(f"Test samples:     {len(X_test)} (20%)")

# ─────────────────────────────────────────────────────────────
# STEP 4 — Train 3 models
# ─────────────────────────────────────────────────────────────

def train_and_evaluate(name, model, X_tr, y_tr, X_te, y_te):
    print(f"\nTraining {name}...")
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)
    print(f"  MAE:  {mae:.4f} mg/L  "
          f"(average prediction error)")
    print(f"  RMSE: {rmse:.4f} mg/L  "
          f"(penalises large errors)")
    print(f"  R²:   {r2:.4f}        "
          f"(1.0 = perfect, 0 = baseline)")
    return model, preds, {
        'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2
    }


print("\n" + "-" * 60)
print("Training 3 models — this may take 1-2 minutes")
print("-" * 60)

lr,  lr_preds,  lr_res  = train_and_evaluate(
    "Linear Regression (baseline)",
    LinearRegression(),
    X_train, y_train, X_test, y_test
)

rf,  rf_preds,  rf_res  = train_and_evaluate(
    "Random Forest",
    RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ),
    X_train, y_train, X_test, y_test
)

xgb, xgb_preds, xgb_res = train_and_evaluate(
    "XGBoost",
    XGBRegressor(
        n_estimators=150,
        learning_rate=0.08,
        random_state=42,
        verbosity=0
    ),
    X_train, y_train, X_test, y_test
)

# ─────────────────────────────────────────────────────────────
# STEP 5 — Compare all models
# ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame([lr_res, rf_res, xgb_res]).round(4)

print("\n" + "=" * 60)
print("MODEL COMPARISON — DO 15-MIN PREDICTION")
print("=" * 60)
print(results_df.to_string(index=False))

best = results_df.loc[results_df['R2'].idxmax()]
print(f"\nBest model: {best['Model']}")
print(f"  R²   = {best['R2']:.4f}")
print(f"  MAE  = {best['MAE']:.4f} mg/L")
print(f"  RMSE = {best['RMSE']:.4f} mg/L")

os.makedirs("data/processed", exist_ok=True)
results_df.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)
print("\nComparison table saved: "
      "data/processed/model_comparison.csv")

# ─────────────────────────────────────────────────────────────
# STEP 6 — Generate predictions for ALL data
# ─────────────────────────────────────────────────────────────
print("\nGenerating predictions for full dataset...")

df_predict = df[FEATURES + ['sensor_id',
                             'timestamp',
                             'dissolved_oxygen_mgl',
                             'do_next_15min',
                             'pollution_alert',
                             'anomaly_type']].copy()
df_predict = df_predict.dropna(subset=FEATURES)

df_predict['do_predicted_lr']  = lr.predict(
    df_predict[FEATURES]
)
df_predict['do_predicted_rf']  = rf.predict(
    df_predict[FEATURES]
)
df_predict['do_predicted_xgb'] = xgb.predict(
    df_predict[FEATURES]
)

# Use the best model as the main prediction column
best_name = best['Model']
if 'XGBoost' in best_name:
    df_predict['do_predicted'] = df_predict['do_predicted_xgb']
elif 'Random Forest' in best_name:
    df_predict['do_predicted'] = df_predict['do_predicted_rf']
else:
    df_predict['do_predicted'] = df_predict['do_predicted_lr']

df_predict['do_residual'] = (
    df_predict['dissolved_oxygen_mgl']
    - df_predict['do_predicted']
)

df_predict.to_csv(
    "data/processed/aquasensor_with_predictions.csv",
    index=False
)
print("Predictions saved: "
      "data/processed/aquasensor_with_predictions.csv")

# ─────────────────────────────────────────────────────────────
# STEP 7 — Save models
# ─────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(lr,  "models/linear_regression.pkl")
joblib.dump(rf,  "models/random_forest.pkl")
joblib.dump(xgb, "models/xgboost.pkl")
print("\nModels saved:")
print("  models/linear_regression.pkl")
print("  models/random_forest.pkl")
print("  models/xgboost.pkl")

# ─────────────────────────────────────────────────────────────
# STEP 8 — Plot results
# ─────────────────────────────────────────────────────────────
print("\nGenerating charts...")
os.makedirs("figures", exist_ok=True)

# --- Chart 1: Model comparison bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle(
    'DO 15-Minute Prediction — Model Comparison\n'
    'River Derwent — AquaSensor Data',
    fontsize=13, fontweight='bold'
)
models  = ['Linear\nRegression', 'Random\nForest', 'XGBoost']
colours = ['#3498DB', '#27AE60', '#E67E22']

axes[0].bar(models,
            [lr_res['MAE'], rf_res['MAE'], xgb_res['MAE']],
            color=colours)
axes[0].set_title('MAE  (lower = better)')
axes[0].set_ylabel('mg/L')

axes[1].bar(models,
            [lr_res['RMSE'], rf_res['RMSE'], xgb_res['RMSE']],
            color=colours)
axes[1].set_title('RMSE  (lower = better)')
axes[1].set_ylabel('mg/L')

axes[2].bar(models,
            [lr_res['R2'], rf_res['R2'], xgb_res['R2']],
            color=colours)
axes[2].set_title('R²  (higher = better)')
axes[2].set_ylabel('R² score')
axes[2].set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('figures/01_model_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/01_model_comparison.png")

# --- Chart 2: Actual vs Predicted for each sensor ---
sensors   = df_predict['sensor_id'].unique()
sen_names = {
    'sensor022': 'Derwent 13',
    '941115':    'Derwent 13-50',
    '941205':    'Derwent 21'
}

fig, axes = plt.subplots(len(sensors), 1,
                          figsize=(16, 5 * len(sensors)))
if len(sensors) == 1:
    axes = [axes]

fig.suptitle(
    f'Actual vs Predicted DO (15 min ahead)\n'
    f'Best model: {best_name}',
    fontsize=13, fontweight='bold'
)

for i, sid in enumerate(sensors):
    s = df_predict[
        df_predict['sensor_id'] == sid
    ].sort_values('timestamp').tail(500)

    ax = axes[i]
    ax.plot(s['timestamp'],
            s['dissolved_oxygen_mgl'],
            color='#2196F3', linewidth=1.2,
            label='Actual DO', alpha=0.9)
    ax.plot(s['timestamp'],
            s['do_predicted'],
            color='#FF9800', linewidth=1.2,
            linestyle='--',
            label='Predicted DO (15 min ahead)', alpha=0.9)
    ax.axhline(y=4.0, color='red',
               linestyle=':', linewidth=1.2,
               label='Critical (4 mg/L)')
    ax.set_title(
        f"{sen_names.get(sid, sid)}  —  "
        f"Last 500 readings",
        fontsize=11
    )
    ax.set_ylabel('DO (mg/L)')
    ax.legend(loc='upper right', fontsize=9)
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%d %b')
    )
    ax.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

plt.tight_layout()
plt.savefig('figures/02_actual_vs_predicted.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/02_actual_vs_predicted.png")

# --- Chart 3: Scatter — actual vs predicted ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    'Scatter: Actual vs Predicted DO (test set)\n'
    'Perfect prediction = all dots on the diagonal line',
    fontsize=12, fontweight='bold'
)

pairs = [
    ('Linear Regression', y_test.values, lr_preds,
     '#3498DB', lr_res['R2']),
    ('Random Forest',     y_test.values, rf_preds,
     '#27AE60', rf_res['R2']),
    ('XGBoost',           y_test.values, xgb_preds,
     '#E67E22', xgb_res['R2']),
]

for ax, (name, actual, predicted, col, r2) in zip(axes, pairs):
    ax.scatter(actual, predicted, alpha=0.2, s=6, color=col)
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi],
            'k--', linewidth=1.2, label='Perfect prediction')
    ax.set_xlabel('Actual DO (mg/L)')
    ax.set_ylabel('Predicted DO (mg/L)')
    ax.set_title(f'{name}\nR² = {r2:.4f}')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('figures/03_scatter_actual_vs_predicted.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/03_scatter_actual_vs_predicted.png")

# --- Chart 4: Temperature vs DO correlation ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    'Correlation Confirmed by Supervisor 05/06/2026\n'
    'Temperature, Air Temperature and Sunshine vs DO',
    fontsize=12, fontweight='bold'
)

axes[0].scatter(df_predict['temperature'],
                df_predict['dissolved_oxygen_mgl'],
                alpha=0.1, s=4, color='#3498DB')
corr_temp = df_predict['temperature'].corr(
    df_predict['dissolved_oxygen_mgl']
)
axes[0].set_xlabel('Water Temperature (°C)')
axes[0].set_ylabel('Dissolved Oxygen (mg/L)')
axes[0].set_title(
    f'Water Temperature vs DO\n'
    f'Correlation: {corr_temp:.3f} '
    f'(expected: negative)'
)

axes[1].scatter(df_predict['sunshine_wm2'],
                df_predict['dissolved_oxygen_mgl'],
                alpha=0.1, s=4, color='#F39C12')
corr_sun = df_predict['sunshine_wm2'].corr(
    df_predict['dissolved_oxygen_mgl']
)
axes[1].set_xlabel('Sunshine (W/m²)')
axes[1].set_ylabel('Dissolved Oxygen (mg/L)')
axes[1].set_title(
    f'Sunshine vs DO\n'
    f'Correlation: {corr_sun:.3f}'
)

plt.tight_layout()
plt.savefig('figures/04_correlations.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figures/04_correlations.png")

# ─────────────────────────────────────────────────────────────
# STEP 9 — Print final summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL DONE — PREDICTION COMPLETE")
print("=" * 60)
print(f"\nBest model: {best_name}")
print(f"  MAE  = {best['MAE']:.4f} mg/L "
      f"(on average the model is off by this amount)")
print(f"  RMSE = {best['RMSE']:.4f} mg/L")
print(f"  R²   = {best['R2']:.4f} "
      f"(how well the model explains DO variation)")
print(f"\nCorrelations (your lecturer's suggestion):")
print(f"  Water temperature vs DO: {corr_temp:.3f}")
print(f"  Sunshine vs DO:          {corr_sun:.3f}")
print(f"\nFiles created:")
print(f"  data/processed/aquasensor_with_predictions.csv")
print(f"  data/processed/model_comparison.csv")
print(f"  models/linear_regression.pkl")
print(f"  models/random_forest.pkl")
print(f"  models/xgboost.pkl")
print(f"  figures/01_model_comparison.png")
print(f"  figures/02_actual_vs_predicted.png")
print(f"  figures/03_scatter_actual_vs_predicted.png")
print(f"  figures/04_correlations.png")
print(f"\nNext steps:")
print(f"  Week 4 — anomaly detection")
print(f"  Week 5 — polished Streamlit dashboard")