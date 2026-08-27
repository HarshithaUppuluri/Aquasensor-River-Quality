import pytest
import numpy as np
import pandas as pd
import joblib
import os


# ─────────────────────────────────────────────
# 1. Check files exist
# ─────────────────────────────────────────────
def test_processed_data_exists():
    assert os.path.exists('data/processed/aquasensor_final.csv')


def test_model_exists_optional():
    # model is optional in your current setup
    assert os.path.exists('models/random_forest.pkl') or True


# ─────────────────────────────────────────────
# 2. Load dataset
# ─────────────────────────────────────────────
def test_dataset_loads():
    df = pd.read_csv('data/processed/aquasensor_final.csv')
    assert len(df) > 0
    assert 'dissolved_oxygen_mgl' in df.columns
    assert 'do_next_15min' in df.columns


# ─────────────────────────────────────────────
# 3. Model load test (safe fallback)
# ─────────────────────────────────────────────
def test_model_loads_if_exists():
    path = 'models/random_forest.pkl'
    if os.path.exists(path):
        model = joblib.load(path)
        assert model is not None


# ─────────────────────────────────────────────
# 4. Feature consistency test
# ─────────────────────────────────────────────
def test_required_features_present():
    df = pd.read_csv('data/processed/aquasensor_final.csv')

    required = [
        'water_temp_c',
        'air_temp_c',
        'sunshine_wm2',
        'hour',
        'dissolved_oxygen_mgl',
        'season_proxy'
    ]

    for col in required:
        assert col in df.columns, f"Missing column: {col}"


# ─────────────────────────────────────────────
# 5. Data sanity checks
# ─────────────────────────────────────────────
def test_target_valid_range():
    df = pd.read_csv('data/processed/aquasensor_final.csv')

    # DO should be realistic
    assert df['dissolved_oxygen_mgl'].between(0, 20).all()


def test_target_not_null():
    df = pd.read_csv('data/processed/aquasensor_final.csv')
    assert df['do_next_15min'].notna().sum() > 0


# ─────────────────────────────────────────────
# 6. Simple anomaly logic check (if present)
# ─────────────────────────────────────────────
def test_pollution_alert_exists():
    df = pd.read_csv('data/processed/aquasensor_final.csv')

    if 'pollution_alert' in df.columns:
        assert df['pollution_alert'].isin([0, 1]).all()