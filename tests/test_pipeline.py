import pytest
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.inference import SpaceRadiationForecaster
from src.models.train_lstm import SpaceWeatherAttentionGRU

def test_processed_tables_exist():
    required_files = [
        "wind_5min_master.parquet",
        "goes15_target_flux_5min.parquet",
        "geomagnetic_indices_5min.parquet",
        "isro_gsat19_grasp_2018_5min.parquet",
        "master_space_weather_2014_2016_5min.parquet",
        "features_space_weather_2014_2016_5min.parquet",
        "train_test_split.pkl"
    ]
    for fname in required_files:
        p = PROCESSED_DATA_DIR / fname
        assert p.exists(), f"Missing processed asset: {fname}"

def test_model_artifacts_exist():
    artifacts = [
        "scaler.pkl",
        "xgb_model_1h.json",
        "xgb_model_6h.json",
        "xgb_model_24h.json",
        "deep_gru_attention.pt"
    ]
    for fname in artifacts:
        p = MODELS_DIR / fname
        assert p.exists(), f"Missing model checkpoint: {fname}"

def test_inference_engine_output():
    with open(PROCESSED_DATA_DIR / "train_test_split.pkl", "rb") as f:
        data = pickle.load(f)
    df_sample = data['df_test'].iloc[:24]
    
    forecaster = SpaceRadiationForecaster()
    preds = forecaster.predict(df_sample)
    
    assert len(preds) == len(df_sample)
    assert "log_flux_lead_1h" in preds.columns
    assert "flux_lead_24h" in preds.columns
    assert not preds.isnull().values.any()
    assert (preds["flux_lead_1h"] > 0).all()

def test_deep_gru_forward_pass():
    checkpoint = torch.load(MODELS_DIR / "deep_gru_attention.pt", map_location='cpu')
    input_dim = checkpoint['input_dim']
    
    model = SpaceWeatherAttentionGRU(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dummy_input = torch.randn(4, 72, input_dim) # Batch=4, SeqLen=72
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (4, 3)
