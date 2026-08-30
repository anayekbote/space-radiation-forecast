import pickle
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from src.config import MODELS_DIR

class SpaceRadiationForecaster:
    def __init__(self):
        # 1. Load Scaler & Feature Definitions
        scaler_path = MODELS_DIR / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler checkpoint missing: {scaler_path}")
            
        with open(scaler_path, "rb") as f:
            meta = pickle.load(f)
        self.scaler = meta['scaler']
        self.feature_cols = meta['features']
        
        # 2. Load Pretrained Multi-Horizon XGBoost Models
        self.models = {}
        for h in [1, 6, 24]:
            m_path = MODELS_DIR / f"xgb_model_{h}h.json"
            if not m_path.exists():
                raise FileNotFoundError(f"Model checkpoint missing: {m_path}")
            m = xgb.XGBRegressor()
            m.load_model(str(m_path))
            self.models[f"{h}h"] = m
            
    def predict(self, feature_df: pd.DataFrame):
        """
        Validates column alignment, applies RobustScaler, and generates multi-horizon forecasts.
        """
        missing = [c for c in self.feature_cols if c not in feature_df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing[:5]}...")
            
        X_df = feature_df[self.feature_cols]
        X_scaled = self.scaler.transform(X_df)
        
        preds = {}
        for h_key, model in self.models.items():
            log_pred = model.predict(X_scaled)
            linear_flux = 10.0 ** log_pred
            preds[f"log_flux_lead_{h_key}"] = log_pred
            preds[f"flux_lead_{h_key}"] = linear_flux
            
        return pd.DataFrame(preds, index=feature_df.index)

def run_demo_inference():
    print("=" * 65)
    print("OPERATIONAL INFERENCE DEMO")
    print("=" * 65)
    
    forecaster = SpaceRadiationForecaster()
    split_path = MODELS_DIR.parent / "data" / "processed" / "train_test_split.pkl"
    
    if split_path.exists():
        with open(split_path, "rb") as f:
            data = pickle.load(f)
        df_sample = data['df_test'].iloc[-288:]
    else:
        # Self-contained synthetic validation sequence (24 hours at 5-min intervals)
        timestamps = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=288, freq='5min')
        synthetic_data = {col: np.random.randn(288) * 0.5 for col in forecaster.feature_cols}
        # Inject realistic solar wind baseline values
        synthetic_data['wind_sw_speed'] = np.full(288, 450.0)
        synthetic_data['wind_sw_density'] = np.full(288, 5.0)
        synthetic_data['dst_index'] = np.full(288, -15.0)
        synthetic_data['kp_index'] = np.full(288, 2.0)
        df_sample = pd.DataFrame(synthetic_data, index=timestamps)
        print("[*] Running standalone verification sample (24-hour lead cycle)...")
    
    forecasts = forecaster.predict(df_sample)
    
    print(f"Generated forecasts for {len(forecasts)} 5-minute intervals.")
    print("\n--- Latest Forecast Output ---")
    latest_ts = forecasts.index[-1]
    print(f"Timestamp: {latest_ts}")
    for h in [1, 6, 24]:
        log_val = forecasts[f'log_flux_lead_{h}h'].iloc[-1]
        lin_val = forecasts[f'flux_lead_{h}h'].iloc[-1]
        print(f"  + {h:02d}h Horizon -> Log Flux: {log_val:6.3f} | Estimated Flux: {lin_val:8.2f} cm⁻² s⁻¹ sr⁻¹")

if __name__ == "__main__":
    run_demo_inference()
