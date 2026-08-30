import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

def prepare_dataset(df: pd.DataFrame, horizons=[12, 72, 288]):
    print("Preparing multi-horizon forecast targets...")
    df_data = df.copy()
    
    # 1. Generate Multi-Horizon Lead Targets (1h, 6h, 24h)
    for h in horizons:
        h_label = f"target_lead_{h*5//60}h"
        df_data[h_label] = df_data['target_log_flux'].shift(-h)
        
    # Drop trailing rows where future targets are NaN due to lead shift
    df_data = df_data.dropna(subset=[f"target_lead_{h*5//60}h" for h in horizons])
    
    # 2. Define Features vs Target
    target_cols = ['goes_flux_e2_target', 'target_log_flux'] + [f"target_lead_{h*5//60}h" for h in horizons]
    feature_cols = [col for col in df_data.columns if col not in target_cols]
    
    # 3. Chronological Train/Test Split (2014–2015 Train, 2016 Test)
    split_date = '2016-01-01'
    df_train = df_data[df_data.index < split_date]
    df_test = df_data[df_data.index >= split_date]
    
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    
    # 4. Robust Scaling (Fitted ONLY on Training Set)
    print("Fitting RobustScaler on training distribution...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({'scaler': scaler, 'features': feature_cols}, f)
        
    # Package split summaries
    split_info = {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'df_train': df_train,
        'df_test': df_test,
        'feature_cols': feature_cols,
        'horizons': horizons
    }
    
    out_split = PROCESSED_DATA_DIR / "train_test_split.pkl"
    with open(out_split, "wb") as f:
        pickle.dump(split_info, f)
        
    print(f"\n[✓] Train Set: {len(X_train):,} samples (2014–2015)")
    print(f"[✓] Test Set : {len(X_test):,} samples (2016)")
    print(f"[✓] Feature Count: {len(feature_cols)}")
    print(f"[✓] Scaler & Datasets Saved to {scaler_path} and {out_split}")
    return split_info

def run_preprocessing():
    print("=" * 65)
    print("PHASE 16–18: TEMPORAL SPLIT & NORMALIZATION")
    print("=" * 65)
    
    feat_path = PROCESSED_DATA_DIR / "features_space_weather_2014_2016_5min.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Engineered features file missing: {feat_path}")
        
    df = pd.read_parquet(feat_path)
    prepare_dataset(df)

if __name__ == "__main__":
    run_preprocessing()
