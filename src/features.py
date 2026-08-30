from pathlib import Path
import pandas as pd
import numpy as np
from src.config import PROCESSED_DATA_DIR

def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Computing magnetospheric physics parameters...")
    df_feat = df.copy()
    
    # 1. Solar Wind Dynamic Pressure P_dyn (nPa)
    # Np in cm^-3, V_sw in km/s -> P_dyn ≈ 1.6726e-6 * Np * V^2
    df_feat['p_dyn'] = 1.6726e-6 * df_feat['wind_sw_density'] * (df_feat['wind_sw_speed'] ** 2)
    
    # 2. Interplanetary Electric Field Ey (mV/m)
    # Ey = -V * Bz * 10^-3
    df_feat['electric_field_ey'] = -1.0 * df_feat['wind_sw_speed'] * df_feat['wind_imf_bz'] * 1e-3
    
    # 3. Clock Angle theta_c and Newell Coupling Function
    # theta_c = arctan2(By, Bz)
    by = df_feat['wind_imf_by']
    bz = df_feat['wind_imf_bz']
    bt = np.sqrt(by**2 + bz**2)
    theta_c = np.arctan2(by, bz)
    theta_c = np.where(theta_c < 0, theta_c + 2 * np.pi, theta_c)
    
    # Newell rate of magnetic flux merging at magnetopause
    sin_half_theta = np.sin(theta_c / 2.0)
    df_feat['newell_coupling'] = (df_feat['wind_sw_speed'] ** (4.0 / 3.0)) * (bt ** (2.0 / 3.0)) * (sin_half_theta ** (8.0 / 3.0))
    
    # 4. Geomagnetic Storm Severity Flags
    df_feat['dst_gradient_6h'] = df_feat['dst_index'] - df_feat['dst_index'].shift(72) # 72 * 5min = 6 hours
    
    # 5. Multi-Scale Cumulative & Rolling History (1h, 6h, 24h, 72h)
    # 1h = 12 steps, 6h = 72 steps, 24h = 288 steps, 72h = 864 steps
    rolling_windows = {'1h': 12, '6h': 72, '24h': 288, '72h': 864}
    
    core_signals = ['wind_sw_speed', 'p_dyn', 'electric_field_ey', 'dst_index', 'kp_index', 'newell_coupling']
    
    print("Building multi-scale rolling temporal features...")
    for sig in core_signals:
        for w_label, w_steps in rolling_windows.items():
            df_feat[f"{sig}_mean_{w_label}"] = df_feat[sig].rolling(w_steps, min_periods=1).mean()
            df_feat[f"{sig}_std_{w_label}"] = df_feat[sig].rolling(w_steps, min_periods=1).std().fillna(0)
            
    # Forward/backward fill any remaining window boundary artifacts
    df_feat = df_feat.bfill().ffill()
    return df_feat

def engineer_all_features():
    print("=" * 65)
    print("PHASE 11–15: FEATURE EXTRACTION PIPELINE")
    print("=" * 65)
    
    master_path = PROCESSED_DATA_DIR / "master_space_weather_2014_2016_5min.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"Master parquet not found at {master_path}")
        
    df_master = pd.read_parquet(master_path)
    df_engineered = compute_physics_features(df_master)
    
    out_path = PROCESSED_DATA_DIR / "features_space_weather_2014_2016_5min.parquet"
    df_engineered.to_parquet(out_path)
    
    print("\n" + "=" * 65)
    print(f"[✓] Feature Engineering Complete!")
    print(f"Total Rows Processed : {len(df_engineered):,}")
    print(f"Total Feature Count  : {df_engineered.shape[1]}")
    print(f"Saved To             : {out_path}")
    print("=" * 65)
    return df_engineered

if __name__ == "__main__":
    engineer_all_features()
