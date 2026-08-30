from pathlib import Path
import pandas as pd
import numpy as np
from src.config import PROCESSED_DATA_DIR

def build_master_dataset():
    print("=" * 65)
    print("BUILDING UNIFIED 2014–2016 MULTI-MISSION MASTER DATASET...")
    print("=" * 65)
    
    # 1. Load Processed Streams
    p_wind = PROCESSED_DATA_DIR / "wind_5min_master.parquet"
    p_goes = PROCESSED_DATA_DIR / "goes15_target_flux_5min.parquet"
    p_geom = PROCESSED_DATA_DIR / "geomagnetic_indices_5min.parquet"
    
    for p in [p_wind, p_goes, p_geom]:
        if not p.exists():
            raise FileNotFoundError(f"Missing processed input: {p}")
            
    df_wind = pd.read_parquet(p_wind)
    df_goes = pd.read_parquet(p_goes)
    df_geom = pd.read_parquet(p_geom)
    
    # Strip timezone awareness if present so indices align cleanly
    if df_goes.index.tz is not None:
        df_goes.index = df_goes.index.tz_localize(None)
    if df_wind.index.tz is not None:
        df_wind.index = df_wind.index.tz_localize(None)
    if df_geom.index.tz is not None:
        df_geom.index = df_geom.index.tz_localize(None)
        
    print(f"NASA WIND Shape       : {df_wind.shape}")
    print(f"NOAA GOES-15 Shape    : {df_goes.shape}")
    print(f"Geomagnetic Indices   : {df_geom.shape}")
    
    # 2. Synchronized Inner Join on Master 5-min Grid
    df_master = df_wind.join(df_goes, how='inner').join(df_geom, how='inner')
    
    # Forward-fill remaining minor telemetry dropouts (< 1 hr)
    df_master = df_master.interpolate(method='time', limit=12).bfill().ffill()
    
    # Target log-transform
    df_master['target_log_flux'] = np.log10(np.clip(df_master['goes_flux_e2_target'], 1e-4, None))
    
    out_path = PROCESSED_DATA_DIR / "master_space_weather_2014_2016_5min.parquet"
    df_master.to_parquet(out_path)
    
    print("\n" + "=" * 65)
    print(f"[✓] Unified Master Dataset Built Successfully!")
    print(f"Total Synchronized Rows : {len(df_master):,}")
    print(f"Total Feature Columns   : {df_master.shape[1]}")
    print(f"Export Path             : {out_path}")
    print("=" * 65)
    return df_master

if __name__ == "__main__":
    build_master_dataset()
