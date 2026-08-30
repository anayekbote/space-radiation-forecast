from pathlib import Path
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def ingest_geomagnetic_indices():
    print("=" * 65)
    print("INGESTING GEOMAGNETIC ACTIVITY INDICES (KP & DST)...")
    print("=" * 65)
    
    omni_files = sorted(glob.glob(str(RAW_DATA_DIR / "geomagnetic_indices" / "omni2_hourly_*.dat")))
    if not omni_files:
        raise FileNotFoundError(f"No OMNI2 hourly files found in {RAW_DATA_DIR / 'geomagnetic_indices'}")
        
    records = []
    for fpath in tqdm(omni_files, desc="Parsing OMNI2 hourly data"):
        with open(fpath, 'r') as f:
            for line in f:
                tokens = line.split()
                if len(tokens) < 41:
                    continue
                try:
                    year = int(tokens[0])
                    doy = int(tokens[1])
                    hour = int(tokens[2])
                    
                    # Convert Year + DOY + Hour to Datetime
                    dt = datetime(year, 1, 1, hour=hour) + timedelta(days=doy - 1)
                    
                    # Kp is stored as integer Kp*10 (0..90). 99 is fill.
                    kp_raw = float(tokens[38])
                    kp_val = kp_raw / 10.0 if kp_raw <= 90 else np.nan
                    
                    # Dst index in nT. 99999 is fill.
                    dst_raw = float(tokens[40])
                    dst_val = dst_raw if abs(dst_raw) < 1000 else np.nan
                    
                    records.append({
                        'datetime': dt,
                        'kp_index': kp_val,
                        'dst_index': dst_val
                    })
                except Exception:
                    continue
                    
    if not records:
        raise ValueError("Failed to parse geomagnetic indices records.")
        
    df_geom = pd.DataFrame(records).set_index('datetime').sort_index()
    df_geom = df_geom[~df_geom.index.duplicated(keep='first')]
    
    # Resample from 1-hour to continuous 5-minute grid using time interpolation
    df_geom_5m = df_geom.resample('5min').interpolate(method='time', limit=24)
    
    out_path = PROCESSED_DATA_DIR / "geomagnetic_indices_5min.parquet"
    df_geom_5m.to_parquet(out_path)
    print(f"\n[✓] Geomagnetic Indices Ingestion Complete: {len(df_geom_5m):,} timesteps -> {out_path}")
    return df_geom_5m

if __name__ == "__main__":
    ingest_geomagnetic_indices()
