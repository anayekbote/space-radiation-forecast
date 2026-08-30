from pathlib import Path
import glob
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def ingest_goes15_eps():
    print("=" * 65)
    print("INGESTING NOAA GOES-15 EPEAD RELATIVISTIC ELECTRONS (>2 MeV)...")
    print("=" * 65)
    
    nc_files = sorted(glob.glob(str(RAW_DATA_DIR / "goes15_eps" / "**" / "*.nc"), recursive=True))
    if not nc_files:
        raise FileNotFoundError(f"No GOES-15 NetCDF files found in {RAW_DATA_DIR / 'goes15_eps'}")
        
    records = []
    for fpath in tqdm(nc_files, desc="Parsing GOES NetCDFs"):
        try:
            nc = Dataset(fpath, 'r')
            time_ms = nc.variables['time_tag'][:]
            a2e_flux = nc.variables['A2E_FLUX'][:]
            a2w_flux = nc.variables['A2W_FLUX'][:]
            
            # Epoch to UTC Datetime
            dt = pd.to_datetime(time_ms, unit='ms', utc=True)
            
            # Physical quality gating
            a2e_clean = np.where((a2e_flux > 0) & (a2e_flux < 1e7), a2e_flux, np.nan)
            a2w_clean = np.where((a2w_flux > 0) & (a2w_flux < 1e7), a2w_flux, np.nan)
            
            # Isotropic >2 MeV Target Flux (particles / (cm² s sr))
            target_flux = np.nanmean(np.vstack([a2e_clean, a2w_clean]), axis=0)
            
            df_sub = pd.DataFrame({
                'goes_flux_e2_target': target_flux
            }, index=pd.DatetimeIndex(dt))
            
            records.append(df_sub)
            nc.close()
        except Exception:
            continue
            
    if not records:
        raise ValueError("Failed to extract records from GOES NetCDFs.")
        
    df_goes = pd.concat(records).sort_index()
    df_goes = df_goes[~df_goes.index.duplicated(keep='first')]
    
    # 5-minute Master Resampling
    df_goes_5m = df_goes.resample('5min').mean().interpolate(method='time', limit=12)
    
    out_path = PROCESSED_DATA_DIR / "goes15_target_flux_5min.parquet"
    df_goes_5m.to_parquet(out_path)
    print(f"\n[✓] GOES-15 Ingestion Complete: {len(df_goes_5m):,} timesteps -> {out_path}")
    return df_goes_5m

if __name__ == "__main__":
    ingest_goes15_eps()
