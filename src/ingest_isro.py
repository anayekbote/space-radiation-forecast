from pathlib import Path
import glob
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def ingest_isro_gsat19(year=2018):
    print("=" * 65)
    print("INGESTING ISRO GSAT-19 GRASP L2 DATA (2018 BENCHMARK)...")
    print("=" * 65)
    
    zip_files = sorted(glob.glob(str(RAW_DATA_DIR / "isro_gsat19_grasp" / "**" / "*.zip"), recursive=True))
    if not zip_files:
        raise FileNotFoundError(f"No GSAT-19 ZIP archives found in {RAW_DATA_DIR / 'isro_gsat19_grasp'}")
        
    records = []
    base_year_dt = datetime(year, 1, 1)
    
    for zpath in tqdm(zip_files, desc="Parsing GSAT-19 ZIPs"):
        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                for fname in zf.namelist():
                    if fname.endswith('.txt'):
                        with zf.open(fname) as f:
                            # Skip the header line
                            header = f.readline()
                            for line in f:
                                line_str = line.decode('utf-8', errors='ignore').strip()
                                if not line_str:
                                    continue
                                parts = line_str.split()
                                if len(parts) < 2:
                                    continue
                                try:
                                    doy_frac = float(parts[0])
                                    e_flux = float(parts[1])
                                    p_flux = float(parts[2]) if len(parts) > 2 else np.nan
                                    
                                    # Fractional DOY conversion to timestamp
                                    # Day 1.0 is Jan 1 00:00:00
                                    days_offset = doy_frac - 1.0
                                    dt = base_year_dt + timedelta(days=days_offset)
                                    
                                    # Physical gating
                                    e_clean = e_flux if (0 <= e_flux < 1e7) else np.nan
                                    p_clean = p_flux if (0 <= p_flux < 1e7) else np.nan
                                    
                                    records.append({
                                        'datetime': dt,
                                        'isro_electron_flux': e_clean,
                                        'isro_proton_flux': p_clean
                                    })
                                except Exception:
                                    continue
        except Exception:
            continue
            
    if not records:
        raise ValueError("Failed to extract valid data from GSAT-19 archives.")
        
    df_isro = pd.DataFrame(records).set_index('datetime').sort_index()
    df_isro = df_isro[~df_isro.index.duplicated(keep='first')]
    
    # 5-minute continuous resampling and interpolation
    df_isro_5m = df_isro.resample('5min').mean().interpolate(method='time', limit=12)
    
    # Compute log10 flux for direct multi-mission transfer benchmark
    df_isro_5m['log_isro_electron_flux'] = np.log10(np.clip(df_isro_5m['isro_electron_flux'], 1e-4, None))
    
    out_path = PROCESSED_DATA_DIR / "isro_gsat19_grasp_2018_5min.parquet"
    df_isro_5m.to_parquet(out_path)
    print(f"\n[✓] ISRO GSAT-19 Ingestion Complete: {len(df_isro_5m):,} timesteps -> {out_path}")
    return df_isro_5m

if __name__ == "__main__":
    ingest_isro_gsat19()
