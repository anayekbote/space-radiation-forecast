from pathlib import Path
import glob
import pandas as pd
import numpy as np
import cdflib
from tqdm import tqdm
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def ingest_wind_swe():
    print("=" * 65)
    print("INGESTING NASA WIND SWE (SOLAR WIND PLASMA)...")
    print("=" * 65)
    
    swe_files = sorted(glob.glob(str(RAW_DATA_DIR / "wind_swe" / "**" / "*.cdf"), recursive=True))
    if not swe_files:
        raise FileNotFoundError(f"No SWE CDFs found in {RAW_DATA_DIR / 'wind_swe'}")
    
    records = []
    for fpath in tqdm(swe_files, desc="Parsing SWE CDFs"):
        try:
            cdf = cdflib.CDF(fpath)
            epoch = cdf.varget("Epoch")
            # V_GSE_p has shape (N, 3): [Speed, Latitude, Longitude]
            v_polar = cdf.varget("V_GSE_p")
            np_den = cdf.varget("Np")
            v_th = cdf.varget("THERMAL_SPD")
            
            dt = cdflib.cdfepoch.to_datetime(epoch)
            
            # Bulk speed is column 0 of V_GSE_p
            if v_polar.ndim > 1:
                v_bulk = v_polar[:, 0]
            else:
                v_bulk = v_polar
                
            v_bulk = np.where((v_bulk > 0) & (v_bulk < 2500), v_bulk, np.nan)
            np_den = np.where((np_den > 0) & (np_den < 300), np_den, np.nan)
            v_th = np.where((v_th > 0) & (v_th < 500), v_th, np.nan)
            
            df_sub = pd.DataFrame({
                'wind_sw_speed': v_bulk,
                'wind_sw_density': np_den,
                'wind_thermal_spd': v_th
            }, index=pd.DatetimeIndex(dt))
            records.append(df_sub)
        except Exception:
            continue
            
    if not records:
        raise ValueError("No SWE records could be extracted from CDF files.")
        
    df_swe = pd.concat(records).sort_index()
    df_swe = df_swe[~df_swe.index.duplicated(keep='first')]
    df_swe_5m = df_swe.resample('5min').mean().interpolate(method='time', limit=6)
    print(f"[✓] SWE Ingestion Complete: {len(df_swe_5m):,} timesteps")
    return df_swe_5m

def ingest_wind_mfi():
    print("\n" + "=" * 65)
    print("INGESTING NASA WIND MFI (IMF MAGNETIC FIELD VECTORS)...")
    print("=" * 65)
    
    mfi_files = sorted(glob.glob(str(RAW_DATA_DIR / "wind_mfi" / "**" / "*.cdf"), recursive=True))
    if not mfi_files:
        raise FileNotFoundError(f"No MFI CDFs found in {RAW_DATA_DIR / 'wind_mfi'}")
        
    records = []
    for fpath in tqdm(mfi_files, desc="Parsing MFI CDFs"):
        try:
            cdf = cdflib.CDF(fpath)
            epoch = cdf.varget("Epoch")
            b_gsm = cdf.varget("BGSM")
            b_mag = cdf.varget("BF1")
            
            dt = cdflib.cdfepoch.to_datetime(epoch)
            
            bx = np.where(np.abs(b_gsm[:, 0]) < 200, b_gsm[:, 0], np.nan)
            by = np.where(np.abs(b_gsm[:, 1]) < 200, b_gsm[:, 1], np.nan)
            bz = np.where(np.abs(b_gsm[:, 2]) < 200, b_gsm[:, 2], np.nan)
            bt = np.where((b_mag > 0) & (b_mag < 250), b_mag, np.nan)
            
            df_sub = pd.DataFrame({
                'wind_imf_bx': bx,
                'wind_imf_by': by,
                'wind_imf_bz': bz,
                'wind_imf_bt': bt
            }, index=pd.DatetimeIndex(dt))
            records.append(df_sub)
        except Exception:
            continue
            
    if not records:
        raise ValueError("No MFI records could be extracted from CDF files.")
        
    df_mfi = pd.concat(records).sort_index()
    df_mfi = df_mfi[~df_mfi.index.duplicated(keep='first')]
    df_mfi_5m = df_mfi.resample('5min').mean().interpolate(method='time', limit=6)
    print(f"[✓] MFI Ingestion Complete: {len(df_mfi_5m):,} timesteps")
    return df_mfi_5m

if __name__ == "__main__":
    df_swe = ingest_wind_swe()
    df_mfi = ingest_wind_mfi()
    
    df_wind = df_swe.join(df_mfi, how='inner')
    out_path = PROCESSED_DATA_DIR / "wind_5min_master.parquet"
    df_wind.to_parquet(out_path)
    print(f"\n[✓] NASA WIND Master Table Saved: {len(df_wind):,} timesteps -> {out_path}")
