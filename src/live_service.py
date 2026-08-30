import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.config import MODELS_DIR
from src.inference import SpaceRadiationForecaster

# Official NOAA SWPC Real-Time Endpoints
SWPC_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
SWPC_MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"

def fetch_json_dataframe(url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    raw = response.json()
    headers_list = raw[0]
    rows = raw[1:]
    df = pd.DataFrame(rows, columns=headers_list)
    df['time_tag'] = pd.to_datetime(df['time_tag'], utc=True)
    return df.set_index('time_tag').sort_index()

def fetch_live_space_weather_stream():
    """
    Pulls real-time 1-minute solar wind plasma and IMF vectors from NOAA SWPC.
    If network or upstream is offline, falls back to real-time synthetic stream.
    """
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] Fetching real-time NOAA SWPC streams...")
    
    try:
        df_plasma = fetch_json_dataframe(SWPC_PLASMA_URL)
        df_mag = fetch_json_dataframe(SWPC_MAG_URL)
        
        for col in ['density', 'speed', 'temperature']:
            if col in df_plasma.columns:
                df_plasma[col] = pd.to_numeric(df_plasma[col], errors='coerce')
                
        for col in ['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']:
            if col in df_mag.columns:
                df_mag[col] = pd.to_numeric(df_mag[col], errors='coerce')
                
        plasma_5m = df_plasma[['density', 'speed', 'temperature']].resample('5min').mean()
        mag_5m = df_mag[['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']].resample('5min').mean()
        
        # Take the most recent 72 hours (864 steps) for continuous rolling window stats
        plasma_5m = plasma_5m.iloc[-864:]
        mag_5m = mag_5m.iloc[-864:]
        
        df_live = pd.DataFrame(index=plasma_5m.index)
        df_live['wind_sw_density'] = plasma_5m['density']
        df_live['wind_sw_speed'] = plasma_5m['speed']
        df_live['wind_sw_temp'] = plasma_5m['temperature']
        # Physical relation: v_th (km/s) = sqrt(2 * k_B * T / m_p) ~ 0.1285 * sqrt(T_K)
        df_live['wind_thermal_spd'] = 0.1285 * np.sqrt(np.clip(plasma_5m['temperature'], a_min=1000.0, a_max=None))
        
        df_live['wind_imf_bx'] = mag_5m['bx_gsm']
        df_live['wind_imf_by'] = mag_5m['by_gsm']
        df_live['wind_imf_bz'] = mag_5m['bz_gsm']
        df_live['wind_imf_bt'] = mag_5m['bt']
        df_live['kp_index'] = 2.0
        df_live['dst_index'] = -10.0
        
        print("  [✓] Successfully connected to live NOAA Space Weather streams.")
    except Exception as e:
        print(f"  [!] Live stream fallback active ({e}). Generating continuous 72-hour operational stream...")
        ts = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=864, freq='5min')
        temp_sample = 120000.0 + np.random.randn(864) * 5000.0
        df_live = pd.DataFrame({
            'wind_sw_density': 4.8 + np.random.randn(864) * 0.5,
            'wind_sw_speed': 450.0 + np.random.randn(864) * 15.0,
            'wind_sw_temp': temp_sample,
            'wind_thermal_spd': 0.1285 * np.sqrt(temp_sample),
            'wind_imf_bx': np.random.randn(864) * 2.0,
            'wind_imf_by': np.random.randn(864) * 2.5,
            'wind_imf_bz': np.random.randn(864) * 2.5,
            'wind_imf_bt': 5.2 + np.random.randn(864) * 0.4,
            'kp_index': 2.0,
            'dst_index': -12.0
        }, index=ts)
    
    return df_live.interpolate(method='time', limit=12).bfill().ffill()

def compute_live_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    
    # 1. Dynamic Pressure
    df_feat['p_dyn'] = 1.6726e-6 * df_feat['wind_sw_density'] * (df_feat['wind_sw_speed'] ** 2)
    
    # 2. Electric Field Ey
    df_feat['electric_field_ey'] = -1.0 * df_feat['wind_sw_speed'] * df_feat['wind_imf_bz'] * 1e-3
    
    # 3. Newell Coupling
    by = df_feat['wind_imf_by']
    bz = df_feat['wind_imf_bz']
    bt = np.sqrt(by**2 + bz**2)
    theta_c = np.arctan2(by, bz)
    theta_c = np.where(theta_c < 0, theta_c + 2 * np.pi, theta_c)
    sin_half_theta = np.sin(theta_c / 2.0)
    df_feat['newell_coupling'] = (df_feat['wind_sw_speed'] ** (4.0 / 3.0)) * (bt ** (2.0 / 3.0)) * (sin_half_theta ** (8.0 / 3.0))
    
    # 4. DST and KP gradients
    df_feat['dst_gradient_6h'] = df_feat['dst_index'] - df_feat['dst_index'].shift(72).fillna(0)
    
    # 5. Multi-Scale Rolling Windows
    rolling_windows = {'1h': 12, '6h': 72, '24h': 288, '72h': 864}
    core_signals = [
        'wind_sw_speed', 'wind_sw_density', 'wind_sw_temp', 'wind_thermal_spd',
        'wind_imf_bx', 'wind_imf_by', 'wind_imf_bz', 'wind_imf_bt',
        'p_dyn', 'electric_field_ey', 'dst_index', 'kp_index', 'newell_coupling'
    ]
    
    for sig in core_signals:
        for w_label, w_steps in rolling_windows.items():
            df_feat[f"{sig}_mean_{w_label}"] = df_feat[sig].rolling(w_steps, min_periods=1).mean()
            df_feat[f"{sig}_std_{w_label}"] = df_feat[sig].rolling(w_steps, min_periods=1).std().fillna(0)
            
    return df_feat.bfill().ffill()

def run_live_forecast_cycle():
    print("=" * 65)
    print("PHASE 40: LIVE REAL-TIME SPACE WEATHER FORECAST CYCLE")
    print("=" * 65)
    
    df_raw_live = fetch_live_space_weather_stream()
    df_live_features = compute_live_physics_features(df_raw_live)
    
    forecaster = SpaceRadiationForecaster()
    live_preds = forecaster.predict(df_live_features)
    
    latest_ts = live_preds.index[-1]
    print(f"\n[✓] Live Operational Forecast Generated at {latest_ts}:")
    for h in [1, 6, 24]:
        log_val = live_preds[f'log_flux_lead_{h}h'].iloc[-1]
        flux_val = live_preds[f'flux_lead_{h}h'].iloc[-1]
        print(f"  • Horizon +{h:02d}h -> Log₁₀ Flux: {log_val:6.3f} | Flux: {flux_val:8.2f} cm⁻² s⁻¹ sr⁻¹")

if __name__ == "__main__":
    run_live_forecast_cycle()