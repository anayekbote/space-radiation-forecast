import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from src.config import MODELS_DIR, PLOTS_DIR

def generate_diagnostic_plots():
    print("=" * 65)
    print("PHASE 35–38: GENERATING PUBLICATION PLOTS")
    print("=" * 65)
    
    # Load test split
    with open(MODELS_DIR.parent / "data" / "processed" / "train_test_split.pkl", "rb") as f:
        data = pickle.load(f)
        
    X_test_scaled = data['X_test_scaled']
    df_test = data['df_test']
    
    # Load 24h Model
    model_24h = xgb.XGBRegressor()
    model_24h.load_model(str(MODELS_DIR / "xgb_model_24h.json"))
    
    y_true = df_test['target_lead_24h'].values
    y_pred = model_24h.predict(X_test_scaled)
    
    # 1. 24-Hour Forecast vs Ground Truth Time Series (Sample 2-Week Window)
    sample_slice = slice(1000, 1000 + 2016) # ~1 week of 5-min intervals
    ts_time = df_test.index[sample_slice]
    
    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(ts_time, y_true[sample_slice], label='GOES-15 Measured Flux (log10)', color='black', alpha=0.8, lw=1.2)
    plt.plot(ts_time, y_pred[sample_slice], label='24h Ahead XGBoost Forecast', color='#e74c3c', linestyle='--', lw=1.5)
    plt.title('Multi-Mission Relativistic Electron Flux Forecasting (+24h Horizon)', fontsize=12, fontweight='bold')
    plt.ylabel('log₁₀ [Flux (cm⁻² s⁻¹ sr⁻¹)]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plot_ts_path = PLOTS_DIR / "forecast_timeseries_24h.png"
    plt.savefig(plot_ts_path)
    plt.close()
    print(f"[✓] Time Series Forecast Saved -> {plot_ts_path}")
    
    # 2. Feature Importance Pareto Chart
    feat_df = pd.read_csv(MODELS_DIR / "feature_importance_24h.csv").head(10)
    plt.figure(figsize=(9, 4.5), dpi=300)
    plt.barh(feat_df['feature'][::-1], feat_df['importance'][::-1], color='#2980b9')
    plt.title('Top 10 Physical Drivers of 24-Hour Electron Acceleration', fontsize=11, fontweight='bold')
    plt.xlabel('Normalized Gini Importance')
    plt.tight_layout()
    plot_imp_path = PLOTS_DIR / "feature_importance_top10.png"
    plt.savefig(plot_imp_path)
    plt.close()
    print(f"[✓] Feature Importance Plot Saved -> {plot_imp_path}")

if __name__ == "__main__":
    generate_diagnostic_plots()
