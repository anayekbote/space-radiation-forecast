import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

def evaluate_predictions(y_true, y_pred, label=""):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"[{label}] -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R² / PE: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

def train_xgboost_models():
    print("=" * 65)
    print("PHASE 19–23: MULTI-HORIZON XGBOOST TRAINING & BENCHMARKING")
    print("=" * 65)
    
    split_path = PROCESSED_DATA_DIR / "train_test_split.pkl"
    with open(split_path, "rb") as f:
        data = pickle.load(f)
        
    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    df_train = data['df_train']
    df_test = data['df_test']
    features = data['feature_cols']
    horizons = data['horizons']
    
    metrics_summary = {}
    
    for h in horizons:
        h_hours = h * 5 // 60
        target_col = f"target_lead_{h_hours}h"
        print(f"\n--- Training Horizon: +{h_hours} Hours ({target_col}) ---")
        
        y_train = df_train[target_col].values
        y_test = df_test[target_col].values
        
        # 1. Baseline Persistence Model Evaluation
        # Persistence assumes log_flux at t+H is log_flux at t
        y_persist = df_test['target_log_flux'].values
        print("Baseline (Persistence):")
        persist_metrics = evaluate_predictions(y_test, y_persist, label=f"Persistence +{h_hours}h")
        
        # 2. XGBoost Regressor Configuration
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            tree_method='hist'
        )
        
        print("Training XGBoost Regressor...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # 3. Model Predictions & Scoring
        y_pred = model.predict(X_test)
        print("XGBoost Forecast:")
        xgb_metrics = evaluate_predictions(y_test, y_pred, label=f"XGBoost +{h_hours}h")
        
        # 4. Save Model Artifact
        model_path = MODELS_DIR / f"xgb_model_{h_hours}h.json"
        model.save_model(str(model_path))
        print(f"[✓] Saved model artifact -> {model_path}")
        
        # 5. Extract Feature Importances
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        importance_path = MODELS_DIR / f"feature_importance_{h_hours}h.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"Top 3 Drivers: {', '.join(importance_df['feature'].head(3).tolist())}")
        
        metrics_summary[f"{h_hours}h"] = {
            "persistence": persist_metrics,
            "xgboost": xgb_metrics,
            "top_features": importance_df.head(5).to_dict(orient='records')
        }
        
    metrics_path = MODELS_DIR / "xgboost_benchmark_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
        
    print("\n" + "=" * 65)
    print(f"[✓] Multi-Horizon XGBoost Training Complete!")
    print(f"Metrics Report Saved: {metrics_path}")
    print("=" * 65)

if __name__ == "__main__":
    train_xgboost_models()
