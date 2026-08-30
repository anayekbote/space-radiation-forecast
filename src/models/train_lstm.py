import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

# 1. Sliding Window Temporal Dataset
class SpaceWeatherSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 72, step: int = 6):
        self.seq_len = seq_len
        self.samples = []
        # Strided window extraction for memory & training efficiency
        for i in range(0, len(X) - seq_len, step):
            self.samples.append((
                torch.tensor(X[i:i + seq_len], dtype=torch.float32),
                torch.tensor(y[i + seq_len], dtype=torch.float32)
            ))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

# 2. Attention-Augmented GRU Network
class SpaceWeatherAttentionGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        gru_out, _ = self.gru(x) # [Batch, Seq_Len, Hidden*2]
        weights = self.attention(gru_out) # [Batch, Seq_Len, 1]
        context = torch.sum(weights * gru_out, dim=1) # [Batch, Hidden*2]
        out = self.head(context) # [Batch, Output_Dim]
        return out

def train_sequence_model():
    print("=" * 65)
    print("PHASE 24–28: ATTENTION-AUGMENTED SEQUENCE MODEL (PYTORCH)")
    print("=" * 65)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing training on compute device: {device}")
    
    split_path = PROCESSED_DATA_DIR / "train_test_split.pkl"
    with open(split_path, "rb") as f:
        data = pickle.load(f)
        
    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    df_train = data['df_train']
    df_test = data['df_test']
    
    # Target vectors: [+1h, +6h, +24h]
    target_cols = ['target_lead_1h', 'target_lead_6h', 'target_lead_24h']
    y_train = df_train[target_cols].values
    y_test = df_test[target_cols].values
    
    seq_len = 72 # 6 hours of continuous 5-min history
    print(f"Constructing sliding window sequences (Seq Len: {seq_len} steps)...")
    train_dataset = SpaceWeatherSequenceDataset(X_train, y_train, seq_len=seq_len, step=4)
    test_dataset = SpaceWeatherSequenceDataset(X_test, y_test, seq_len=seq_len, step=4)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    input_dim = X_train.shape[1]
    model = SpaceWeatherAttentionGRU(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    
    criterion = nn.SmoothL1Loss() # Huber loss for robustness against space storms
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    epochs = 12
    best_val_loss = float('inf')
    best_model_path = MODELS_DIR / "deep_gru_attention.pt"
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        y_preds, y_trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * len(batch_x)
                y_preds.append(pred.cpu().numpy())
                y_trues.append(batch_y.cpu().numpy())
                
        val_loss /= len(test_dataset)
        scheduler.step(val_loss)
        
        y_preds = np.vstack(y_preds)
        y_trues = np.vstack(y_trues)
        
        rmse_1h = np.sqrt(mean_squared_error(y_trues[:, 0], y_preds[:, 0]))
        rmse_6h = np.sqrt(mean_squared_error(y_trues[:, 1], y_preds[:, 1]))
        rmse_24h = np.sqrt(mean_squared_error(y_trues[:, 2], y_preds[:, 2]))
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | RMSE (1h/6h/24h): [{rmse_1h:.3f}, {rmse_6h:.3f}, {rmse_24h:.3f}]")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'seq_len': seq_len,
                'horizons': target_cols
            }, best_model_path)
            
    print(f"\n[✓] Best Deep Attention-GRU Checkpoint: {best_model_path}")

if __name__ == "__main__":
    train_sequence_model()
