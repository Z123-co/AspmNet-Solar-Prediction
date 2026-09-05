"""
================================================================================
AspmNet: Complete Standalone Executable Program
Solar Photovoltaic Power Prediction Model
================================================================================

Usage:
    python run_aspmnet.py --data solar_data.csv --epochs 50 --batch_size 32

Features:
    ✓ End-to-end data processing
    ✓ Time series decomposition (Trend + Seasonal)
    ✓ Model training with early stopping
    ✓ Comprehensive evaluation metrics
    ✓ Result visualization and saving
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import uniform_filter1d
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import argparse
from pathlib import Path


# ===============================================================================
# CONFIGURATION
# ===============================================================================

class Config:
    """Configuration parameters"""
    
    # Data
    CSV_FILE = 'solar_data.csv'
    SEQ_LENGTH = 96  # 5 minutes * 96 = 480 minutes = 8 hours
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    TRAIN_SIZE = 0.70
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE = 10
    
    # Model
    SEASONAL_HIDDEN = 96
    TREND_HIDDEN = 48
    DECOMPOSE_WINDOW = 10
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Output
    OUTPUT_DIR = './results'
    MODEL_PATH = './results/aspmnet_best.pth'
    PLOT_PATH = './results/predictions.png'
    METRICS_PATH = './results/metrics.txt'


# ===============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ===============================================================================

class SolarDataLoader:
    """Load and preprocess solar PV data from CSV"""
    
    def __init__(self, file_path, test_size=0.15, val_size=0.15):
        self.file_path = file_path
        self.test_size = test_size
        self.val_size = val_size
        self.df = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_normalized = None
        
    def load_data(self):
        """Load CSV data"""
        print("="*80)
        print("STEP 1: LOADING DATA")
        print("="*80)
        print(f"📁 Loading from: {self.file_path}")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        self.df = pd.read_csv(self.file_path, parse_dates=['timestamp'])
        print(f"✓ Data shape: {self.df.shape}")
        print(f"✓ Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"✓ Total days: {(self.df['timestamp'].max() - self.df['timestamp'].min()).days}")
        
        return self.df
    
    def aggregate_power(self):
        """Aggregate all Active_Power columns to get total power"""
        print("\n" + "="*80)
        print("STEP 2: AGGREGATING ACTIVE POWER")
        print("="*80)
        
        # Find all Active_Power columns
        power_cols = [col for col in self.df.columns if 'Active_Power' in col]
        print(f"✓ Found {len(power_cols)} Active Power columns")
        
        # Sum all power columns to get total power
        self.df['Total_Active_Power'] = self.df[power_cols].fillna(0).sum(axis=1)
        self.df['Total_Active_Power_kW'] = self.df['Total_Active_Power'] / 1000
        
        power_min = self.df['Total_Active_Power_kW'].min()
        power_max = self.df['Total_Active_Power_kW'].max()
        power_mean = self.df['Total_Active_Power_kW'].mean()
        power_std = self.df['Total_Active_Power_kW'].std()
        
        print(f"✓ Power Statistics (kW):")
        print(f"  • Minimum: {power_min:.4f}")
        print(f"  • Maximum: {power_max:.4f}")
        print(f"  • Mean: {power_mean:.4f}")
        print(f"  • Std Dev: {power_std:.4f}")
        
        return self.df[['timestamp', 'Total_Active_Power_kW']]
    
    def handle_missing_values(self, method='forward_fill'):
        """Handle missing values in the data"""
        print("\n" + "="*80)
        print("STEP 3: HANDLING MISSING VALUES")
        print("="*80)
        
        null_count = self.df['Total_Active_Power_kW'].isnull().sum()
        print(f"✓ Missing values: {null_count}")
        
        if null_count > 0:
            if method == 'forward_fill':
                self.df['Total_Active_Power_kW'].fillna(method='ffill', inplace=True)
                self.df['Total_Active_Power_kW'].fillna(method='bfill', inplace=True)
            elif method == 'interpolate':
                self.df['Total_Active_Power_kW'].interpolate(method='linear', inplace=True)
            elif method == 'drop':
                self.df.dropna(subset=['Total_Active_Power_kW'], inplace=True)
            
            print(f"✓ Missing values handled using: {method}")
        else:
            print(f"✓ No missing values found")
        
        print(f"✓ Remaining data points: {len(self.df)}")
        return self.df
    
    def normalize_data(self):
        """Normalize data using MinMaxScaler"""
        print("\n" + "="*80)
        print("STEP 4: NORMALIZING DATA")
        print("="*80)
        
        power_data = self.df[['Total_Active_Power_kW']].values
        self.data_normalized = self.scaler.fit_transform(power_data)
        
        print(f"✓ Normalized range: [{self.data_normalized.min():.6f}, {self.data_normalized.max():.6f}]")
        print(f"✓ Mean: {self.data_normalized.mean():.6f}")
        print(f"✓ Std: {self.data_normalized.std():.6f}")
        
        return self.data_normalized
    
    def split_data(self):
        """Split data into train, validation, and test sets"""
        print("\n" + "="*80)
        print("STEP 5: SPLITTING DATA")
        print("="*80)
        
        n_samples = len(self.data_normalized)
        train_size = int(n_samples * (1 - self.val_size - self.test_size))
        val_size = int(n_samples * self.val_size)
        
        train_data = self.data_normalized[:train_size]
        val_data = self.data_normalized[train_size:train_size + val_size]
        test_data = self.data_normalized[train_size + val_size:]
        
        print(f"✓ Total samples: {n_samples:,}")
        print(f"✓ Train: {len(train_data):,} ({len(train_data)/n_samples*100:.1f}%)")
        print(f"✓ Validation: {len(val_data):,} ({len(val_data)/n_samples*100:.1f}%)")
        print(f"✓ Test: {len(test_data):,} ({len(test_data)/n_samples*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def create_sequences(self, data, seq_length):
        """Create sequences for time series prediction"""
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        
        return np.array(X), np.array(y)
    
    def get_processed_data(self, seq_length=96):
        """Complete pipeline"""
        self.load_data()
        self.aggregate_power()
        self.handle_missing_values()
        self.normalize_data()
        
        train_data, val_data, test_data = self.split_data()
        
        print("\n" + "="*80)
        print("STEP 6: CREATING SEQUENCES")
        print("="*80)
        
        X_train, y_train = self.create_sequences(train_data, seq_length)
        X_val, y_val = self.create_sequences(val_data, seq_length)
        X_test, y_test = self.create_sequences(test_data, seq_length)
        
        print(f"✓ Sequence length: {seq_length} steps")
        print(f"✓ X_train: {X_train.shape} | y_train: {y_train.shape}")
        print(f"✓ X_val: {X_val.shape} | y_val: {y_val.shape}")
        print(f"✓ X_test: {X_test.shape} | y_test: {y_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': self.scaler,
            'timestamps': self.df['timestamp'].values
        }


# ===============================================================================
# PART 2: TIME SERIES DECOMPOSITION
# ===============================================================================

class TimeSeriesDecomposition:
    """Decompose time series into Trend and Seasonal components"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.trend = None
        self.seasonal = None
        
    def extract_trend(self, data):
        """Extract trend component using moving average"""
        print(f"  Extracting trend (window={self.window_size})...")
        
        self.trend = uniform_filter1d(data.flatten(), size=self.window_size*2+1, mode='nearest')
        
        return self.trend.reshape(-1, 1)
    
    def extract_seasonal(self, data):
        """Extract seasonal component"""
        print("  Extracting seasonal component...")
        
        if self.trend is None:
            raise ValueError("Extract trend first!")
        
        self.seasonal = data.flatten() - self.trend
        
        return self.seasonal.reshape(-1, 1)
    
    def decompose(self, data):
        """Complete decomposition"""
        print("\n" + "="*80)
        print("TIME SERIES DECOMPOSITION")
        print("="*80)
        print(f"Input shape: {data.shape}")
        
        trend = self.extract_trend(data)
        seasonal = self.extract_seasonal(data)
        
        print(f"✓ Trend range: [{trend.min():.6f}, {trend.max():.6f}]")
        print(f"✓ Seasonal range: [{seasonal.min():.6f}, {seasonal.max():.6f}]")
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'original': data
        }


# ===============================================================================
# PART 3: MODEL LAYERS
# ===============================================================================

class InstanceNormalization(nn.Module):
    """Instance Normalization"""
    
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True) + self.eps
        return (x - mean) / std


class CausalConvolution(nn.Module):
    """Causal Convolution Layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                            padding=self.padding, dilation=dilation)
        self.instance_norm = InstanceNormalization()
        
    def forward(self, x):
        x = x.transpose(1, 2)
        output = self.conv(x)
        output = output[:, :, :x.size(2)]
        output = output.transpose(1, 2)
        return self.instance_norm(output)


class TemporalAttention(nn.Module):
    """Temporal-level Attention Mechanism"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(self.hidden_dim)
        weights = F.softmax(scores, dim=-1)
        
        return torch.matmul(weights, V)


class FeatureAttention(nn.Module):
    """Feature-level Attention Mechanism"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        e = torch.tanh(self.W_f(x))
        weights = F.softmax(self.W_v(e), dim=-1)
        return x * weights


class HierarchicalAttention(nn.Module):
    """Hierarchical Attention"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.temporal = TemporalAttention(hidden_dim)
        self.feature = FeatureAttention(hidden_dim)
        
    def forward(self, x):
        x = self.temporal(x)
        x = self.feature(x)
        return x


class SeasonalComponentPredictionUnit(nn.Module):
    """Seasonal Component Prediction Unit"""
    
    def __init__(self, input_size, hidden_size=96):
        super().__init__()
        
        self.conv1 = CausalConvolution(input_size, hidden_size, 3, 1)
        self.conv2 = CausalConvolution(hidden_size, hidden_size, 3, 2)
        self.attention = HierarchicalAttention(hidden_size)
        
        self.bilstm = nn.LSTM(hidden_size, hidden_size, 1, bidirectional=True, 
                             batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.attention(x)
        lstm_out, _ = self.bilstm(x)
        return self.fc(lstm_out)


class TrendComponentPredictionUnit(nn.Module):
    """Trend Component Prediction Unit"""
    
    def __init__(self, input_size, hidden_size=48):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


# ===============================================================================
# PART 4: COMPLETE ASPMNET MODEL
# ===============================================================================

class AspmNet(nn.Module):
    """Complete AspmNet Model"""
    
    def __init__(self, input_size=1, seasonal_hidden=96, trend_hidden=48):
        super().__init__()
        
        self.seasonal_unit = SeasonalComponentPredictionUnit(input_size, seasonal_hidden)
        self.trend_unit = TrendComponentPredictionUnit(input_size, trend_hidden)
        
    def forward(self, x_seasonal, x_trend):
        """Forward pass"""
        # Seasonal prediction
        seasonal_pred = self.seasonal_unit(x_seasonal)
        
        # Trend prediction
        x_trend_last = x_trend[:, -1, :]
        trend_pred = self.trend_unit(x_trend_last)
        
        # Integration
        trend_expanded = trend_pred.unsqueeze(1).expand_as(seasonal_pred)
        output = seasonal_pred + trend_expanded
        
        return output, seasonal_pred, trend_pred


# ===============================================================================
# PART 5: TRAINING AND EVALUATION
# ===============================================================================

class AspmNetTrainer:
    """Training pipeline for AspmNet"""
    
    def __init__(self, model, device='cpu', learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(self.device).float()
            y = y.to(self.device).float()
            
            # Simple decomposition
            seasonal = X - X.mean(dim=1, keepdim=True)
            trend = X.mean(dim=1, keepdim=True).expand_as(X)
            
            # Forward
            output, _, _ = self.model(seasonal, trend)
            loss = self.criterion(output, y)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(self.device).float()
                y = y.to(self.device).float()
                
                seasonal = X - X.mean(dim=1, keepdim=True)
                trend = X.mean(dim=1, keepdim=True).expand_as(X)
                
                output, _, _ = self.model(seasonal, trend)
                loss = self.criterion(output, y)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=50, patience=10):
        """Complete training loop"""
        print("\n" + "="*80)
        print("TRAINING ASPMNET MODEL")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Early stopping patience: {patience}\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            self.scheduler.step()
            
            print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"  ✓ Best validation loss: {val_loss:.6f}")
                
                # Create output directory if needed
                os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n✓ Early stopping at epoch {epoch + 1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(Config.MODEL_PATH))
        print("\n✓ Training completed!")
    
    def test(self, test_loader):
        """Test model"""
        print("\n" + "="*80)
        print("TESTING ASPMNET MODEL")
        print("="*80)
        
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device).float()
                y = y.to(self.device).float()
                
                seasonal = X - X.mean(dim=1, keepdim=True)
                trend = X.mean(dim=1, keepdim=True).expand_as(X)
                
                output, _, _ = self.model(seasonal, trend)
                
                predictions.append(output.cpu().numpy())
                targets.append(y.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        print(f"\n✓ Test Metrics:")
        print(f"  • MSE: {mse:.6f}")
        print(f"  • MAE: {mae:.6f}")
        print(f"  • RMSE: {rmse:.6f}")
        print(f"  • MAPE: {mape:.2f}%")
        
        return predictions, targets, {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape}


def plot_results(predictions, targets, scaler):
    """Plot results"""
    pred_denorm = scaler.inverse_transform(predictions.reshape(-1, 1))
    targ_denorm = scaler.inverse_transform(targets.reshape(-1, 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Full predictions
    axes[0, 0].plot(targ_denorm, label='Actual', linewidth=1.5, alpha=0.7)
    axes[0, 0].plot(pred_denorm, label='Predicted', linewidth=1.5, alpha=0.7)
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Power (kW)')
    axes[0, 0].set_title('Full Test Set Predictions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Last 500 steps
    axes[0, 1].plot(targ_denorm[-500:], label='Actual', linewidth=2)
    axes[0, 1].plot(pred_denorm[-500:], label='Predicted', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Power (kW)')
    axes[0, 1].set_title('Last 500 Time Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    errors = pred_denorm.flatten() - targ_denorm.flatten()
    axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.3f}')
    axes[1, 0].set_xlabel('Prediction Error (kW)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot
    axes[1, 1].scatter(targ_denorm, pred_denorm, alpha=0.5, s=10)
    axes[1, 1].plot([targ_denorm.min(), targ_denorm.max()], 
                    [targ_denorm.min(), targ_denorm.max()], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('Actual Power (kW)')
    axes[1, 1].set_ylabel('Predicted Power (kW)')
    axes[1, 1].set_title('Actual vs Predicted')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(Config.PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {Config.PLOT_PATH}")
    plt.show()


# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def main():
    """Main execution pipeline"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "AspmNet: Solar PV Power Prediction".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='AspmNet Solar Power Prediction')
    parser.add_argument('--data', type=str, default=Config.CSV_FILE, 
                       help='CSV data file path')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, 
                       help='Batch size')
    parser.add_argument('--seq_length', type=int, default=Config.SEQ_LENGTH, 
                       help='Sequence length')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, 
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print(f"\n✓ Configuration:")
    print(f"  • Data file: {args.data}")
    print(f"  • Epochs: {args.epochs}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Sequence length: {args.seq_length}")
    print(f"  • Learning rate: {args.lr}")
    print(f"  • Device: {Config.DEVICE}")
    
    # ===== Step 1: Load data =====
    loader = SolarDataLoader(args.data, test_size=Config.TEST_SIZE, val_size=Config.VAL_SIZE)
    data = loader.get_processed_data(seq_length=args.seq_length)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    scaler = data['scaler']
    
    # ===== Step 2: Decompose =====
    decomposer = TimeSeriesDecomposition(window_size=Config.DECOMPOSE_WINDOW)
    X_train_flat = X_train.flatten()
    decomposer.decompose(X_train_flat.reshape(-1, 1))
    
    # ===== Step 3: Create data loaders =====
    print("\n" + "="*80)
    print("CREATING DATA LOADERS")
    print("="*80)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # ===== Step 4: Initialize model =====
    print("\n" + "="*80)
    print("INITIALIZING ASPMNET MODEL")
    print("="*80)
    
    model = AspmNet(input_size=1, seasonal_hidden=Config.SEASONAL_HIDDEN, 
                   trend_hidden=Config.TREND_HIDDEN)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model initialized!")
    print(f"  • Total parameters: {total_params:,}")
    print(f"  • Trainable parameters: {trainable_params:,}")
    
    # ===== Step 5: Train =====
    trainer = AspmNetTrainer(model, device=Config.DEVICE, learning_rate=args.lr)
    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=Config.PATIENCE)
    
    # ===== Step 6: Test =====
    predictions, targets, metrics = trainer.test(test_loader)
    
    # ===== Step 7: Visualize =====
    plot_results(predictions, targets, scaler)
    
    # ===== Step 8: Save metrics =====
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    with open(Config.METRICS_PATH, 'w') as f:
        f.write("AspmNet Solar Power Prediction - Test Metrics\n")
        f.write("="*50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data: {args.data}\n")
        f.write(f"Sequence Length: {args.seq_length}\n")
        f.write(f"Epochs Trained: {len(trainer.train_losses)}\n\n")
        f.write("Metrics:\n")
        f.write(f"  MSE:  {metrics['mse']:.6f}\n")
        f.write(f"  MAE:  {metrics['mae']:.6f}\n")
        f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
    
    print(f"\n✓ Metrics saved to: {Config.METRICS_PATH}")
    
    # ===== Summary =====
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Training completed in {len(trainer.train_losses)} epochs")
    print(f"✓ Best validation loss: {min(trainer.val_losses):.6f}")
    print(f"✓ Test RMSE: {metrics['rmse']:.6f} kW")
    print(f"✓ Test MAPE: {metrics['mape']:.2f}%")
    print(f"✓ Model saved to: {Config.MODEL_PATH}")
    print(f"✓ Results saved to: {Config.OUTPUT_DIR}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
