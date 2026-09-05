"""
================================================================================
AspmNet: Amplify Seasonality, Prioritize Meteorological
Solar Photovoltaic Power Prediction Model
================================================================================

Complete Implementation Based on:
"Amplify Seasonality, Prioritize Meteorological for Photovoltaic Power 
Prediction: A Deep Learning Approach"
Applied Energy 2025

Author: Based on paper by Niu et al.
Date: 2026
================================================================================

This is a complete, integrated implementation that includes:
1. Data Loading and Preprocessing
2. Time Series Decomposition (Trend + Seasonal)
3. Model Architecture with Attention Mechanisms
4. Training, Validation, and Testing
5. Evaluation Metrics and Visualization
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

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
        print(f"Loading data from: {self.file_path}")
        
        self.df = pd.read_csv(self.file_path, parse_dates=['timestamp'])
        print(f"✓ Data shape: {self.df.shape}")
        print(f"✓ Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"✓ Columns: {len(self.df.columns)}")
        
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
        
        print(f"✓ Power Statistics:")
        print(f"  - Min: {power_min:.4f} kW")
        print(f"  - Max: {power_max:.4f} kW")
        print(f"  - Mean: {power_mean:.4f} kW")
        print(f"  - Std: {self.df['Total_Active_Power_kW'].std():.4f} kW")
        
        return self.df[['timestamp', 'Total_Active_Power_kW']]
    
    def handle_missing_values(self, method='forward_fill'):
        """Handle missing values in the data"""
        print("\n" + "="*80)
        print("STEP 3: HANDLING MISSING VALUES")
        print("="*80)
        
        null_count = self.df['Total_Active_Power_kW'].isnull().sum()
        print(f"✓ Missing values detected: {null_count}")
        
        if method == 'forward_fill':
            self.df['Total_Active_Power_kW'] = self.df['Total_Active_Power_kW'].fillna(method='ffill')
            self.df['Total_Active_Power_kW'] = self.df['Total_Active_Power_kW'].fillna(method='bfill')
        elif method == 'interpolate':
            self.df['Total_Active_Power_kW'] = self.df['Total_Active_Power_kW'].interpolate(method='linear')
        elif method == 'drop':
            self.df = self.df.dropna(subset=['Total_Active_Power_kW'])
        
        print(f"✓ After handling - Remaining data points: {len(self.df)}")
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
        
        print(f"✓ Total samples: {n_samples}")
        print(f"✓ Train samples: {len(train_data)} ({len(train_data)/n_samples*100:.1f}%)")
        print(f"✓ Validation samples: {len(val_data)} ({len(val_data)/n_samples*100:.1f}%)")
        print(f"✓ Test samples: {len(test_data)} ({len(test_data)/n_samples*100:.1f}%)")
        
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
        
        print(f"✓ Input sequence length: {seq_length}")
        print(f"✓ X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
        print(f"✓ X_val shape: {X_val.shape} | y_val shape: {y_val.shape}")
        print(f"✓ X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")
        
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
        """Extract trend component using moving average (Eq. 1)"""
        print(f"Extracting trend with window size {self.window_size}...")
        
        self.trend = uniform_filter1d(data.flatten(), size=self.window_size*2+1, mode='nearest')
        
        print(f"  Trend range: [{self.trend.min():.6f}, {self.trend.max():.6f}]")
        
        return self.trend.reshape(-1, 1)
    
    def extract_seasonal(self, data):
        """Extract seasonal component by subtracting trend (Eq. 2)"""
        print("Extracting seasonal component...")
        
        if self.trend is None:
            raise ValueError("Extract trend first!")
        
        self.seasonal = data.flatten() - self.trend
        
        print(f"  Seasonal range: [{self.seasonal.min():.6f}, {self.seasonal.max():.6f}]")
        print(f"  Seasonal std: {self.seasonal.std():.6f}")
        
        return self.seasonal.reshape(-1, 1)
    
    def decompose(self, data):
        """Complete decomposition"""
        print("\n" + "="*80)
        print("TIME SERIES DECOMPOSITION (Trend + Seasonal)")
        print("="*80)
        print(f"Input data shape: {data.shape}")
        print(f"Input range: [{data.min():.6f}, {data.max():.6f}]")
        
        trend = self.extract_trend(data)
        seasonal = self.extract_seasonal(data)
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'original': data
        }


# ===============================================================================
# PART 3: MODEL LAYERS AND COMPONENTS
# ===============================================================================

class InstanceNormalization(nn.Module):
    """Instance Normalization (Eq. 4)"""
    
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True) + self.eps
        return (x - mean) / std


class CausalConvolution(nn.Module):
    """Causal Convolution Layer (Eq. 3)"""
    
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
    """Temporal-level Attention Mechanism (Eq. 8)"""
    
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
    """Feature-level Attention Mechanism (Eq. 10)"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        e = torch.tanh(self.W_f(x))
        weights = F.softmax(self.W_v(e), dim=-1)
        return x * weights


class HierarchicalAttention(nn.Module):
    """Hierarchical Attention (Eq. 12)"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.temporal = TemporalAttention(hidden_dim)
        self.feature = FeatureAttention(hidden_dim)
        
    def forward(self, x):
        x = self.temporal(x)
        x = self.feature(x)
        return x


class QuantileRegression(nn.Module):
    """Quantile Regression for Interval Forecasting (Eq. 25)"""
    
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 3)  # 3 quantiles: 0.05, 0.50, 0.95
        self.quantiles = torch.tensor([0.05, 0.50, 0.95])
        
    def forward(self, x):
        return self.fc(x)
    
    def quantile_loss(self, y_pred, y_true):
        errors = y_true.unsqueeze(1) - y_pred
        quantiles = self.quantiles.to(y_pred.device)
        
        loss = torch.zeros(1, device=y_pred.device)
        for i, q in enumerate(quantiles):
            condition = errors[:, i] >= 0
            loss += torch.where(condition, q * errors[:, i], 
                              (q - 1) * errors[:, i]).mean()
        
        return loss / len(quantiles)


class SeasonalComponentPredictionUnit(nn.Module):
    """Seasonal Component Prediction Unit (Section 3.3)"""
    
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
    """Trend Component Prediction Unit with Quantile Regression (Section 3.4)"""
    
    def __init__(self, input_size, hidden_size=48):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
        self.quantile_reg = QuantileRegression(hidden_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # MLP with hidden layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        # Quantile regression
        return self.quantile_reg(x)


# ===============================================================================
# PART 4: COMPLETE ASPMNET MODEL
# ===============================================================================

class AspmNet(nn.Module):
    """
    Complete AspmNet Model
    
    Architecture:
    Step 1: Decomposition (Trend + Seasonal)
    Step 2: Amplify Seasonality, Prioritize Meteorological
    Step 3: Seasonal-Trend Prediction Unit
    Step 4: Integration Stage (Matrix Summation)
    """
    
    def __init__(self, input_size=1, seasonal_hidden=96, trend_hidden=48):
        super().__init__()
        
        self.seasonal_unit = SeasonalComponentPredictionUnit(input_size, seasonal_hidden)
        self.trend_unit = TrendComponentPredictionUnit(input_size, trend_hidden)
        
    def forward(self, x_seasonal, x_trend):
        """
        Forward pass
        
        Args:
            x_seasonal: Seasonal component (batch, seq_length, 1)
            x_trend: Trend component (batch, seq_length, 1)
            
        Returns:
            output: Final prediction (batch, seq_length, 1)
        """
        # Seasonal component prediction
        seasonal_pred = self.seasonal_unit(x_seasonal)  # (batch, seq_length, 1)
        
        # Trend component prediction (use last values)
        x_trend_last = x_trend[:, -1, :]  # (batch, 1)
        trend_quantiles = self.trend_unit(x_trend_last)  # (batch, 3)
        trend_pred = trend_quantiles[:, 1:2]  # Use median quantile (batch, 1)
        
        # Integration: combine seasonal and trend
        # Expand trend prediction to match seasonal sequence length
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
            
            # Decompose (simple version - using average)
            seasonal = X - X.mean(dim=1, keepdim=True)
            trend = X.mean(dim=1, keepdim=True).expand_as(X)
            
            # Forward pass
            output, _, _ = self.model(seasonal, trend)
            loss = self.criterion(output, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.6f}")
        
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
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            self.scheduler.step()
            
            print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"  ✓ Best validation loss improved to {val_loss:.6f}")
                torch.save(self.model.state_dict(), 'aspmnet_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n✓ Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('aspmnet_best.pth'))
        print("\n✓ Training completed!")
    
    def test(self, test_loader):
        """Test model and compute metrics"""
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
        
        # Compute metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        print(f"\n✓ Test Metrics:")
        print(f"  - MSE: {mse:.6f}")
        print(f"  - MAE: {mae:.6f}")
        print(f"  - RMSE: {rmse:.6f}")
        
        return predictions, targets, {'mse': mse, 'mae': mae, 'rmse': rmse}


def plot_results(predictions, targets, scaler, title="AspmNet Predictions"):
    """Plot prediction results"""
    # Denormalize predictions
    predictions_denorm = scaler.inverse_transform(predictions.reshape(-1, 1))
    targets_denorm = scaler.inverse_transform(targets.reshape(-1, 1))
    
    plt.figure(figsize=(15, 6))
    
    # Plot predictions vs targets
    plt.subplot(1, 2, 1)
    plt.plot(targets_denorm[-500:], label='Actual', linewidth=2)
    plt.plot(predictions_denorm[-500:], label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Power (kW)')
    plt.title(f'{title} - Last 500 Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot error distribution
    plt.subplot(1, 2, 2)
    errors = predictions_denorm.flatten() - targets_denorm.flatten()
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error (kW)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('aspmnet_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Results plot saved as 'aspmnet_results.png'")
    plt.show()


# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def main():
    """Complete pipeline execution"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "AspmNet: Solar Photovoltaic Power Prediction Model".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Configuration
    CSV_FILE = 'solar_data.csv'  # Change to your data file path
    SEQ_LENGTH = 96
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # ===== STEP 1: Load and preprocess data =====
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    loader = SolarDataLoader(CSV_FILE, test_size=0.15, val_size=0.15)
    data = loader.get_processed_data(seq_length=SEQ_LENGTH)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    scaler = data['scaler']
    
    # ===== STEP 2: Time series decomposition =====
    print("\n" + "="*80)
    print("TIME SERIES DECOMPOSITION")
    print("="*80)
    
    decomposer = TimeSeriesDecomposition(window_size=10)
    
    # Decompose training data
    X_train_flat = X_train.flatten()
    decomp = decomposer.decompose(X_train_flat.reshape(-1, 1))
    
    print("✓ Decomposition completed!")
    print(f"  - Trend extracted: {decomp['trend'].shape}")
    print(f"  - Seasonal extracted: {decomp['seasonal'].shape}")
    
    # ===== STEP 3: Create data loaders =====
    print("\n" + "="*80)
    print("CREATING DATA LOADERS")
    print("="*80)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # ===== STEP 4: Initialize model =====
    print("\n" + "="*80)
    print("INITIALIZING ASPMNET MODEL")
    print("="*80)
    
    model = AspmNet(input_size=1, seasonal_hidden=96, trend_hidden=48)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully!")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # ===== STEP 5: Train model =====
    trainer = AspmNetTrainer(model, device=device, learning_rate=LEARNING_RATE)
    trainer.train(train_loader, val_loader, epochs=EPOCHS, patience=10)
    
    # ===== STEP 6: Test model =====
    predictions, targets, metrics = trainer.test(test_loader)
    
    # ===== STEP 7: Visualization =====
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_results(predictions, targets, scaler, title="AspmNet Solar Power Prediction")
    
    # ===== STEP 8: Summary =====
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Model trained for {len(trainer.train_losses)} epochs")
    print(f"✓ Best validation loss: {min(trainer.val_losses):.6f}")
    print(f"✓ Test MSE: {metrics['mse']:.6f}")
    print(f"✓ Test MAE: {metrics['mae']:.6f}")
    print(f"✓ Test RMSE: {metrics['rmse']:.6f}")
    print(f"✓ Best model saved as: aspmnet_best.pth")
    print("✓ Results plot saved as: aspmnet_results.png")
    print("\n✓ Training completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
