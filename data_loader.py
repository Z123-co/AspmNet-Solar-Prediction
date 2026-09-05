"""
Data Loading and Preprocessing Module
Handles loading PV data from CSV and preprocessing for AspmNet model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SolarDataLoader:
    """Load and preprocess solar PV data from DKA Solar Centre"""
    
    def __init__(self, file_path, test_size=0.15, val_size=0.15):
        """
        Initialize data loader
        
        Args:
            file_path: Path to CSV file
            test_size: Test set ratio
            val_size: Validation set ratio
        """
        self.file_path = file_path
        self.test_size = test_size
        self.val_size = val_size
        self.df = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_normalized = None
        
    def load_data(self):
        """Load CSV data and basic preprocessing"""
        print("Loading data from CSV...")
        self.df = pd.read_csv(self.file_path, parse_dates=['timestamp'])
        print(f"Data shape: {self.df.shape}")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        
        return self.df
    
    def aggregate_power(self):
        """Aggregate all Active_Power columns to get total power output"""
        print("Aggregating Active Power columns...")
        
        # Find all Active_Power columns
        power_cols = [col for col in self.df.columns if 'Active_Power' in col]
        print(f"Found {len(power_cols)} Active Power columns")
        
        # Sum all power columns to get total power
        self.df['Total_Active_Power'] = self.df[power_cols].fillna(0).sum(axis=1)
        
        # Convert to kW (assuming data is in W)
        self.df['Total_Active_Power_kW'] = self.df['Total_Active_Power'] / 1000
        
        print(f"Power range (kW): {self.df['Total_Active_Power_kW'].min():.4f} to {self.df['Total_Active_Power_kW'].max():.4f}")
        
        return self.df[['timestamp', 'Total_Active_Power_kW']]
    
    def handle_missing_values(self, method='forward_fill'):
        """
        Handle missing values in the data
        
        Args:
            method: 'forward_fill', 'interpolate', or 'drop'
        """
        print(f"Handling missing values using {method}...")
        
        if method == 'forward_fill':
            self.df['Total_Active_Power_kW'] = self.df['Total_Active_Power_kW'].fillna(method='ffill')
            self.df['Total_Active_Power_kW'] = self.df['Total_Active_Power_kW'].fillna(method='bfill')
        elif method == 'interpolate':
            self.df['Total_Active_Power_kW'] = self.df['Total_Active_Power_kW'].interpolate(method='linear')
        elif method == 'drop':
            self.df = self.df.dropna(subset=['Total_Active_Power_kW'])
        
        print(f"Remaining data points: {len(self.df)}")
        return self.df
    
    def normalize_data(self):
        """Normalize power data using MinMaxScaler"""
        print("Normalizing data...")
        
        power_data = self.df[['Total_Active_Power_kW']].values
        self.data_normalized = self.scaler.fit_transform(power_data)
        
        print(f"Normalized data range: {self.data_normalized.min():.4f} to {self.data_normalized.max():.4f}")
        
        return self.data_normalized
    
    def split_data(self):
        """Split data into train, validation, and test sets"""
        print("Splitting data into train/val/test sets...")
        
        n_samples = len(self.data_normalized)
        
        # Calculate split indices
        train_size = int(n_samples * (1 - self.val_size - self.test_size))
        val_size = int(n_samples * self.val_size)
        
        train_data = self.data_normalized[:train_size]
        val_data = self.data_normalized[train_size:train_size + val_size]
        test_data = self.data_normalized[train_size + val_size:]
        
        print(f"Train samples: {len(train_data)} ({len(train_data)/n_samples*100:.1f}%)")
        print(f"Validation samples: {len(val_data)} ({len(val_data)/n_samples*100:.1f}%)")
        print(f"Test samples: {len(test_data)} ({len(test_data)/n_samples*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def create_sequences(self, data, seq_length):
        """
        Create sequences for time series prediction
        
        Args:
            data: Normalized data
            seq_length: Sequence length (input length)
            
        Returns:
            X: Input sequences
            y: Target values
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        
        return np.array(X), np.array(y)
    
    def get_processed_data(self, seq_length=96):
        """
        Complete pipeline: load -> aggregate -> normalize -> split -> create sequences
        
        Args:
            seq_length: Sequence length for model input
            
        Returns:
            Processed data for model training
        """
        # Load and preprocess
        self.load_data()
        self.aggregate_power()
        self.handle_missing_values()
        self.normalize_data()
        
        # Split data
        train_data, val_data, test_data = self.split_data()
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_data, seq_length)
        X_val, y_val = self.create_sequences(val_data, seq_length)
        X_test, y_test = self.create_sequences(test_data, seq_length)
        
        print(f"\nSequence shapes:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        
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
