"""
Time Series Decomposition Module
Implements STL decomposition to separate Trend and Seasonal components
Following the paper's Section 3.2 methodology
"""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDecomposition:
    """
    Decompose time series into Trend and Seasonal components
    Using moving average approach as described in the paper
    """
    
    def __init__(self, window_size=10):
        """
        Initialize decomposition module
        
        Args:
            window_size: Time window size for trend extraction (Wmu in paper)
                        Larger window captures long-term trends
        """
        self.window_size = window_size
        self.trend = None
        self.seasonal = None
        
    def extract_trend(self, data):
        """
        Extract trend component using moving average
        
        Equation (1) from paper:
        T_ph[t] = (sum of F_ph[t+d] for d=-Wmu to Wmu) / (2*Wmu + 1)
        
        Args:
            data: Input time series (normalized)
            
        Returns:
            Trend component
        """
        print(f"Extracting trend component with window size {self.window_size}...")
        
        # Use uniform filter for moving average
        # Handle edge cases with edge padding
        self.trend = uniform_filter1d(data.flatten(), size=self.window_size*2+1, mode='nearest')
        
        print(f"Trend shape: {self.trend.shape}")
        print(f"Trend range: {self.trend.min():.6f} to {self.trend.max():.6f}")
        
        return self.trend.reshape(-1, 1)
    
    def extract_seasonal(self, data):
        """
        Extract seasonal component by subtracting trend from original
        
        Equation: S_ph = F_ph - T_ph
        
        Args:
            data: Input time series (normalized)
            
        Returns:
            Seasonal component
        """
        print("Extracting seasonal component...")
        
        if self.trend is None:
            raise ValueError("Trend must be extracted first. Call extract_trend().")
        
        self.seasonal = data.flatten() - self.trend
        
        print(f"Seasonal shape: {self.seasonal.shape}")
        print(f"Seasonal range: {self.seasonal.min():.6f} to {self.seasonal.max():.6f}")
        print(f"Seasonal std: {self.seasonal.std():.6f}")
        
        return self.seasonal.reshape(-1, 1)
    
    def decompose(self, data):
        """
        Complete decomposition: extract both trend and seasonal components
        
        Args:
            data: Input normalized time series
            
        Returns:
            Dictionary with trend and seasonal components
        """
        print("\n" + "="*60)
        print("TIME SERIES DECOMPOSITION")
        print("="*60)
        print(f"Input data shape: {data.shape}")
        print(f"Input data range: {data.min():.6f} to {data.max():.6f}")
        
        # Extract components
        trend = self.extract_trend(data)
        seasonal = self.extract_seasonal(data)
        
        print("\n" + "="*60)
        print("Decomposition completed successfully!")
        print("="*60)
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'original': data
        }
    
    def reconstruct(self, trend, seasonal):
        """
        Reconstruct original signal from components
        
        Equation: F_ph = T_ph + S_ph
        
        Args:
            trend: Trend component
            seasonal: Seasonal component
            
        Returns:
            Reconstructed signal
        """
        reconstructed = trend + seasonal
        
        # Verify reconstruction quality
        if hasattr(self, 'original'):
            reconstruction_error = np.mean(np.abs(reconstructed - self.original))
            print(f"Reconstruction error: {reconstruction_error:.6f}")
        
        return reconstructed
    
    def analyze_components(self, trend, seasonal):
        """
        Analyze and print statistics of decomposed components
        """
        print("\n" + "="*60)
        print("COMPONENT ANALYSIS")
        print("="*60)
        
        print("\nTREND Component:")
        print(f"  Min: {trend.min():.6f}")
        print(f"  Max: {trend.max():.6f}")
        print(f"  Mean: {trend.mean():.6f}")
        print(f"  Std: {trend.std():.6f}")
        
        print("\nSEASONAL Component:")
        print(f"  Min: {seasonal.min():.6f}")
        print(f"  Max: {seasonal.max():.6f}")
        print(f"  Mean: {seasonal.mean():.6f}")
        print(f"  Std: {seasonal.std():.6f}")
        
        # Calculate explained variance
        total_var = np.var(self.original)
        trend_var = np.var(trend)
        seasonal_var = np.var(seasonal)
        
        print(f"\nVariance Decomposition:")
        print(f"  Total: {total_var:.6f}")
        print(f"  Trend: {trend_var:.6f} ({trend_var/total_var*100:.2f}%)")
        print(f"  Seasonal: {seasonal_var:.6f} ({seasonal_var/total_var*100:.2f}%)")
