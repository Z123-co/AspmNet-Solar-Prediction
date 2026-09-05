"""
AspmNet Model Layers and Components
Implements hierarchical attention, LSTM, and other custom layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InstanceNormalization(nn.Module):
    """
    Instance Normalization as described in Eq. (4) of the paper
    Reduces internal covariate shift and improves generalization
    """
    
    def __init__(self, eps=1e-7):
        super(InstanceNormalization, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        """
        Instance Normalization
        
        Args:
            x: Input tensor (batch_size, seq_length) or (batch_size, seq_length, channels)
            
        Returns:
            Normalized tensor
        """
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True) + self.eps
        normalized = (x - mean) / std
        
        return normalized


class CausalConvolution(nn.Module):
    """
    Causal Convolution Layer with Instance Normalization
    Prevents information leakage from future time steps
    
    Equation (3) from paper:
    Y_ca[t] = sum(W_ca[d] * I_ca[t - d * D_co]) for d=0 to k_ca
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(CausalConvolution, self).__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        
        self.instance_norm = InstanceNormalization()
        
    def forward(self, x):
        """
        Forward pass with causal masking
        
        Args:
            x: (batch_size, seq_length, in_channels)
            
        Returns:
            output: (batch_size, seq_length, out_channels)
        """
        # Transpose for Conv1d: (batch, channels, seq_length)
        x = x.transpose(1, 2)
        
        # Apply convolution
        output = self.conv(x)
        
        # Remove excess padding to match input length
        output = output[:, :, :x.size(2)]
        
        # Transpose back: (batch, seq_length, channels)
        output = output.transpose(1, 2)
        
        # Apply instance normalization
        output = self.instance_norm(output)
        
        return output


class TemporalAttention(nn.Module):
    """
    Temporal-level Attention Mechanism (Eq. 8)
    Evaluates the importance of each time step
    """
    
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Temporal Attention
        
        Args:
            x: (batch_size, seq_length, hidden_dim)
            
        Returns:
            attended: (batch_size, seq_length, hidden_dim)
        """
        batch_size, seq_length, _ = x.size()
        
        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq_length, hidden_dim)
        K = self.W_k(x)  # (batch, seq_length, hidden_dim)
        V = self.W_v(x)  # (batch, seq_length, hidden_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(self.hidden_dim)  # (batch, seq_length, seq_length)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_length, seq_length)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (batch, seq_length, hidden_dim)
        
        return attended


class FeatureAttention(nn.Module):
    """
    Feature-level Attention Mechanism (Eq. 10)
    Evaluates the importance of different features at different times
    """
    
    def __init__(self, hidden_dim):
        super(FeatureAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.W_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        """
        Feature Attention
        
        Args:
            x: (batch_size, seq_length, hidden_dim)
            
        Returns:
            attended: (batch_size, seq_length, hidden_dim)
        """
        # Compute feature-level attention
        e = torch.tanh(self.W_f(x))  # (batch, seq_length, hidden_dim)
        attention_weights = F.softmax(self.W_v(e), dim=-1)  # (batch, seq_length, hidden_dim)
        
        # Apply attention
        attended = x * attention_weights
        
        return attended


class HierarchicalAttention(nn.Module):
    """
    Hierarchical Attention combining temporal and feature-level attention (Eq. 12)
    """
    
    def __init__(self, hidden_dim):
        super(HierarchicalAttention, self).__init__()
        
        self.temporal_attention = TemporalAttention(hidden_dim)
        self.feature_attention = FeatureAttention(hidden_dim)
        
    def forward(self, x):
        """
        Apply hierarchical attention
        
        Args:
            x: (batch_size, seq_length, hidden_dim)
            
        Returns:
            output: (batch_size, seq_length, hidden_dim)
        """
        # Apply temporal attention
        temporal_out = self.temporal_attention(x)
        
        # Apply feature attention
        feature_out = self.feature_attention(temporal_out)
        
        return feature_out


class QuantileRegression(nn.Module):
    """
    Quantile Regression Layer for Interval Forecasting
    Produces point forecasts and prediction intervals (upper/medium/lower)
    
    Loss function (Eq. 25):
    L_qr = (1/P_p) * sum((1-phi) * |P[t] - P^[t]| if P[t] - P^[t] <= 0
                         phi * |P[t] - P^[t]| if P[t] - P^[t] > 0)
    """
    
    def __init__(self, input_size, output_size=3):
        """
        Args:
            input_size: Input feature dimension
            output_size: Number of quantiles (default: 3 for lower/medium/upper)
        """
        super(QuantileRegression, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Output 3 quantiles: 0.05 (lower), 0.50 (medium), 0.95 (upper)
        self.quantiles = torch.tensor([0.05, 0.50, 0.95])
        
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        """
        Generate quantile predictions
        
        Args:
            x: (batch_size, input_size)
            
        Returns:
            quantile_outputs: (batch_size, output_size)
        """
        return self.fc(x)
    
    def quantile_loss(self, y_pred, y_true):
        """
        Compute quantile regression loss
        
        Args:
            y_pred: Predicted quantiles (batch_size, 3)
            y_true: True values (batch_size,)
            
        Returns:
            loss: Scalar loss value
        """
        errors = y_true.unsqueeze(1) - y_pred  # (batch_size, 3)
        
        # For each quantile, compute asymmetric loss
        quantiles = self.quantiles.to(y_pred.device)
        
        loss = torch.zeros(1, device=y_pred.device)
        
        for i, q in enumerate(quantiles):
            condition = errors[:, i] >= 0
            loss += torch.where(
                condition,
                q * errors[:, i],
                (q - 1) * errors[:, i]
            ).mean()
        
        return loss / len(quantiles)
