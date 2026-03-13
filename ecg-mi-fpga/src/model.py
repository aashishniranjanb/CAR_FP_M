"""
FPGA-Optimized 1D-CNN Model Architecture
=========================================

Hardware-friendly 1D Convolutional Neural Network for ECG-based
Myocardial Infarction detection.

Architecture designed for minimal parameter count and optimal
mapping to Xilinx DSP48 slices and Block RAMs.

Key Design Decisions:
- Global Average Pooling (GAP) instead of Flatten to reduce parameters
- Channel progression: 12 → 32 → 64 → 128 (maps well to DSP tiling)
- ReLU activation (zero DSP cost in RTL - simple multiplexer)
- MaxPool1D for dimensionality reduction
- Total parameters: <200k (fits in on-chip BRAM when quantized to INT8)

Author: FPGA Hackathon Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLU6(nn.Module):
    """
    ReLU6 activation: clip output to [0, 6]
    Better for fixed-point quantization in hardware.
    """
    def forward(self, x):
        return F.relu6(x)


class FPGA_ECG_CNN(nn.Module):
    """
    FPGA-Optimized 1D-CNN for ECG MI Classification.
    
    Input:  (batch, 12 leads, 1000 samples) - 10 seconds at 100Hz
    Output: (batch, 2) - [NORM, MI] class logits
    
    Architecture:
    1. Conv1D Block 1: 12 → 32 channels, kernel=7, pool=2
    2. Conv1D Block 2: 32 → 64 channels, kernel=5, pool=2  
    3. Conv1D Block 3: 64 → 128 channels, kernel=3, pool=2
    4. Global Average Pooling (GAP)
    5. Fully Connected: 128 → 64 → 2
    """
    
    def __init__(self, num_classes=2):
        super(FPGA_ECG_CNN, self).__init__()
        
        # Convolutional Block 1: Feature Extraction
        # Input: 12 leads x 1000 samples
        # Output: 32 feature maps x 500 samples (after pool)
        self.conv1 = nn.Conv1d(
            in_channels=12, 
            out_channels=32, 
            kernel_size=7, 
            stride=1, 
            padding=3  # Same padding to maintain length
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 1000 → 500
        
        # Convolutional Block 2: Deep Feature Extraction
        # Input: 32 x 500
        # Output: 64 feature maps x 250 samples (after pool)
        self.conv2 = nn.Conv1d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=5, 
            stride=1, 
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 500 → 250
        
        # Convolutional Block 3: High-Level Features
        # Input: 64 x 250
        # Output: 128 feature maps x 125 samples (after pool)
        self.conv3 = nn.Conv1d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 250 → 125
        
        # Global Average Pooling
        # Reduces 128 x 125 to 128 (one value per channel)
        # This saves massive parameter count vs. flattening to FC
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully Connected Layers
        # Using GAP output: 128 features
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 12, 1000)
        
        Returns:
            Output logits (batch, 2)
        """
        # Block 1: Conv → BN → ReLU → Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)  # ReLU is hardware-friendly (just a comparator)
        x = self.pool1(x)
        
        # Block 2: Conv → BN → ReLU → Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3: Conv → BN → ReLU → Pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Global Average Pooling
        # x: (batch, 128, 125) → (batch, 128, 1)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 128)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Lightweight_ECG_CNN(nn.Module):
    """
    Ultra-lightweight version for resource-constrained FPGAs.
    Fewer channels for minimal BRAM usage.
    
    Architecture: 12 → 16 → 32 → 64 → GAP → 32 → 2
    Estimated parameters: ~30k (vs ~200k for full model)
    """
    
    def __init__(self, num_classes=2):
        super(Lightweight_ECG_CNN, self).__init__()
        
        # Lightweight Convolutional Blocks
        self.conv1 = nn.Conv1d(12, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        
        # GAP
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # FC
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_type='standard', num_classes=2):
    """
    Factory function to create ECG CNN models.
    
    Args:
        model_type: 'standard' or 'lightweight'
        num_classes: Number of output classes
    
    Returns:
        ECG CNN model
    """
    if model_type == 'standard':
        return FPGA_ECG_CNN(num_classes=num_classes)
    elif model_type == 'lightweight':
        return Lightweight_ECG_CNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test model creation and forward pass
    print("Testing ECG CNN Models...")
    
    # Standard model
    model = FPGA_ECG_CNN(num_classes=2)
    print(f"Standard Model Parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    x = torch.randn(4, 12, 1000)  # Batch of 4, 12 leads, 1000 samples
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Lightweight model
    model_light = Lightweight_ECG_CNN(num_classes=2)
    print(f"\nLightweight Model Parameters: {model_light.get_num_parameters():,}")
    
    y_light = model_light(x)
    print(f"Lightweight Output shape: {y_light.shape}")
    
    print("\nModel architecture test PASSED!")
