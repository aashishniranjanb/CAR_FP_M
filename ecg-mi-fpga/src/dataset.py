"""
PTB-XL Dataset Loader and Preprocessing
======================================

Implements ECG signal loading, bandpass filtering, and PyTorch Dataset
for the PTB-XL myocardial infarction classification task.

Key Features:
- 100Hz data selection (reduces BRAM requirements 5x vs 500Hz)
- Binary classification: MI (1) vs NORM (0)
- 0.5Hz-40Hz bandpass filter for clinical-grade preprocessing
- Stratified fold splits (Folds 1-8: Train, 9: Val, 10: Test)
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import wfdb
from scipy.signal import butter, filtfilt
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
PTBXL_CSV = os.path.join(DATA_DIR, 'ptbxl_database.csv')
SCP_CSV = os.path.join(DATA_DIR, 'scp_statements.csv')

# Use PhysioNet streaming - records loaded on-demand
# This avoids downloading 2GB+ of data upfront
PTBXL_RECORDS_DIR = 'records100'  # wfdb will stream from physionet
PTBXL_PHYSIONET_DIR = 'ptb-xl'  # PhysioNet database name

# ECG parameters
SAMPLING_RATE = 100  # Hz - using 100Hz to reduce FPGA resource usage
SECONDS = 10  # 10-second ECG recordings
LEADS = 12  # Standard 12-lead ECG
SAMPLES_PER_LEAD = SAMPLING_RATE * SECONDS  # 1000 samples per lead

# Filter parameters (clinical-grade preprocessing)
LOWCUT = 0.5   # Hz - removes baseline wander
HIGHCUT = 40.0  # Hz - removes 50/60Hz powerline interference


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Design a Butterworth bandpass filter.
    
    Args:
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
    
    Returns:
        b, a: Filter coefficients
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(signal, fs=SAMPLING_RATE, order=4):
    """
    Apply zero-phase bandpass filter using filtfilt.
    Zero-phase filtering eliminates temporal distortion of QRS complexes.
    
    Args:
        signal: Input ECG signal (1D or 2D array)
        fs: Sampling frequency
        order: Filter order
    
    Returns:
        Filtered signal
    """
    b, a = butter_bandpass(LOWCUT, HIGHCUT, fs, order=order)
    
    # Handle both 1D and 2D signals
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        # Apply filter along the time axis (last dimension)
        filtered = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            filtered[i] = filtfilt(b, a, signal[i])
        return filtered


def normalize_signal(signal):
    """
    Normalize signal to zero mean and unit variance per lead.
    
    Args:
        signal: ECG signal array
    
    Returns:
        Normalized signal
    """
    # Normalize each lead independently
    normalized = np.zeros_like(signal, dtype=np.float32)
    for i in range(signal.shape[0]):
        lead = signal[i]
        mean = np.mean(lead)
        std = np.std(lead)
        if std > 0:
            normalized[i] = (lead - mean) / std
        else:
            normalized[i] = lead - mean
    return normalized


def load_ptbxl_record(record_name, data_dir=DATA_DIR):
    """
    Load a single PTB-XL record.
    Checks for local file first, then falls back to PhysioNet streaming.
    
    Args:
        record_name: PTB-XL record identifier (e.g., 'records100/00000/00001_lr')
        data_dir: Base directory containing records
    
    Returns:
        ECG signal as numpy array (12 leads x 1000 samples)
    """
    try:
        # Construct local path
        local_path = os.path.join(data_dir, record_name)
        
        if os.path.exists(local_path + '.hea'):
            # Load from local file
            record = wfdb.rdrecord(record_name=local_path)
        else:
            # Fallback to PhysioNet streaming
            print(f"Streaming {record_name} from PhysioNet...")
            record = wfdb.rdrecord(record_name=record_name, pn_dir=PTBXL_PHYSIONET_DIR)
            
        signal = record.p_signal  # Shape: (1000, 12)
        # Transpose to (12 leads x 1000 samples)
        return signal.T.astype(np.float32)
    except Exception as e:
        print(f"Error loading {record_name}: {e}")
        return None


def parse_scp_codes(scp_str):
    """
    Parse the scp_codes column string to extract diagnostic classes.
    
    Args:
        scp_str: String representation of dictionary (e.g., "{'NORM': 0, 'MI': 1}")
    
    Returns:
        Dictionary of diagnostic codes
    """
    if pd.isna(scp_str):
        return {}
    try:
        return ast.literal_eval(scp_str)
    except:
        return {}


def create_binary_labels(df, scp_df):
    """
    Create binary labels: MI (1) vs NORM (0).
    Discards ambiguous or multi-class records.
    
    Args:
        df: PTB-XL database dataframe
        scp_df: SCP statements dataframe
    
    Returns:
        Dataframe with binary labels
    """
    # Get MI-related diagnostic classes from SCP statements
    # MI codes: IMI, ASMI, ILMI, AMI, ALMI, INJAS, LMI, INJAL, IPLMI, IPMI, INJIN, INJLA, PMI, INJIL
    # These are the indices in scp_statements.csv with diagnostic_class == 'MI'
    mi_codes = ['IMI', 'ASMI', 'ILMI', 'AMI', 'ALMI', 'INJAS', 'LMI', 'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI', 'INJIL']
    
    # Create binary labels
    labels = []
    for idx, row in df.iterrows():
        scp = parse_scp_codes(row.get('scp_codes', '{}'))
        
        # Check for MI
        has_mi = any(code in scp.keys() for code in mi_codes)
        has_norm = 'NORM' in scp.keys()
        
        if has_mi and not has_norm:
            labels.append(1)  # MI
        elif has_norm and not has_mi:
            labels.append(0)  # NORM
        else:
            labels.append(-1)  # Discard ambiguous
    
    df['label'] = labels
    # Keep only MI and NORM samples
    df = df[df['label'] >= 0].reset_index(drop=True)
    
    return df


class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL ECG classification.
    
    Returns:
        Tuple of (input_tensor, label) where:
        - input_tensor: (12 leads x 1000 samples) normalized ECG
        - label: 0 (NORM) or 1 (MI)
    """
    
    def __init__(self, df, data_dir=PTBXL_RECORDS_DIR, transform=None):
        """
        Initialize PTB-XL Dataset.
        
        Args:
            df: DataFrame with 'filename_lr' and 'label' columns
            data_dir: Directory containing PTB-XL records
            transform: Optional transform to apply
        """
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Load and preprocess a single ECG sample.
        
        Returns:
            Tuple (ecg_tensor, label)
        """
        row = self.df.iloc[idx]
        record_name = row['filename_lr']  # Use filename_lr for 100Hz data
        label = row['label']
        
        # Load ECG signal
        signal = load_ptbxl_record(record_name, self.data_dir)
        
        if signal is None:
            # Return zeros if loading fails (fallback)
            signal = np.zeros((LEADS, SAMPLES_PER_LEAD), dtype=np.float32)
        
        # Apply bandpass filter (0.5Hz - 40Hz)
        signal = apply_bandpass_filter(signal, fs=SAMPLING_RATE)
        
        # Normalize each lead
        signal = normalize_signal(signal)
        
        # Convert to tensor
        ecg_tensor = torch.from_numpy(signal)
        
        if self.transform:
            ecg_tensor = self.transform(ecg_tensor)
        
        return ecg_tensor, torch.tensor(label, dtype=torch.long)


def get_data_loaders(batch_size=32, num_workers=2, data_dir=DATA_DIR):
    """
    Create train, validation, and test DataLoaders using official PTB-XL folds.
    
    Args:
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        data_dir: Path to data directory
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load database CSV
    ptbxl_csv = os.path.join(data_dir, 'ptbxl_database.csv')
    scp_csv = os.path.join(data_dir, 'scp_statements.csv')
    
    if not os.path.exists(ptbxl_csv):
        print(f"PTB-XL database not found at {ptbxl_csv}")
        print("Please run: python -m src.download_data")
        return None, None, None
    
    df = pd.read_csv(ptbxl_csv)
    scp_df = pd.read_csv(scp_csv)
    
    # Create binary labels
    df = create_binary_labels(df, scp_df)
    
    # Stratified splits based on PTB-XL fold column
    # Fold 1-8: Training, Fold 9: Validation, Fold 10: Test
    train_df = df[df['strat_fold'] <= 8].reset_index(drop=True)
    val_df = df[df['strat_fold'] == 9].reset_index(drop=True)
    test_df = df[df['strat_fold'] == 10].reset_index(drop=True)
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_df)} samples ({train_df['label'].value_counts().to_dict()})")
    print(f"  Val:   {len(val_df)} samples ({val_df['label'].value_counts().to_dict()})")
    print(f"  Test:  {len(test_df)} samples ({test_df['label'].value_counts().to_dict()})")
    
    # Create datasets
    # Note: Records are streamed from PhysioNet on-demand
    train_dataset = PTBXLDataset(train_df, data_dir=PTBXL_RECORDS_DIR)
    val_dataset = PTBXLDataset(val_df, data_dir=PTBXL_RECORDS_DIR)
    test_dataset = PTBXLDataset(test_df, data_dir=PTBXL_RECORDS_DIR)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def compute_class_weights(df):
    """
    Compute class weights for handling imbalanced dataset.
    MI cases are rarer than Normal cases.
    
    Args:
        df: DataFrame with 'label' column
    
    Returns:
        Tensor of class weights [weight_norm, weight_mi]
    """
    label_counts = df['label'].value_counts().sort_index()
    total = len(df)
    n_norm = label_counts.get(0, 1)
    n_mi = label_counts.get(1, 1)
    
    # Inverse frequency weighting
    weight_norm = total / (2 * n_norm)
    weight_mi = total / (2 * n_mi)
    
    return torch.tensor([weight_norm, weight_mi], dtype=torch.float32)


if __name__ == '__main__':
    # Test dataset loading
    print("Testing PTB-XL Dataset loader...")
    
    # Check if data exists
    if os.path.exists(PTBXL_CSV):
        train_loader, val_loader, test_loader = get_data_loaders(batch_size=8)
        
        # Test one batch
        for batch_x, batch_y in train_loader:
            print(f"Batch shape: {batch_x.shape}, Labels: {batch_y.shape}")
            print(f"Input range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
            break
        
        print("Dataset loading test PASSED!")
    else:
        print(f"PTB-XL data not found at {DATA_DIR}")
        print("Please run download_data.py first to download the dataset.")
