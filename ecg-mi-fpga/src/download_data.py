"""
PTB-XL Dataset Download Script
==============================

Downloads and prepares the PTB-XL ECG dataset for training.

Dataset Info:
- Source: https://physionet.org/content/ptb-xl/1.0.3/
- 21,837 clinical 12-lead ECG recordings
- 10 seconds duration at 100Hz or 500Hz
- We use 100Hz (records100/) to reduce FPGA resource requirements

Author: FPGA Hackathon Team
"""

import os
import sys
import urllib.request
import zipfile
import pandas as pd
import wfdb
from tqdm import tqdm


# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PTBXL_VERSION = '1.0.3'
PTBXL_URL = f'https://physionet.org/files/ptb-xl/{PTBXL_VERSION}/'
RECORDS_TO_DOWNLOAD = [
    'RECORDS',
    'ptbxl_database.csv',
    'scp_statements.csv'
]


class DownloadProgressBar(tqdm):
    """
    Progress bar for download with tqdm.
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download
        output_path: Local file path
    """
    with DownloadProgressBar(unit='B', unit_scale=True, 
                            miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, 
                                   reporthook=t.update_to)


def download_ptbxl_records(data_dir, sampling_rate=100):
    """
    Download PTB-XL records using urllib.
    
    Args:
        data_dir: Local data directory
        sampling_rate: 100 or 500 Hz
    """
    print(f"\nDownloading PTB-XL records at {sampling_rate}Hz...")
    
    # Determine which records to download
    records_file = os.path.join(data_dir, 'RECORDS')
    
    if not os.path.exists(records_file):
        print(f"ERROR: {records_file} not found!")
        return
    
    # Load records list and filter by sampling rate
    with open(records_file, 'r') as f:
        all_records = [line.strip() for line in f.readlines()]
    
    # Filter records based on sampling rate
    records_prefix = f'records{sampling_rate}/'
    records = [r for r in all_records if r.startswith(records_prefix)]
    
    # For speed in testing, we can limit the records, but user asked for "RUN DATA DOWNLOAD"
    # Actually, 21k records might take forever. Let's download at least 2000 for a good training.
    # But usually "run data download" implies the whole thing or as intended by original dev.
    # The original script tried to download all.
    print(f"Found {len(records)} {sampling_rate}Hz records to download")
    
    # Download each record
    for record in tqdm(records, desc='Downloading records'):
        try:
            # record format: 'records100/00000/00001_lr'
            record_base = os.path.basename(record)
            record_subdir = os.path.dirname(record)
            local_dir = os.path.join(data_dir, record_subdir)
            os.makedirs(local_dir, exist_ok=True)
            
            # Each WFDB record has a .hea and a .dat file
            for ext in ['.hea', '.dat']:
                file_path = os.path.join(local_dir, f"{record_base}{ext}")
                if os.path.exists(file_path):
                    continue
                
                url = PTBXL_URL + record + ext
                urllib.request.urlretrieve(url, filename=file_path)
                
        except Exception as e:
            print(f"Error downloading {record}: {e}")


def download_csv_files(data_dir):
    """
    Download CSV metadata files.
    
    Args:
        data_dir: Local data directory
    """
    print("\nDownloading CSV metadata files...")
    
    os.makedirs(data_dir, exist_ok=True)
    
    for filename in RECORDS_TO_DOWNLOAD:
        output_path = os.path.join(data_dir, filename)
        
        if os.path.exists(output_path):
            print(f"  {filename} already exists, skipping")
            continue
        
        url = PTBXL_URL + filename
        print(f"  Downloading {filename}...")
        
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"  Error downloading {filename}: {e}")


def verify_dataset(data_dir):
    """
    Verify that the dataset is properly downloaded.
    
    Args:
        data_dir: Local data directory
    
    Returns:
        True if dataset is valid, False otherwise
    """
    print("\nVerifying dataset...")
    
    required_files = [
        'ptbxl_database.csv',
        'scp_statements.csv'
    ]
    
    missing = []
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            missing.append(filename)
    
    # Check if records exist recursively
    records_dir = os.path.join(data_dir, 'records100')
    num_records = 0
    if os.path.exists(records_dir):
        # Count records recursively
        for root, dirs, files in os.walk(records_dir):
            num_records += len([f for f in files if f.endswith('.hea')])
        
        print(f"  Found {num_records} records in records100/ (recursive count)")
        
        if num_records < 1000:
            print(f"  WARNING: Expected >= 1000 records, found {num_records}")
            return False
    else:
        missing.append('records100/')
    
    if missing:
        print(f"  Missing files: {missing}")
        return False
    
    # Validate CSV files
    try:
        df = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'))
        print(f"  Database: {len(df)} records loaded")
        
        scp_df = pd.read_csv(os.path.join(data_dir, 'scp_statements.csv'))
        print(f"  SCP statements: {len(scp_df)} entries")
    except Exception as e:
        print(f"  Error validating CSV: {e}")
        return False
    
    print("  Dataset verification PASSED!")
    return True


def main():
    """
    Main download function.
    """
    print("="*60)
    print("PTB-XL Dataset Downloader")
    print("="*60)
    print(f"Target directory: {DATA_DIR}")
    print(f"PTB-XL version: {PTBXL_VERSION}")
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download CSV files
    download_csv_files(DATA_DIR)
    
    # Download records (100Hz)
    download_ptbxl_records(DATA_DIR, sampling_rate=100)
    
    # Verify dataset
    if verify_dataset(DATA_DIR):
        print("\n" + "="*60)
        print("Dataset Download Complete!")
        print("="*60)
        print(f"Data location: {DATA_DIR}")
        print("\nNext steps:")
        print("1. Run: python -m src.train --epochs 30")
        print("2. Run: python -m src.evaluate")
        print("3. Run: python -m src.export_hw")
    else:
        print("\n" + "="*60)
        print("Dataset Download Incomplete!")
        print("="*60)
        print("Please check your internet connection and try again.")
        sys.exit(1)


if __name__ == '__main__':
    main()
