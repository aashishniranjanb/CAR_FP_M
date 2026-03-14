"""
Training Script for FPGA-Optimized ECG MI Classifier
=====================================================

PyTorch training loop with class-weighted loss for handling
imbalanced MI vs NORM dataset.

Features:
- Adam optimizer with learning rate 1e-3
- Class-weighted CrossEntropyLoss
- Early stopping mechanism
- Learning rate scheduling
- Model checkpointing

Author: FPGA Hackathon Team
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class EarlyStopping:
    """Simple early stopping handler."""
    def __init__(self, patience=5, verbose=True, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        if self.verbose:
            print(f'Validation accuracy improved. Saving model...')
        torch.save(model.state_dict(), self.path)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_data_loaders, compute_class_weights
from src.model import create_model, FPGA_ECG_CNN


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: 'cuda' or 'cpu'
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: Neural network model
        val_loader: Validation DataLoader
        criterion: Loss function
        device: 'cuda' or 'cpu'
    
    Returns:
        Tuple of (average validation loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main(args):
    """
    Main training function.
    
    Args:
        args: Command-line arguments
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading PTB-XL dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Handle case where data loading fails
    if train_loader is None:
        print("Error: Failed to load dataset. Please run download_data.py first.")
        return

    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = create_model(model_type=args.model_type, num_classes=2)
    model = model.to(device)
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Compute class weights for imbalanced dataset
    # Hardcoded weights to penalize MI misclassification heavily
    # NORM: 733, MI: 385. Total: 1118
    # Weight = Total / (Number of Classes * Class Count)
    weight_norm = 1118 / (2 * 733)  # Roughly 0.76
    weight_mi   = 1118 / (2 * 385)  # Roughly 1.45
    
    class_weights = torch.tensor([weight_norm, weight_mi], dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=5, 
        verbose=True,
        path=os.path.join(args.output_dir, 'best_model.pth')
    )
    
    # Training loop
    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Log history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"[+] Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping check
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            break
    
    # Save training history
    np.save(os.path.join(args.output_dir, 'history.npy'), history)
    
    print("\n" + "="*50)
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.output_dir}/best_model.pth")
    print("="*50)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    
    # Test evaluation
    print("\nFinal Test Evaluation:")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ECG MI Classifier')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--model_type', type=str, default='standard', 
                        choices=['standard', 'lightweight'], help='Model type')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    main(args)
