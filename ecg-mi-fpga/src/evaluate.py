"""
Evaluation Script for ECG MI Classifier
========================================

Generates comprehensive metrics and visualizations for the technical report:
- Confusion Matrix
- ROC Curve with AUC
- Accuracy, Precision, Recall, F1-Score

Author: FPGA Hackathon Team
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_data_loaders
from src.model import create_model


def evaluate_model(model, test_loader, device):
    """
    Run inference on test set and collect predictions.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: 'cuda' or 'cpu'
    
    Returns:
        Tuple of (y_true, y_pred, y_prob)
    """
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data = data.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get predictions
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            # Store results
            y_true.extend(target.numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob[:, 1].cpu().numpy())  # Probability of MI class
    
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """
    Generate and save confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['NORM', 'MI'],
        yticklabels=['NORM', 'MI'],
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix: ECG MI Classification', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def plot_roc_curve(y_true, y_prob, save_path='results/roc_curve.png'):
    """
    Generate and save ROC curve plot.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for MI class
        save_path: Path to save plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: ECG MI Classification', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {save_path}")
    return roc_auc


def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),  # Sensitivity
        'f1_score': f1_score(y_true, y_pred, average='binary'),
        'specificity': recall_score(y_true, y_pred, pos_label=0),
    }
    
    # Compute AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics['auc'] = auc(fpr, tpr)
    
    return metrics


def print_metrics(metrics):
    """
    Pretty print metrics dictionary.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("Classification Metrics")
    print("="*50)
    print(f"Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"Precision:   {metrics['precision']*100:.2f}%")
    print(f"Recall:      {metrics['recall']*100:.2f}%  (Sensitivity)")
    print(f"Specificity: {metrics['specificity']*100:.2f}%")
    print(f"F1-Score:    {metrics['f1_score']*100:.2f}%")
    print(f"AUC:         {metrics['auc']:.4f}")
    print("="*50)


def main(args):
    """
    Main evaluation function.
    
    Args:
        args: Command-line arguments
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading test dataset...")
    _, _, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = create_model(model_type=args.model_type, num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Evaluate
    print("\nRunning evaluation...")
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion Matrix
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # ROC Curve
    roc_auc = plot_roc_curve(
        y_true, y_prob,
        save_path=os.path.join(args.output_dir, 'roc_curve.png')
    )
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("ECG MI Classification - Test Metrics\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy:    {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Precision:   {metrics['precision']*100:.2f}%\n")
        f.write(f"Recall:      {metrics['recall']*100:.2f}%\n")
        f.write(f"Specificity: {metrics['specificity']*100:.2f}%\n")
        f.write(f"F1-Score:    {metrics['f1_score']*100:.2f}%\n")
        f.write(f"AUC:         {metrics['auc']:.4f}\n")
        
        # Classification Report
        f.write("\n" + "="*50 + "\n")
        f.write("Detailed Classification Report\n")
        f.write("="*50 + "\n\n")
        f.write(classification_report(y_true, y_pred, target_names=['NORM', 'MI']))
    
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"\nEvaluation complete! Results in: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ECG MI Classifier')
    parser.add_argument('--model_path', type=str, default='results/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for plots and metrics')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader workers')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'lightweight'],
                        help='Model type')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    
    args = parser.parse_args()
    main(args)
