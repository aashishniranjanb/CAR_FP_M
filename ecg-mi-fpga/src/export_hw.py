"""
Hardware Export Script - INT8 Quantization and BRAM .hex Generation
==================================================================

Converts trained PyTorch model to INT8 fixed-point and generates
Verilog-ready .hex files for BRAM initialization.

Key Features:
- Symmetric INT8 quantization (-128 to 127)
- Weight and activation quantization
- .hex file generation for Vivado $readmemh
- Test vector export for RTL verification

Author: FPGA Hackathon Team
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_data_loaders, PTBXLDataset
from src.model import create_model


def symmetric_quantize(tensor, num_bits=8):
    """
    Symmetric fixed-point quantization.
    
    Maps float values to int8 range [-128, 127].
    Zero is perfectly preserved (symmetric).
    
    Args:
        tensor: Input tensor (PyTorch)
        num_bits: Number of bits (8 for INT8)
    
    Returns:
        Quantized tensor and scale factor
    """
    max_val = tensor.abs().max()
    
    if max_val == 0:
        return torch.zeros_like(tensor, dtype=torch.int8), 1.0
    
    # Scale factor to map max value to (2^(num_bits-1) - 1)
    scale = (2 ** (num_bits - 1) - 1) / max_val
    
    # Quantize
    quantized = torch.round(tensor * scale)
    
    # Clamp to int8 range [-128, 127]
    max_int = 2 ** (num_bits - 1) - 1  # 127 for 8-bit
    quantized = torch.clamp(quantized, -max_int, max_int)
    
    return quantized.to(torch.int8), scale.item()


def quantize_model(model):
    """
    Quantize all model weights to INT8.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary of quantized weights and scale factors
    """
    quantized_state = {}
    scales = {}
    
    print("Quantizing model weights to INT8...")
    
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            quantized, scale = symmetric_quantize(param.data)
            quantized_state[name] = quantized
            scales[name] = scale
            print(f"  {name}: scale={scale:.6f}, range=[{quantized.min()}, {quantized.max()}]")
    
    return quantized_state, scales


def export_weights_hex(quantized_state, output_dir='weights_hex'):
    """
    Export quantized weights to .hex files for Vivado BRAM.
    
    Format: One hex value per line (e.g., "A5", "0F")
    Suitable for $readmemh in Verilog.
    
    Args:
        quantized_state: Dictionary of quantized weight tensors
        output_dir: Output directory for .hex files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Map layer names to file names
    layer_mapping = {
        'conv1.weight': 'conv1_weights.hex',
        'conv2.weight': 'conv2_weights.hex',
        'conv3.weight': 'conv3_weights.hex',
        'fc1.weight': 'fc1_weights.hex',
        'fc2.weight': 'fc2_weights.hex',
        'conv1.bias': 'conv1_bias.hex',
        'conv2.bias': 'conv2_bias.hex',
        'conv3.bias': 'conv3_bias.hex',
        'fc1.bias': 'fc1_bias.hex',
        'fc2.bias': 'fc2_bias.hex',
    }
    
    print(f"\nExporting weights to {output_dir}/")
    
    for param_name, quantized_tensor in quantized_state.items():
        # Flatten tensor
        flat = quantized_tensor.flatten().numpy()
        
        # Convert to hex strings
        # Handle negative values (two's complement)
        hex_lines = []
        for val in flat:
            # Convert to unsigned 8-bit representation
            if val < 0:
                val = val + 256
            hex_lines.append(f"{val:02X}")
        
        # Write to file
        if param_name in layer_mapping:
            filepath = os.path.join(output_dir, layer_mapping[param_name])
            with open(filepath, 'w') as f:
                f.write('\n'.join(hex_lines))
            print(f"  {layer_mapping[param_name]}: {len(hex_lines)} values")


def quantize_test_sample(sample, num_bits=8):
    """
    Quantize a test ECG sample to INT8.
    
    Args:
        sample: ECG sample tensor (12 leads x 1000 samples)
        num_bits: Quantization bits
    
    Returns:
        Quantized sample and scale
    """
    return symmetric_quantize(sample, num_bits)


def export_test_vectors(model, test_loader, output_dir='test_vectors'):
    """
    Export test vectors for RTL verification.
    
    Exports:
    - input_ecg.hex: Quantized ECG input (12 leads x 1000 samples)
    - expected_output.txt: Predicted class and logits
    
    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nExporting test vectors to {output_dir}/")
    
    # Get first batch
    model.eval()
    for batch_x, batch_y in test_loader:
        # Take first sample
        sample = batch_x[0]  # Shape: (12, 1000)
        label = batch_y[0].item()
        
        # Quantize sample
        quantized_sample, scale = quantize_test_sample(sample)
        
        # Export to hex (12 leads x 1000 samples = 12000 values)
        sample_flat = quantized_sample.flatten().numpy()
        
        hex_lines = []
        for val in sample_flat:
            if val < 0:
                val = val + 256
            hex_lines.append(f"{val:02X}")
        
        input_hex_path = os.path.join(output_dir, 'input_ecg.hex')
        with open(input_hex_path, 'w') as f:
            f.write('\n'.join(hex_lines))
        
        print(f"  input_ecg.hex: {len(hex_lines)} values (scale: {scale:.6f})")
        
        # Get expected output
        with torch.no_grad():
            sample_batch = sample.unsqueeze(0)
            output = model(sample_batch)
            logits = output[0].numpy()
            pred_class = output.argmax(dim=1).item()
        
        # Save expected output
        expected_path = os.path.join(output_dir, 'expected_output.txt')
        with open(expected_path, 'w') as f:
            f.write(f"Predicted Class: {pred_class}\n")
            f.write(f"Label: {'MI' if pred_class == 1 else 'NORM'}\n")
            f.write(f"True Label: {label}\n")
            f.write(f"Logits: {logits[0]:.6f}, {logits[1]:.6f}\n")
            f.write(f"Probabilities: NORM={1-logits[1]:.6f}, MI={logits[1]:.6f}\n")
        
        print(f"  expected_output.txt: pred={pred_class}, true={label}")
        print(f"    Logits: NORM={logits[0]:.4f}, MI={logits[1]:.4f}")
        
        break  # Only export first sample
    
    print(f"\nTest vectors exported to {output_dir}/")


def verify_quantized_inference(model, quantized_state, sample):
    """
    Verify that quantized inference produces reasonable results.
    
    Args:
        model: Original float model
        quantized_state: Quantized weights
        sample: Test sample
    
    Returns:
        Float and quantized outputs
    """
    model.eval()
    
    # Float inference
    with torch.no_grad():
        float_output = model(sample.unsqueeze(0))[0]
    
    # Quantized inference (using quantized weights)
    q_model = create_model(model_type='standard', num_classes=2)
    q_model.load_state_dict({k: v.float() for k, v in quantized_state.items()}, strict=False)
    q_model.eval()
    
    with torch.no_grad():
        quantized_output = q_model(sample.unsqueeze(0))[0]
    
    return float_output.numpy(), quantized_output.numpy()


def main(args):
    """
    Main export function.
    
    Args:
        args: Command-line arguments
    """
    print("="*60)
    print("Hardware Export - INT8 Quantization")
    print("="*60)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = create_model(model_type=args.model_type, num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Quantize model
    quantized_state, scales = quantize_model(model)
    
    # Export weights to .hex files
    export_weights_hex(quantized_state, output_dir=args.weights_dir)
    
    # Load test data
    print("\nLoading test data for export...")
    _, _, test_loader = get_data_loaders(
        batch_size=1,
        num_workers=0  # Use 0 for single sample
    )
    
    # Export test vectors
    export_test_vectors(model, test_loader, output_dir=args.test_vectors_dir)
    
    print("\n" + "="*60)
    print("Hardware Export Complete!")
    print("="*60)
    print(f"Weights: {args.weights_dir}/*.hex")
    print(f"Test Vectors: {args.test_vectors_dir}/")
    print("\nThese files are ready for Vivado BRAM initialization.")
    print("Use $readmemh in Verilog testbench for RTL verification.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export hardware weights')
    parser.add_argument('--model_path', type=str, default='results/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--weights_dir', type=str, default='weights_hex',
                        help='Output directory for weights .hex files')
    parser.add_argument('--test_vectors_dir', type=str, default='test_vectors',
                        help='Output directory for test vectors')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'lightweight'],
                        help='Model type')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    
    args = parser.parse_args()
    main(args)
