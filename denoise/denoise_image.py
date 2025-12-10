#!/usr/bin/env python3
"""
Complete Denoising Pipeline: Detect noise type and apply optimal filter

This script:
1. Detects noise type using trained Random Forest model
2. Applies optimal denoising filter based on detected noise
3. Saves denoised image and generates comparison visualization

Usage:
    python denoise_image.py <input_image> [--output <output_path>]
    python denoise_image.py <input_folder> --batch [--output <output_dir>]
"""

import sys
import subprocess
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import noise detection
sys.path.append(str(Path(__file__).parent.parent / 'noise_detecting'))
from detect_noise import detect_noise


def denoise_image(img_path, output_path=None, use_ml=True, verbose=True):
    """
    Denoise a single image using ML-based detection + optimal filtering
    
    Args:
        img_path: Path to noisy image
        output_path: Path to save denoised image (optional)
        use_ml: Use ML model for detection (True) or legacy method (False)
        verbose: Print progress messages
    
    Returns:
        tuple: (denoised_array, noise_type) or (None, None) on error
    """
    img_path = Path(img_path).resolve()
    
    if not img_path.exists():
        print(f"Error: Image not found: {img_path}")
        return None, None
    
    if verbose:
        print("=" * 80)
        print(f"DENOISING: {img_path.name}")
        print("=" * 80)
    
    # Step 1: Detect noise type
    if verbose:
        print("\nStep 1: Detecting noise type...")
    
    try:
        noise_type = detect_noise(str(img_path), use_ml=use_ml)
        if verbose:
            method = "ML Model" if use_ml else "Threshold-based"
            print(f"  Detected noise: {noise_type} (using {method})")
    except Exception as e:
        print(f"Error detecting noise: {e}")
        return None, None
    
    # Step 2: Apply denoising filter
    if verbose:
        print(f"\nStep 2: Applying optimal filter for '{noise_type}' noise...")
    
    try:
        denoised = apply_denoise_filter(str(img_path), noise_type, verbose=verbose)
        if denoised is None:
            print("Error: Denoising failed")
            return None, None
    except Exception as e:
        print(f"Error during denoising: {e}")
        return None, None
    
    # Step 3: Save denoised image
    if output_path is not None:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to uint8 for saving
        denoised_uint8 = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(denoised_uint8).save(output_path)
        
        if verbose:
            print(f"\n[OK] Denoised image saved to: {output_path}")
    
    return denoised, noise_type


def apply_denoise_filter(img_path, noise_type, verbose=True):
    """
    Apply denoising filter using MATLAB denoise_filters.m
    
    Args:
        img_path: Path to noisy image
        noise_type: Detected noise type
        verbose: Print filter details
    
    Returns:
        numpy array: Denoised image (grayscale, float64 [0,1])
    """
    script_dir = Path(__file__).parent.resolve()
    denoise_func_path = script_dir / 'denoise_filters.m'
    
    if not denoise_func_path.exists():
        raise FileNotFoundError(f"denoise_filters.m not found at {denoise_func_path}")
    
    matlab_func_dir = str(script_dir)
    
    # Create temporary file for MATLAB output
    temp_output = script_dir / 'temp_denoised.png'
    
    matlab_cmd = (
        f"addpath('{matlab_func_dir}'); "
        f"try, "
        f"denoised = denoise_filters('{img_path}', '{noise_type}'); "
        f"if ~isempty(denoised), "
        f"imwrite(denoised, '{temp_output}'); "
        f"fprintf('SUCCESS\\n'); "
        f"else, "
        f"fprintf('ERROR: Denoising returned empty result\\n'); "
        f"exit(1); "
        f"end; "
        f"catch e, "
        f"fprintf('ERROR: %s\\n', e.message); "
        f"disp(getReport(e)); "
        f"exit(1); "
        f"end; "
        f"exit(0);"
    )
    
    if verbose:
        print(f"  Running MATLAB denoising filter...")
    
    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True, text=True
    )
    
    # Print MATLAB output if verbose
    if verbose:
        for line in result.stdout.strip().splitlines():
            if line.strip() and not line.startswith('MATLAB'):
                print(f"    {line}")
    
    # Check if denoising succeeded
    if not temp_output.exists():
        print("Error: MATLAB did not produce output image")
        print("MATLAB output:")
        print(result.stdout)
        return None
    
    # Read denoised image
    denoised = np.array(Image.open(temp_output).convert('L')).astype(np.float64) / 255.0
    
    # Clean up temporary file
    temp_output.unlink()
    
    return denoised


def create_comparison(original_path, denoised_array, noise_type, output_path):
    """Create side-by-side comparison visualization"""
    
    # Load original image
    original = np.array(Image.open(original_path).convert('L')).astype(np.float64) / 255.0
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Noisy Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Denoised image
    axes[1].imshow(denoised_array, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Denoised Image\n(Detected: {noise_type})', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Difference image
    diff = np.abs(original - denoised_array)
    im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    axes[2].set_title('Removed Noise\n(Difference)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Comparison saved to: {output_path}")


def batch_denoise(input_folder, output_folder=None, use_ml=True, create_comparisons=True):
    """Batch denoise all images in a folder"""
    
    input_folder = Path(input_folder).resolve()
    
    if not input_folder.is_dir():
        print(f"Error: {input_folder} is not a valid directory")
        return
    
    # Setup output folder
    if output_folder is None:
        output_folder = input_folder / 'denoised'
    else:
        output_folder = Path(output_folder).resolve()
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = [f for f in input_folder.iterdir() 
              if f.suffix.lower() in img_extensions]
    
    if not images:
        print(f"No images found in {input_folder}")
        return
    
    print("\n" + "=" * 80)
    print("BATCH DENOISING")
    print("=" * 80)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Images to process: {len(images)}")
    print(f"Detection method: {'ML Model' if use_ml else 'Threshold-based'}")
    print("=" * 80 + "\n")
    
    results = []
    
    for i, img_path in enumerate(sorted(images), 1):
        print(f"\n[{i}/{len(images)}] Processing: {img_path.name}")
        print("-" * 80)
        
        # Denoise image
        output_path = output_folder / f"denoised_{img_path.name}"
        denoised, noise_type = denoise_image(
            img_path, 
            output_path=output_path,
            use_ml=use_ml,
            verbose=True
        )
        
        if denoised is not None:
            results.append((img_path.name, noise_type, 'Success'))
            
            # Create comparison visualization
            if create_comparisons:
                comparison_path = output_folder / f"comparison_{img_path.stem}.png"
                create_comparison(img_path, denoised, noise_type, comparison_path)
        else:
            results.append((img_path.name, 'N/A', 'Failed'))
    
    # Summary
    print("\n" + "=" * 80)
    print("BATCH DENOISING SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for _, _, status in results if status == 'Success')
    
    for filename, noise_type, status in results:
        status_symbol = "[OK]" if status == 'Success' else "[X]"
        print(f"{status_symbol} {filename:40s} -> {noise_type:15s}")
    
    print("=" * 80)
    print(f"Successfully denoised: {success_count}/{len(results)} images")
    print(f"Output directory: {output_folder}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Denoise images using ML-based noise detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single image:
    python denoise_image.py noisy.png
    python denoise_image.py noisy.png --output clean.png
    
  Batch processing:
    python denoise_image.py images/ --batch
    python denoise_image.py images/ --batch --output results/
    
  Use legacy detection:
    python denoise_image.py noisy.png --legacy
        """
    )
    
    parser.add_argument('input', help='Input image or folder')
    parser.add_argument('--output', '-o', help='Output path for denoised image(s)')
    parser.add_argument('--batch', action='store_true', 
                        help='Process all images in input folder')
    parser.add_argument('--legacy', action='store_true',
                        help='Use legacy threshold-based detection instead of ML')
    parser.add_argument('--no-comparison', action='store_true',
                        help='Skip creating comparison visualizations (batch mode only)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    use_ml = not args.legacy
    
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        return 1
    
    if args.batch or input_path.is_dir():
        # Batch processing
        batch_denoise(
            input_path, 
            output_folder=args.output,
            use_ml=use_ml,
            create_comparisons=not args.no_comparison
        )
    else:
        # Single image
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"denoised_{input_path.name}"
        
        denoised, noise_type = denoise_image(
            input_path,
            output_path=output_path,
            use_ml=use_ml,
            verbose=True
        )
        
        if denoised is not None:
            # Create comparison
            comparison_path = output_path.parent / f"comparison_{output_path.stem}.png"
            create_comparison(input_path, denoised, noise_type, comparison_path)
            
            print("\n" + "=" * 80)
            print("DENOISING COMPLETE")
            print("=" * 80)
            return 0
        else:
            return 1


if __name__ == '__main__':
    sys.exit(main())
