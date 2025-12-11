#!/usr/bin/env python3
"""
Prepare training data for ML-based noise detection.

This script:
1. Generates noisy images using your existing noise_gen.py
2. Extracts features from each image using MATLAB extract_features.m
3. Creates a labeled CSV dataset for ML training

Usage:
    python prepare_training_data.py <clean_image_path> --num-per-type 20
"""

import subprocess
import sys
from pathlib import Path
import argparse
import pandas as pd
import re


def generate_training_images(clean_image_path, output_dir, num_per_type):
    """Generate noisy images for all noise types"""
    clean_image_path = Path(clean_image_path).resolve()
    output_dir = Path(output_dir).resolve()
    
    if not clean_image_path.exists():
        raise FileNotFoundError(f"Clean image not found: {clean_image_path}")
    
    print("=" * 80)
    print("STEP 1: Generating Noisy Training Images")
    print("=" * 80)
    
    # Use the existing noise_gen.py to generate images
    # Navigate up from training/ to project root
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    noise_gen_script = project_root / "noise_gen" / "noise_gen.py"
    
    if not noise_gen_script.exists():
        raise FileNotFoundError(f"noise_gen.py not found at: {noise_gen_script}")
    
    # Note: noise_gen.py creates images in parent.parent/noisy_output by default
    # We'll use that directory for feature extraction
    actual_output_dir = clean_image_path.parent.parent / "noisy_output"
    
    # Generate all noise types
    cmd = [
        sys.executable,
        str(noise_gen_script),
        str(clean_image_path),
        "-n", str(num_per_type),
        "-t", "all"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0 and "Generated" not in result.stdout:
        print("Error generating images:")
        print(result.stderr)
        return False, None
    
    print(f"\n[OK] Generated training images in: {actual_output_dir}\n")
    return True, actual_output_dir


def extract_features_from_image(image_path, matlab_func_dir):
    """Extract features from a single image using MATLAB"""
    image_path = str(Path(image_path).resolve())
    matlab_func_dir = str(Path(matlab_func_dir).resolve())
    
    # MATLAB command to extract features and print as comma-separated values
    matlab_cmd = (
        f"addpath('{matlab_func_dir}'); "
        f"try, "
        f"  features = extract_features('{image_path}'); "
        f"  fields = fieldnames(features); "
        f"  for i = 1:length(fields), "
        f"    fprintf('%.6f', features.(fields{{i}})); "
        f"    if i < length(fields), fprintf(','); end; "
        f"  end; "
        f"  fprintf('\\n'); "
        f"catch e, "
        f"  fprintf('ERROR: %s\\n', e.message); "
        f"  exit(1); "
        f"end; "
        f"exit(0);"
    )
    
    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        print(f"MATLAB error for {Path(image_path).name}:")
        print(result.stderr)
        return None
    
    # Parse the output - last line should be comma-separated values
    lines = result.stdout.strip().splitlines()
    
    # Find the line with feature values (should be all numbers/commas)
    feature_line = None
    for line in reversed(lines):
        if re.match(r'^[-0-9.,\s]+$', line.strip()):
            feature_line = line.strip()
            break
    
    if not feature_line:
        print(f"Could not parse features from MATLAB output for {Path(image_path).name}")
        return None
    
    try:
        feature_values = [float(x) for x in feature_line.split(',')]
        return feature_values
    except ValueError as e:
        print(f"Error parsing feature values: {e}")
        print(f"Raw output: {feature_line}")
        return None


def extract_label_from_filename(filename):
    """Extract the noise type label from filename"""
    filename_lower = filename.lower()
    
    if 'clean' in filename_lower or 'original' in filename_lower:
        return 'clean'
    elif 'gaussian' in filename_lower:
        return 'gaussian'
    elif 'salt_pepper' in filename_lower:
        return 'salt_pepper'
    elif 'speckle' in filename_lower:
        return 'speckle'
    elif 'uniform' in filename_lower:
        return 'uniform'
    else:
        return 'unknown'


def create_training_dataset(noisy_images_dir, output_csv):
    """Extract features from all images and create labeled CSV dataset"""
    noisy_images_dir = Path(noisy_images_dir).resolve()
    output_csv = Path(output_csv).resolve()
    
    print("=" * 80)
    print("STEP 2: Extracting Features from Training Images")
    print("=" * 80)
    
    # Find MATLAB function directory (now in training/)
    script_dir = Path(__file__).parent.resolve()
    matlab_func_dir = script_dir
    
    if not matlab_func_dir.exists():
        raise FileNotFoundError(f"MATLAB functions directory not found: {matlab_func_dir}")
    
    # Get feature names by running extract_features once
    print("Getting feature names from MATLAB...")
    
    matlab_cmd = (
        f"addpath('{matlab_func_dir}'); "
        f"fields = fieldnames(extract_features('{matlab_func_dir}/extract_features.m')); "
        f"for i = 1:length(fields), "
        f"  fprintf('%s', fields{{i}}); "
        f"  if i < length(fields), fprintf(','); end; "
        f"end; "
        f"fprintf('\\n'); "
        f"exit(0);"
    )
    
    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    # Try to get feature names, or use defaults
    feature_names = None
    for line in result.stdout.strip().splitlines():
        if ',' in line and not line.startswith('Warning'):
            feature_names = line.strip().split(',')
            break
    
    if not feature_names:
        # Default feature names in order
        feature_names = [
            'r2_linear', 'r2_quadratic', 'variance_coefficient', 
            'linear_slope', 'linear_intercept', 'quadratic_a',
            'has_central_peak', 'histogram_flatness', 'bimodal_extreme_ratio',
            'kurtosis', 'skewness', 'noise_variance',
            'var_mean_ratio', 'var_mean_squared_ratio',
            'salt_pepper_score', 'impulse_ratio', 'median_diff_variance',
            'dct_dc_energy', 'dct_ac_energy', 'edge_variance',
            'peak_intensity', 'min_intensity', 'entropy'
        ]
    
    print(f"Feature names: {feature_names}\n")
    
    # Find all image files
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = [f for f in noisy_images_dir.iterdir() 
              if f.suffix.lower() in img_extensions]
    
    if not images:
        raise FileNotFoundError(f"No images found in {noisy_images_dir}")
    
    print(f"Found {len(images)} images to process\n")
    
    # Extract features from each image
    data_rows = []
    
    for idx, img_path in enumerate(sorted(images), 1):
        filename = img_path.name
        label = extract_label_from_filename(filename)
        
        print(f"[{idx}/{len(images)}] Processing {filename} (label: {label})... ", end="", flush=True)
        
        features = extract_features_from_image(img_path, matlab_func_dir)
        
        if features is None:
            print("[X] FAILED")
            continue
        
        if len(features) != len(feature_names):
            print(f"[X] Feature count mismatch ({len(features)} vs {len(feature_names)})")
            continue
        
        # Create data row
        row = {'filename': filename, 'label': label}
        for name, value in zip(feature_names, features):
            row[name] = value
        
        data_rows.append(row)
        print("[OK]")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data_rows)
    
    # Reorder columns: filename, label, then all features
    cols = ['filename', 'label'] + feature_names
    df = df[cols]
    
    df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Total images processed: {len(data_rows)}")
    print(f"Features per image: {len(feature_names)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().to_string())
    print(f"\nDataset saved to: {output_csv}")
    print("=" * 80)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data for ML-based noise detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 20 images per noise type and extract features
  python prepare_training_data.py clean_image.jpg --num-per-type 20
  
  # Use custom output directory
  python prepare_training_data.py clean_image.jpg --num-per-type 30 --output-dir my_training_data
  
  # Only extract features from existing images (skip generation)
  python prepare_training_data.py --skip-generation --images-dir noisy_images
        """
    )
    
    parser.add_argument('clean_image', nargs='?', default=None,
                        help='Path to clean input image (required unless --skip-generation)')
    parser.add_argument('--num-per-type', type=int, default=15,
                        help='Number of images to generate per noise type (default: 15)')
    parser.add_argument('--output-dir', default='training_data',
                        help='Directory for generated images and CSV (default: training_data)')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip image generation, only extract features from existing images')
    parser.add_argument('--images-dir', default=None,
                        help='Directory containing images (used with --skip-generation)')
    
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir).resolve()
        
        if not args.skip_generation:
            # Generate training images
            if args.clean_image is None:
                parser.error("clean_image is required unless --skip-generation is used")
            
            success, actual_images_dir = generate_training_images(args.clean_image, output_dir, args.num_per_type)
            if not success:
                return 1
            
            images_dir = actual_images_dir if actual_images_dir else output_dir
        else:
            # Use existing images
            if args.images_dir is None:
                parser.error("--images-dir is required when using --skip-generation")
            
            images_dir = Path(args.images_dir).resolve()
            if not images_dir.exists():
                print(f"Error: Images directory not found: {images_dir}")
                return 1
        
        # Extract features and create CSV
        output_csv = output_dir.parent / f"{output_dir.name}_features.csv"
        dataset = create_training_dataset(images_dir, output_csv)
        
        print(f"\n[OK] Training data preparation complete!")
        print(f"  - Images: {images_dir}")
        print(f"  - Dataset: {output_csv}")
        
        return 0
        
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
