import subprocess
import sys
from pathlib import Path
import joblib
import numpy as np

def detect_noise(img_path, use_ml=True):
    """
    Detect noise type in an image.
    
    Args:
        img_path: Path to the image file
        use_ml: If True, use ML model; if False, use threshold-based MATLAB method
    
    Returns:
        Detected noise type as string
    """
    img_path = str(Path(img_path).resolve())

    if use_ml:
        # Use ML model for detection
        return detect_noise_ml(img_path)
    else:
        # Use legacy threshold-based MATLAB method
        return detect_noise_matlab(img_path)


def detect_noise_ml(img_path):
    """Detect noise using trained Random Forest model"""
    
    # Load the trained model
    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir.parent / 'training' / 'models' / 'random_forest_model.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"ML model not found at {model_path}. "
            f"Please train the model first by running train_random_forest.py"
        )
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Extract features using MATLAB
    features = extract_features_matlab(img_path)
    
    if features is None:
        return 'error'
    
    # Ensure features are in correct order
    feature_vector = np.array([features]).reshape(1, -1)
    
    # Predict noise type
    noise_type = model.predict(feature_vector)[0]
    
    return noise_type


def extract_features_matlab(img_path):
    """Extract 23 features using MATLAB extract_features.m"""
    
    script_dir = Path(__file__).parent.resolve()
    extract_features_path = script_dir.parent / 'training' / 'extract_features.m'
    
    if not extract_features_path.exists():
        raise FileNotFoundError(f"extract_features.m not found at {extract_features_path}")
    
    matlab_func_dir = str(extract_features_path.parent)
    
    matlab_cmd = (
        f"addpath('{matlab_func_dir}'); "
        f"try, "
        f"features = extract_features('{img_path}'); "
        f"fprintf('%s\\n', jsonencode(features)); "
        f"catch e, disp('ERROR:'), disp(getReport(e)), exit(1); "
        f"end; exit(0);"
    )
    
    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True, text=True
    )
    
    # Parse JSON output
    lines = result.stdout.strip().splitlines()
    
    # Find the JSON line (last line that starts with [ or {)
    json_line = None
    for line in reversed(lines):
        if line.strip().startswith('[') or line.strip().startswith('{'):
            json_line = line.strip()
            break
    
    if json_line is None:
        print("ERROR: Could not extract features from MATLAB output")
        print(result.stdout)
        return None
    
    try:
        import json
        features_dict = json.loads(json_line)
        
        # Convert dict to list in the same order as training CSV
        # Order: r2_linear, r2_quadratic, variance_coefficient, linear_slope, linear_intercept, 
        #        quadratic_a, has_central_peak, histogram_flatness, bimodal_extreme_ratio,
        #        kurtosis, skewness, noise_variance, var_mean_ratio, var_mean_squared_ratio,
        #        salt_pepper_score, impulse_ratio, median_diff_variance, dct_dc_energy,
        #        dct_ac_energy, edge_variance, peak_intensity, min_intensity, entropy
        
        feature_order = [
            'r2_linear', 'r2_quadratic', 'variance_coefficient', 'linear_slope', 
            'linear_intercept', 'quadratic_a', 'has_central_peak', 'histogram_flatness',
            'bimodal_extreme_ratio', 'kurtosis', 'skewness', 'noise_variance',
            'var_mean_ratio', 'var_mean_squared_ratio', 'salt_pepper_score',
            'impulse_ratio', 'median_diff_variance', 'dct_dc_energy', 'dct_ac_energy',
            'edge_variance', 'peak_intensity', 'min_intensity', 'entropy'
        ]
        
        features_list = [features_dict[key] for key in feature_order]
        return features_list
        
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        print(f"JSON line: {json_line}")
        return None
    except KeyError as e:
        print(f"ERROR: Missing feature in MATLAB output: {e}")
        print(f"Features dict: {features_dict}")
        return None


def detect_noise_matlab(img_path):
    """Legacy threshold-based detection using MATLAB detect_noise_type.m"""
    
    script_dir = Path(__file__).parent.resolve()
    matlab_func_dir = str(script_dir)

    matlab_cmd = (
        f"addpath('{matlab_func_dir}'); "
        f"try, "
        f"result = detect_noise_type('{img_path}'); "
        f"disp(result); "
        f"catch e, disp('ERROR:'), disp(getReport(e)), exit(1); "
        f"end; exit(0);"
    )

    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True, text=True
    )

    # Extract last printed line
    lines = result.stdout.strip().splitlines()
    return lines[-1]


def process_folder(folder_path, use_ml=True):
    """
    Process all images in a folder
    
    Args:
        folder_path: Path to folder containing images
        use_ml: If True, use ML model; if False, use threshold-based method
    """
    folder_path = Path(folder_path).resolve()
    
    if not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory")
        return
    
    # Supported image extensions
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}
    
    # Find all images
    images = [f for f in folder_path.iterdir() 
              if f.suffix.lower() in img_extensions]
    
    if not images:
        print(f"No images found in {folder_path}")
        return
    
    method = "ML Model" if use_ml else "Threshold-based"
    print(f"Found {len(images)} image(s) in {folder_path}")
    print(f"Detection method: {method}\n")
    
    results = {}
    for img_path in sorted(images):
        try:
            filename = img_path.name
            # Extract actual noise type from filename (before first number or special char)
            actual_noise = extract_noise_type(filename)
            
            print(f"Processing: {filename} ... ", end="", flush=True)
            noise_type = detect_noise(str(img_path), use_ml=use_ml)
            results[filename] = (noise_type, actual_noise)
            
            match_symbol = "[OK]" if noise_type == actual_noise else "[X]"
            print(f"got: {noise_type} ... actual: {actual_noise} {match_symbol}")
        except Exception as e:
            results[filename] = (f"ERROR: {str(e)}", "unknown")
            print(f"[X] Error: {str(e)}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    correct = 0
    total = 0
    for filename, (detected, actual) in results.items():
        match = "[OK]" if detected == actual else "[X]"
        print(f"{filename}: got {detected:15} | actual {actual:15} {match}")
        if detected == actual:
            correct += 1
        total += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print("="*80)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")


def extract_noise_type(filename):
    """Extract the actual noise type from filename"""
    filename_lower = filename.lower()
    
    if 'clean' in filename_lower or 'original' in filename_lower:
        return 'clean'
    elif 'gaussian' in filename_lower:
        return 'gaussian'
    elif 'jpeg_artifact' in filename_lower:
        return 'jpeg_artifact'
    elif 'poisson' in filename_lower:
        return 'poisson'
    elif 'salt_pepper' in filename_lower:
        return 'salt_pepper'
    elif 'speckle' in filename_lower:
        return 'speckle'
    elif 'uniform' in filename_lower:
        return 'uniform'
    else:
        return 'unknown'


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image:  python detect_noise.py <image_path> [--legacy]")
        print("  Folder batch:  python detect_noise.py <folder_path> [--legacy]")
        print("")
        print("Options:")
        print("  --legacy       Use threshold-based MATLAB method instead of ML model")
        sys.exit(1)
    
    path = sys.argv[1]
    use_ml = '--legacy' not in sys.argv
    
    path_obj = Path(path)
    
    if path_obj.is_dir():
        # Batch process folder
        process_folder(path, use_ml=use_ml)
    elif path_obj.is_file():
        # Single image
        noise = detect_noise(path, use_ml=use_ml)
        print("Detected noise type:", noise)
    else:
        print(f"Error: {path} not found")
        sys.exit(1)
