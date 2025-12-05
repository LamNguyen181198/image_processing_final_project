import subprocess
import sys
from pathlib import Path

def detect_noise(img_path):
    img_path = str(Path(img_path).resolve())

    # Folder where detect_noise.py is located
    script_dir = Path(__file__).parent.resolve()

    # Absolute path to MATLAB function folder
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

    print(result.stdout)

    # Extract last printed line
    lines = result.stdout.strip().splitlines()
    return lines[-1]


def process_folder(folder_path):
    """Process all images in a folder"""
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
    
    print(f"Found {len(images)} image(s) in {folder_path}\n")
    
    results = {}
    for img_path in sorted(images):
        try:
            filename = img_path.name
            # Extract actual noise type from filename (before first number or special char)
            actual_noise = extract_noise_type(filename)
            
            print(f"Processing: {filename} ... ", end="", flush=True)
            noise_type = detect_noise(str(img_path))
            results[filename] = (noise_type, actual_noise)
            
            match_symbol = "✓" if noise_type == actual_noise else "✗"
            print(f"got: {noise_type} ... actual: {actual_noise} {match_symbol}")
        except Exception as e:
            results[filename] = (f"ERROR: {str(e)}", "unknown")
            print(f"✗ Error")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    correct = 0
    total = 0
    for filename, (detected, actual) in results.items():
        match = "✓" if detected == actual else "✗"
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
        print("  Single image:  python detect_noise.py <image_path>")
        print("  Folder batch:  python detect_noise.py <folder_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    path_obj = Path(path)
    
    if path_obj.is_dir():
        # Batch process folder
        process_folder(path)
    elif path_obj.is_file():
        # Single image
        noise = detect_noise(path)
        print("Detected noise type:", noise)
    else:
        print(f"Error: {path} not found")
        sys.exit(1)
