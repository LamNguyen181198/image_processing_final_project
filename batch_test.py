import subprocess
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

def detect_noise_matlab(img_path, matlab_func_dir):
    """Run MATLAB detection on a single image"""
    img_path = str(Path(img_path).resolve())
    
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

    # Extract last line (the detected noise type)
    lines = result.stdout.strip().splitlines()
    return lines[-1] if lines else "error"


def parse_filename(filename):
    """Extract ground truth noise type and parameters from filename"""
    stem = Path(filename).stem
    
    # Match patterns like: gaussian_01_sigma15.0, salt_pepper_02_density0.05, poisson_01_peak1
    if stem == "clean_original":
        return "clean", {}
    
    match = re.match(r'(\w+)_\d+_(.*)', stem)
    if match:
        noise_type = match.group(1)
        params_str = match.group(2)
        
        # Parse parameters
        params = {}
        param_match = re.findall(r'([a-z]+)([\d.]+)', params_str)
        for key, val in param_match:
            params[key] = float(val)
        
        return noise_type, params
    
    return "unknown", {}


def batch_test_images(image_dir, output_csv="detection_results.csv"):
    """Test all images in a directory and save results"""
    image_dir = Path(image_dir)
    script_dir = Path(__file__).parent.resolve() / "noise_detecting"
    matlab_func_dir = str(script_dir)
    
    # Find all PNG images
    image_files = sorted(image_dir.glob("*.png"))
    
    if not image_files:
        print(f"No PNG images found in {image_dir}")
        return None
    
    print(f"Found {len(image_files)} images. Testing...\n")
    
    results = []
    
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Testing: {img_path.name}", end=" ... ")
        
        # Get ground truth from filename
        true_type, params = parse_filename(img_path.name)
        
        # Detect noise
        detected_type = detect_noise_matlab(img_path, matlab_func_dir)
        
        # Check if correct
        correct = (detected_type == true_type)
        
        results.append({
            'filename': img_path.name,
            'true_type': true_type,
            'detected_type': detected_type,
            'correct': correct,
            'params': str(params)
        })
        
        status = "✓" if correct else "✗"
        print(f"{status} (True: {true_type}, Detected: {detected_type})")
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")
    
    return df


def visualize_results(df, output_image="detection_results.png"):
    """Create visualization of detection results"""
    
    # Calculate accuracy metrics
    overall_accuracy = (df['correct'].sum() / len(df)) * 100
    
    # Accuracy by noise type
    accuracy_by_type = df.groupby('true_type').agg({
        'correct': ['sum', 'count']
    })
    accuracy_by_type.columns = ['correct', 'total']
    accuracy_by_type['accuracy'] = (accuracy_by_type['correct'] / accuracy_by_type['total']) * 100
    
    # Confusion matrix data
    confusion_data = df.groupby(['true_type', 'detected_type']).size().reset_index(name='count')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Overall Accuracy
    ax1 = fig.add_subplot(gs[0, :])
    ax1.text(0.5, 0.5, f'Overall Accuracy: {overall_accuracy:.1f}%', 
             fontsize=32, fontweight='bold', ha='center', va='center')
    ax1.text(0.5, 0.3, f'({df["correct"].sum()} / {len(df)} images correct)', 
             fontsize=16, ha='center', va='center', color='gray')
    ax1.axis('off')
    
    # 2. Accuracy by Noise Type
    ax2 = fig.add_subplot(gs[1, 0])
    noise_types = accuracy_by_type.index.tolist()
    accuracies = accuracy_by_type['accuracy'].tolist()
    colors = ['green' if acc >= 80 else 'orange' if acc >= 50 else 'red' for acc in accuracies]
    
    bars = ax2.barh(noise_types, accuracies, color=colors, alpha=0.7)
    ax2.set_xlabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy by Noise Type', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Add percentage labels
    for i, (bar, acc, total) in enumerate(zip(bars, accuracies, accuracy_by_type['total'])):
        ax2.text(acc + 2, i, f'{acc:.1f}% ({accuracy_by_type.iloc[i]["correct"]:.0f}/{total})', 
                va='center', fontsize=10)
    
    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[1, 1])
    pivot_table = confusion_data.pivot(index='true_type', columns='detected_type', values='count').fillna(0)
    
    sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='Blues', ax=ax3, 
                cbar_kws={'label': 'Count'})
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Detected Type', fontsize=12)
    ax3.set_ylabel('True Type', fontsize=12)
    
    # 4. Detection Results Summary
    ax4 = fig.add_subplot(gs[2, :])
    
    summary_text = "Detection Summary:\n\n"
    for noise_type in accuracy_by_type.index:
        row = accuracy_by_type.loc[noise_type]
        summary_text += f"{noise_type.upper()}: {row['accuracy']:.1f}% ({int(row['correct'])}/{int(row['total'])})\n"
    
    # Find common misclassifications
    misclassified = df[~df['correct']]
    if len(misclassified) > 0:
        summary_text += f"\nMost Common Errors:\n"
        error_counts = misclassified.groupby(['true_type', 'detected_type']).size().sort_values(ascending=False).head(5)
        for (true_t, det_t), count in error_counts.items():
            summary_text += f"  • {true_t} → {det_t}: {count} times\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax4.axis('off')
    
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_image}")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_test.py <image_directory> [output_csv]")
        print("Example: python batch_test.py ./noisy_output")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "detection_results.csv"
    
    # Run batch testing
    df = batch_test_images(image_dir, output_csv)
    
    if df is not None:
        # Create visualization
        visualize_results(df, "detection_results.png")
        
        # Print summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Total images tested: {len(df)}")
        print(f"Correctly classified: {df['correct'].sum()}")
        print(f"Overall accuracy: {(df['correct'].sum() / len(df)) * 100:.1f}%")
        print("="*50)


if __name__ == "__main__":
    main()