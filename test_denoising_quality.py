#!/usr/bin/env python3
"""
Test denoising quality improvements

This script tests the updated denoising filters to ensure:
1. Denoised images are better quality than noisy inputs
2. PSNR and SSIM improvements are achieved
3. Brightness/intensity is preserved (especially for Poisson)
4. Visual quality is acceptable

Usage:
    python test_denoising_quality.py
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Try to import scikit-image metrics, fallback to manual implementation
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    USE_SKIMAGE = True
except ImportError:
    print("Warning: scikit-image not found, using manual PSNR/SSIM implementation")
    USE_SKIMAGE = False

# Import denoising function
sys.path.append(str(Path(__file__).parent / 'denoise'))
from denoise.denoise_image import denoise_image


def manual_psnr(img1, img2, data_range=255):
    """Manual PSNR calculation"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def manual_ssim(img1, img2, data_range=255):
    """Simplified SSIM calculation (single-scale)"""
    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    
    # Mean
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Variance and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    return numerator / denominator


def calculate_metrics(clean_img, noisy_img, denoised_img):
    """Calculate PSNR and SSIM metrics"""
    # Convert to numpy arrays if needed
    if isinstance(clean_img, Image.Image):
        clean_img = np.array(clean_img)
    if isinstance(noisy_img, Image.Image):
        noisy_img = np.array(noisy_img)
    if isinstance(denoised_img, Image.Image):
        denoised_img = np.array(denoised_img)
    
    # Ensure grayscale
    if len(clean_img.shape) == 3:
        clean_img = np.mean(clean_img, axis=2)
    if len(noisy_img.shape) == 3:
        noisy_img = np.mean(noisy_img, axis=2)
    if len(denoised_img.shape) == 3:
        denoised_img = np.mean(denoised_img, axis=2)
    
    # Calculate metrics using scikit-image or manual implementation
    if USE_SKIMAGE:
        noisy_psnr = psnr(clean_img, noisy_img, data_range=255)
        denoised_psnr = psnr(clean_img, denoised_img, data_range=255)
        noisy_ssim = ssim(clean_img, noisy_img, data_range=255)
        denoised_ssim = ssim(clean_img, denoised_img, data_range=255)
    else:
        noisy_psnr = manual_psnr(clean_img, noisy_img, data_range=255)
        denoised_psnr = manual_psnr(clean_img, denoised_img, data_range=255)
        noisy_ssim = manual_ssim(clean_img, noisy_img, data_range=255)
        denoised_ssim = manual_ssim(clean_img, denoised_img, data_range=255)
    
    # Brightness preservation (mean intensity difference)
    noisy_brightness = np.mean(noisy_img)
    denoised_brightness = np.mean(denoised_img)
    brightness_diff = abs(denoised_brightness - noisy_brightness)
    brightness_preserved = brightness_diff < 10  # Within 10 gray levels
    
    return {
        'noisy_psnr': noisy_psnr,
        'denoised_psnr': denoised_psnr,
        'psnr_improvement': denoised_psnr - noisy_psnr,
        'noisy_ssim': noisy_ssim,
        'denoised_ssim': denoised_ssim,
        'ssim_improvement': denoised_ssim - noisy_ssim,
        'brightness_diff': brightness_diff,
        'brightness_preserved': brightness_preserved
    }


def test_denoising_on_image(noisy_path, clean_path, output_dir):
    """Test denoising on a single image"""
    noisy_path = Path(noisy_path)
    clean_path = Path(clean_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Testing: {noisy_path.name}")
    print(f"{'='*80}")
    
    # Load images
    clean_img = np.array(Image.open(clean_path).convert('L'))
    noisy_img = np.array(Image.open(noisy_path).convert('L'))
    
    # Denoise
    denoised_output = output_dir / f"denoised_{noisy_path.name}"
    denoised_array, noise_type = denoise_image(
        str(noisy_path), 
        output_path=str(denoised_output),
        use_ml=False,  # Use threshold-based for now
        verbose=True
    )
    
    if denoised_array is None:
        print("ERROR: Denoising failed!")
        return None
    
    # Convert to uint8 for metrics
    denoised_img = (np.clip(denoised_array, 0, 1) * 255).astype(np.uint8)
    
    # Calculate metrics
    metrics = calculate_metrics(clean_img, noisy_img, denoised_img)
    
    # Print results
    print(f"\n📊 Quality Metrics:")
    print(f"  Noisy PSNR:      {metrics['noisy_psnr']:.2f} dB")
    print(f"  Denoised PSNR:   {metrics['denoised_psnr']:.2f} dB")
    print(f"  PSNR Improvement: {metrics['psnr_improvement']:+.2f} dB")
    print(f"")
    print(f"  Noisy SSIM:      {metrics['noisy_ssim']:.4f}")
    print(f"  Denoised SSIM:   {metrics['denoised_ssim']:.4f}")
    print(f"  SSIM Improvement: {metrics['ssim_improvement']:+.4f}")
    print(f"")
    print(f"  Brightness diff:  {metrics['brightness_diff']:.2f}")
    print(f"  Brightness preserved: {'✓ YES' if metrics['brightness_preserved'] else '✗ NO'}")
    
    # Quality check
    is_improvement = (
        metrics['psnr_improvement'] > 0 or 
        metrics['ssim_improvement'] > 0.01
    )
    
    if is_improvement:
        print(f"\n✅ PASS: Denoising improved image quality")
    else:
        print(f"\n❌ FAIL: Denoising did NOT improve quality")
    
    # Create comparison visualization
    create_comparison_plot(
        clean_img, noisy_img, denoised_img, 
        noise_type, metrics,
        output_dir / f"comparison_{noisy_path.stem}.png"
    )
    
    metrics['noise_type'] = noise_type
    metrics['filename'] = noisy_path.name
    metrics['passed'] = is_improvement
    
    return metrics


def create_comparison_plot(clean, noisy, denoised, noise_type, metrics, output_path):
    """Create side-by-side comparison visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(clean, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Clean Original')
    axes[0].axis('off')
    
    axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Noisy ({noise_type})\nPSNR: {metrics["noisy_psnr"]:.2f} dB')
    axes[1].axis('off')
    
    axes[2].imshow(denoised, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'Denoised\nPSNR: {metrics["denoised_psnr"]:.2f} dB '
                      f'({metrics["psnr_improvement"]:+.2f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {output_path.name}")


def main():
    """Main test function"""
    project_root = Path(__file__).parent
    noisy_dir = project_root / "noisy_output"
    clean_image = project_root / "noisy_output" / "clean_original.png"
    output_dir = project_root / "test_results"
    
    if not noisy_dir.exists():
        print(f"ERROR: Noisy images directory not found: {noisy_dir}")
        print("Please generate noisy images first using noise_gen.py")
        return
    
    if not clean_image.exists():
        print(f"ERROR: Clean reference image not found: {clean_image}")
        return
    
    # Find all noisy images (exclude clean original)
    noisy_images = [
        f for f in noisy_dir.glob("*.png") 
        if f.name != "clean_original.png"
    ]
    
    if not noisy_images:
        print(f"ERROR: No noisy images found in {noisy_dir}")
        return
    
    print(f"\n🧪 Testing Denoising Quality")
    print(f"{'='*80}")
    print(f"Found {len(noisy_images)} noisy images to test")
    print(f"Clean reference: {clean_image.name}")
    print(f"Output directory: {output_dir}")
    
    # Test each noise type (sample a few from each category)
    noise_types = {
        'gaussian': [], 'salt_pepper': [], 'poisson': [],
        'speckle': [], 'uniform': []
    }
    
    for img_path in noisy_images:
        name = img_path.stem
        if 'gaussian' in name:
            noise_types['gaussian'].append(img_path)
        elif 'salt_pepper' in name:
            noise_types['salt_pepper'].append(img_path)
        elif 'poisson' in name:
            noise_types['poisson'].append(img_path)
        elif 'speckle' in name:
            noise_types['speckle'].append(img_path)
        elif 'uniform' in name:
            noise_types['uniform'].append(img_path)
    
    # Test 2 images from each noise type
    test_images = []
    for noise_type, images in noise_types.items():
        if images:
            # Take first and last (light and heavy noise)
            test_images.append(images[0])
            if len(images) > 1:
                test_images.append(images[-1])
    
    print(f"\nTesting {len(test_images)} representative images...\n")
    
    # Run tests
    all_results = []
    for img_path in test_images:
        result = test_denoising_on_image(img_path, clean_image, output_dir)
        if result:
            all_results.append(result)
    
    # Summary report
    print(f"\n\n{'='*80}")
    print(f"📋 TEST SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r['passed'])
    
    print(f"\nTotal tests: {total_tests}")
    print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Failed: {total_tests - passed_tests}")
    
    # Per noise type summary
    print(f"\n📊 Results by Noise Type:")
    for noise_type in ['gaussian', 'salt_pepper', 'poisson', 'speckle', 'uniform']:
        type_results = [r for r in all_results if r['noise_type'] == noise_type]
        if type_results:
            avg_psnr_improvement = np.mean([r['psnr_improvement'] for r in type_results])
            avg_ssim_improvement = np.mean([r['ssim_improvement'] for r in type_results])
            passed = sum(1 for r in type_results if r['passed'])
            total = len(type_results)
            
            print(f"\n  {noise_type.upper()}:")
            print(f"    Tests: {passed}/{total} passed")
            print(f"    Avg PSNR improvement: {avg_psnr_improvement:+.2f} dB")
            print(f"    Avg SSIM improvement: {avg_ssim_improvement:+.4f}")
    
    # Brightness preservation check
    brightness_issues = [r for r in all_results if not r['brightness_preserved']]
    if brightness_issues:
        print(f"\n⚠️  Brightness preservation issues detected in:")
        for r in brightness_issues:
            print(f"    - {r['filename']} ({r['noise_type']}): "
                  f"{r['brightness_diff']:.2f} difference")
    else:
        print(f"\n✅ All images preserved brightness within tolerance")
    
    print(f"\n{'='*80}")
    print(f"Test results and visualizations saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
