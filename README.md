# Image Processing Final Project: ML-Based Noise Detection & Denoising

This project provides a **complete ML-powered workflow** for automatic noise detection and optimal denoising of images with various noise types.

## 🎯 Project Overview

An intelligent image denoising application that:
1. **Detects** noise type automatically using Machine Learning (Random Forest)
2. **Applies** optimal denoising filters based on detected noise
3. **Preserves** image details while removing noise effectively

**Supported Noise Types:** Gaussian, Salt & Pepper, Speckle, Uniform

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **MATLAB R2020a+** with Image Processing Toolbox
- **MATLAB Engine for Python** installed

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/LamNguyen181198/image_processing_final_project.git
cd image_processing_final_project
```

2. **Install Python dependencies:**
```bash
pip install pillow numpy scikit-learn pandas matplotlib seaborn matlabengine
```

3. **Verify MATLAB Engine:**
```bash
python -c "import matlab.engine; eng = matlab.engine.start_matlab(); print('MATLAB Engine ready!')"
```

### Running the Application

**🎯 Main GUI Application (Recommended):**
```bash
python denoise_app.py
```

**Using the Application:**
1. Click **"Load Noisy Image"** to select an image
2. Click **"Detect & Denoise"** to process automatically
3. View results with performance metrics
4. Click **"Save Denoised Image"** if satisfied

---

## 📊 Performance & Status

### ✅ **Current Implementation**
- **Gaussian Noise:** Non-Local Means filter with excellent detail preservation
- **Salt & Pepper Noise:** Adaptive median filter for impulse noise removal
- **Uniform Noise:** Multi-stage bilateral filter with aggressive sharpening
- **Speckle Noise:** Non-Local Means in log domain with dual-stage detail restoration

### 🎓 **ML Detection Performance**
- **Model:** Random Forest Classifier (100 trees, max_depth=20)
- **Training Accuracy:** ~95%
- **Test Accuracy:** ~80%
- **Features:** 29 statistical features per image
- **Training Data:** 61 images across 4 noise types

## 🏗️ System Architecture

### **Detection Pipeline**
```
Input Image → Feature Extraction (MATLAB) → Random Forest Classifier → Noise Type
```

**Feature Categories (29 features):**
- Variance-mean relationships (6 features)
- Histogram analysis (4 features)
- Statistical moments (4 features)
- Global statistics (3 features)
- Impulse detection (3 features)
- Frequency domain analysis (6 features)
- Noise characteristics (3 features)

### **Denoising Pipeline**
```
Detected Noise Type → Optimal Filter Selection → MATLAB Processing → Denoised Image
```

**Filter Strategy:**

| Noise Type | Filter Used | Key Parameters |
|------------|-------------|----------------|
| **Gaussian** | Non-Local Means | DegreeOfSmoothing: 0.04-0.08 |
| **Salt & Pepper** | Adaptive Median | Window: 3×3 to 7×7 |
| **Speckle** | NLM in Log Domain + Dual-Stage Detail Restoration | Smoothing: 15×σ, Sharpen: 85% + 25% micro-detail |
| **Uniform** | Bilateral + Multi-Stage Enhancement | Spatial: 0.5-1.2, Intensity: 0.01-0.05, Sharpen: 80-100% + micro + edge + contrast |

## 📁 Project Structure

```
image_processing_final_project/
│
├── denoise_app.py              # Main GUI application (START HERE)
├── README.md                   # This file
├── PRESENTATION_DOCS.md        # Presentation documentation
│
├── denoise/
│   ├── denoise_image.py        # Python interface for denoising
│   └── denoise_filters.m       # MATLAB denoising filters
│
├── noise_detecting/
│   ├── detect_noise.py         # Python ML detection interface
│   └── detect_noise_type.m     # Legacy MATLAB detection (unused)
│
├── pre_transform_image/
│   └── sample1.jpg             # Clean image for training data generation
│
├── training/
│   ├── prepare_training_data.py    # Generate training dataset
│   ├── split_train_test.py         # Train/test split
│   ├── visualize_training_data.py  # Visualize dataset distribution
│   ├── extract_features.m          # MATLAB feature extraction
│   ├── features_to_csv.m           # Export features to CSV
│   ├── training_data_features.csv  # Full dataset
│   ├── training_data_features_train.csv
│   ├── training_data_features_test.csv
│   └── models/
│       ├── train_random_forest.py  # Train ML model
│       ├── predict_noise.py        # Prediction interface
│       ├── random_forest_model.pkl # Pre-trained model ✅
│       └── *.png                   # Model performance visualizations
│
└── noisy_output/               # Generated noisy images (for training)
```

---

## 🔧 Advanced Usage

### Retraining the Model (Optional)

The model is already trained, but you can retrain with custom data:

**Step 1: Generate Training Data**
```bash
python training/prepare_training_data.py pre_transform_image/sample1.jpg --num-per-type 10
```

**Step 2: Split Dataset**
```bash
python training/split_train_test.py training/training_data_features.csv --test-size 0.2
```

**Step 3: Train Model**
```bash
cd training/models
python train_random_forest.py
cd ../..
```

**Step 4: Visualize Data (Optional)**
```bash
python training/visualize_training_data.py training/training_data_features_train.csv
```

### Testing Noise Detection Programmatically

```python
from training.models.predict_noise import predict_noise_type

# Detect noise type
noise_type = predict_noise_type('path/to/noisy/image.jpg')
print(f"Detected: {noise_type}")  # Returns: gaussian, salt_pepper, speckle, or uniform
```

### Batch Processing

```python
from training.models.predict_noise import predict_noise_type
from denoise.denoise_image import denoise_image
from pathlib import Path

# Process multiple images
image_dir = Path('test_images')
for img_path in image_dir.glob('*.jpg'):
    noise_type = predict_noise_type(str(img_path))
    output_path = f'denoised_{img_path.name}'
    denoise_image(str(img_path), output_path, noise_type)
    print(f"Processed {img_path.name} - Detected: {noise_type}")
```

---

## 📈 Performance Metrics

The application displays real-time metrics:
- **Detected Noise Type:** ML classification result
- **Avg. Noise Removed:** Denoising effectiveness measure
- **Noise σ:** Estimated noise standard deviation
- **Processing Time:** Total time for detection + filtering

## 🐛 Known Issues & Limitations

1. **Training Dataset Size:** 61 images may not capture all noise variations
2. **Grayscale Processing:** Most filters convert RGB to grayscale (except uniform)
3. **Fixed Parameters:** Optimal for moderate noise; extreme cases may need adjustment
4. **Single Noise Type:** No support for mixed noise (e.g., Gaussian + Salt & Pepper)
5. **MATLAB Dependency:** Requires MATLAB Engine for Python

---

## 🎯 Future Improvements

### High Priority
- [ ] Increase training dataset to 200-500 images per noise type for better ML generalization
- [ ] Add confidence scores for ML predictions to indicate detection reliability
- [ ] Support batch processing of multiple images
- [ ] Implement mixed noise detection/removal (e.g., Gaussian + Salt & Pepper)

### Advanced Denoising
- [ ] Explore additional noise types (Poisson, JPEG artifacts, compression noise)
## 🎯 Future Improvements

### High Priority
- [ ] Expand training dataset to 200-500 images per noise type
- [ ] Add ML prediction confidence scores
- [ ] Implement batch processing GUI
- [ ] Support mixed noise detection/removal

### Advanced Features
- [ ] Additional noise types (Poisson, JPEG compression artifacts)
- [ ] Automated parameter tuning (Grid search, Bayesian optimization)
- [ ] Deep learning comparison (DnCNN, FFDNet)
- [ ] Real-time preview with adjustable filter strength
- [ ] Color image denoising without RGB→grayscale conversion
- [ ] Quality assessment metrics (PSNR, SSIM, no-reference IQA)
- **Total samples:** 71 images
- **Training set:** 56 samples (78.9%)
- **Test set:** 15 samples (21.1%)
- **Features per sample:** 23
- **Noise types:** 7 classes

**Files:**
- `training/training_data_features.csv` - Full dataset
- `training/training_data_features_train.csv` - Training set
- `training/training_data_features_test.csv` - Test set
- `training/visualizations/` - Data distribution plots

### 🔄 **ML Classifier (Next Step)**

Will replace the old threshold-based `detect_noise_type.m` with:
- Random Forest or SVM classifier
- Trained on extracted features
- Expected accuracy: 85-95%

---

# 📦 Requirements

## MATLAB

* MATLAB R2022a or newer (Trial License works)
* **Image Processing Toolbox** (recommended)

If you do *not* have IPT, you can replace `im2double` / `im2uint8` with manual scaling.

## Python

Required packages:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

All other modules used are standard library.

---

# 🚀 Quick Start Guide

## Phase 1: Training Data Preparation (✅ COMPLETED)

### Step 1: Generate Noisy Images + Extract Features
```bash
cd training
python prepare_training_data.py ../pre_transform_image/sample1.jpg --num-per-type 10
```
**Output:** 71 images in `noisy_output/` + `training/training_data_features.csv`

### Step 2: Split into Train/Test Sets
```bash
cd training
python split_train_test.py training_data_features.csv --test-size 0.2
```
**Output:** `training/training_data_features_train.csv` (56) + `training/training_data_features_test.csv` (15)

### Step 3: Visualize Dataset
```bash
cd training
python visualize_training_data.py training_data_features_train.csv
```
**Output:** 4 plots in `training/visualizations/` directory

## Phase 2: ML Model Training (🔄 NEXT)

Coming soon: Random Forest classifier training and evaluation

---

---

# 📊 Training Dataset Summary

**Current Status:** ✅ Generated and ready for ML training

| Metric | Value |
|--------|-------|
| Total samples | 71 |
| Training samples | 56 (78.9%) |
| Test samples | 15 (21.1%) |
| Features per sample | 23 |
| Noise types | 7 classes |

**Label Distribution (Training Set):**
- Salt & Pepper: 12 samples
- Uniform: 11 samples
- JPEG artifacts: 10 samples
- Speckle: 9 samples
- Poisson: 8 samples
- Gaussian: 6 samples

**Label Distribution (Test Set):**
- Gaussian: 4 samples
- JPEG artifacts: 3 samples
- Speckle: 3 samples
- Poisson: 2 samples
- Clean, Salt & Pepper, Uniform: 1 sample each

**Feature Categories:**
1. Variance-mean relationships (6 features)
2. Histogram shape (3 features)
3. Statistical moments (3 features)
4. Global statistics (2 features)
5. Impulse detection (3 features)
6. Frequency domain (6 features)

See `TRAINING_DATA_GUIDE.md` for detailed feature descriptions.

---

# 📁 Project Files

## Generated Data Files

| File | Description | Samples |
|------|-------------|---------|
| `training/training_data_features.csv` | Full dataset with labels | 71 |
| `training/training_data_features_train.csv` | Training set | 56 |
| `training/training_data_features_test.csv` | Test set | 15 |
| `noisy_output/*.png` | Generated noisy images | 71 |
| `training/visualizations/*.png` | Data distribution plots | 4 |

## Python Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `training/prepare_training_data.py` | Generate images + extract features | ✅ Used |
| `training/split_train_test.py` | Split train/test sets | ✅ Used |
| `training/visualize_training_data.py` | Create visualizations | ✅ Used |
| `batch_test.py` | Test old threshold detector | 🔄 Legacy |

## MATLAB Functions

| Function | Purpose | Location |
|----------|---------|----------|
| `generate_noisy_images.m` | Generate noisy images | `noise_gen/` |
| `extract_features.m` | Extract 23 features from image | `training/` |
| `features_to_csv.m` | Batch feature extraction | `training/` |
| `detect_noise_type.m` | Old threshold-based detector | `noise_detecting/` (legacy) |

---

# 🔍 Old Threshold-Based Detection (Legacy)

The original threshold-based approach is still available but will be replaced by ML:

```matlab
out = detect_noise_type('noisy_output/salt_pepper_01_density0.12.png');
```

Returns one of: `gaussian`, `salt_pepper`, `jpeg_artifact`, `poisson`, `speckle`, `uniform`, `none`

**Issues with threshold approach:**
- Fixed thresholds don't generalize across different images
- Cascading logic creates conflicts between noise types
- Accuracy issues, especially with Gaussian vs Uniform and Poisson vs Speckle

**ML approach advantages:**
- Learns optimal decision boundaries from data
- Handles feature correlations automatically
- Expected 85-95% accuracy vs ~60-70% with thresholds

---

# 🧪 Manual Noise Generation (MATLAB)

You can still generate noisy images directly in MATLAB:

```matlab
generate_noisy_images('path/to/image.jpg', 'noisy_output', 5, 'all');
```

Noise types available: `'all'`, `'gaussian'`, `'salt_pepper'`, `'poisson'`, `'speckle'`, `'uniform'`, `'jpeg_artifacts'`

Generated filenames include metadata (e.g., `salt_pepper_03_density0.1829.png`).

---

# ⚠ Notes

* JPEG artifact detection requires MATLAB’s `blockproc` and `dct2`.
* Poisson noise generation uses physically meaningful photon count scaling.
* Detection scripts run MATLAB in **batch mode**, so MATLAB must support `-batch`.

---

# 📁 Project Structure

```
project/
│
├── noise_detecting/
│   ├── detect_noise_type.m
│   └── batch_test.py
│
├── noise_gen/
│   └── MATLAB noise generation scripts
│
├── noisy_output/        # Generated images
└── README.md            # This file
```

---

# 📝 Author

Lam Nguyen
2025
