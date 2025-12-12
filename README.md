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
- **Python 3.8+** with packages: `tkinter`, `PIL`, `numpy`, `scikit-learn`, `pandas`, `matlab`
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
pip install pillow numpy scikit-learn pandas matplotlib matlabengine
```

3. **Verify MATLAB Engine:**
```python
import matlab.engine
eng = matlab.engine.start_matlab()
print("MATLAB Engine ready!")
```

### Running the Application

**Main GUI Application:**
```bash
python denoise_app.py
```

**Using the Application:**
1. Click **"Load Noisy Image"** to select an image
2. Click **"Detect & Denoise"** to process
3. View results and click **"Save Denoised Image"** if satisfied

---

## 📊 Current Status & Performance

### ✅ **Working Well**
- **Gaussian Noise:** Non-Local Means filter provides excellent noise removal with detail preservation
- **Salt & Pepper Noise:** Adaptive median filter effectively removes impulse noise

### ✅ **Optimally Tuned**
- **Uniform Noise:** Multi-stage bilateral filter with aggressive sharpening provides excellent detail preservation
- **Speckle Noise:** Non-Local Means in log domain with dual-stage detail restoration achieves maximum noise removal while maintaining sharpness

### 🎓 **ML Detection Performance**
- **Model:** Random Forest Classifier (100 trees)
- **Training Accuracy:** ~95%
- **Test Accuracy:** ~80% (10-sample test set)
- **Features:** 29 statistical features extracted from images
- **Training Data:** 61 images (51 train / 10 test split)

---

## 🏗️ System Architecture

### **Detection Pipeline (ML-Based)**
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
| **Gaussian** | Non-Local Means / Bilateral | DegreeOfSmoothing: 0.04-0.08 |
| **Salt & Pepper** | Adaptive Median | Window: 3×3 to 7×7 |
| **Speckle** | NLM in Log Domain + Dual-Stage Detail Restoration | Smoothing: 15×σ, Sharpen: 85% + 25% micro-detail |
| **Uniform** | Bilateral + Multi-Stage Enhancement | Spatial: 0.5-1.2, Intensity: 0.01-0.05, Sharpen: 80-100% + micro + edge + contrast |

---

## 📁 Project Structure

```
image_processing_final_project/
│
├── denoise_app.py              # Main GUI application
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
├── noise_gen/
│   ├── noise_gen.py            # Python noise generation interface
│   └── generate_noisy_images.m # MATLAB noise generation
│
├── training/
│   ├── prepare_training_data.py    # Generate training dataset
│   ├── split_train_test.py         # Train/test split
│   ├── extract_features.m          # MATLAB feature extraction
│   ├── features_to_csv.m           # Export features to CSV
│   ├── training_data_features.csv  # Full dataset
│   ├── training_data_features_train.csv
│   ├── training_data_features_test.csv
│   └── models/
│       ├── train_random_forest.py  # Train ML model
│       ├── predict_noise.py        # Prediction interface
│       └── random_forest_model.pkl # Trained model
│
└── noisy_output/               # Generated noisy images
```

---

## 🔧 Training Your Own Model

### 1. Generate Training Data
```bash
# Generate noisy images (51 training samples)
python training/prepare_training_data.py

# Split into train/test sets
python training/split_train_test.py
```

### 2. Train Model
```bash
cd training/models
python train_random_forest.py
```

### 3. Test Model
```python
from training.models.predict_noise import predict_noise_type

noise_type = predict_noise_type('path/to/test/image.jpg')
print(f"Detected: {noise_type}")
```

---

## 🎨 Noise Generation

Generate custom noisy images:

```python
from noise_gen.noise_gen import add_noise

# Generate specific noise type
add_noise('input.jpg', 'output.jpg', 'gaussian', sigma=10)
add_noise('input.jpg', 'output.jpg', 'salt_pepper', density=0.05)
add_noise('input.jpg', 'output.jpg', 'speckle', variance=0.15)
add_noise('input.jpg', 'output.jpg', 'uniform', range_val=20)
```

**Supported Parameters:**
- **Gaussian:** `sigma` (3-15, default: 10)
- **Salt & Pepper:** `density` (0.01-0.15, default: 0.05)
- **Speckle:** `variance` (0.05-0.25, default: 0.15)
- **Uniform:** `range_val` (10-40, default: 20)

---

## 📈 Performance Metrics

The application provides real-time metrics:
- **Avg. Noise Removed:** Measures denoising effectiveness
- **Noise σ:** Estimated noise standard deviation
- **Processing Time:** ML detection + MATLAB filtering time

---

## 🔬 Technical Details

### **ML Model Specifications**
- **Algorithm:** Random Forest Classifier
- **Parameters:** 100 estimators, max_depth=20, random_state=42
- **Features:** 29 numerical features per image
- **Training Size:** 51 images across 4 noise classes
- **Test Size:** 10 images (stratified split)

### **Denoising Approach**
- **Luminance-Color Separation:** Speckle filter processes LAB color space
- **Adaptive Parameters:** Filter strength adjusts based on estimated noise level
- **Edge Preservation:** Bilateral filters preserve edges while smoothing
- **Detail Enhancement:** Unsharp masking restores sharpness post-filtering

---

## 📝 Recent Updates (Dec 13, 2025)

- ✅ Removed Poisson noise from system (focused on 4 main types)
- ✅ Switched to ML-based detection (superior to rule-based)
- ✅ **Optimally tuned Speckle filter:** Non-Local Means in log domain with 85% + 25% dual-stage detail restoration
- ✅ **Optimally tuned Uniform filter:** Multi-stage enhancement with 80-100% sharpening, micro-detail, edge, and contrast boosts
- ✅ Implemented log-domain processing for multiplicative speckle noise
- ✅ Removed SRAD approach in favor of superior NLM filtering

---

## 🐛 Known Issues

1. **Small Training Set:** 61 images may not capture all noise variations - more training data needed
2. **Grayscale Processing:** Most filters convert RGB to grayscale (except uniform which processes all channels)
3. **Fixed Parameters:** Current tuning optimal for moderate noise levels; extreme cases may benefit from adaptive adjustment
4. **No Mixed Noise Support:** System assumes single noise type per image

---

## 🎯 Future Improvements

### High Priority
- [ ] Increase training dataset to 200-500 images per noise type for better ML generalization
- [ ] Add confidence scores for ML predictions to indicate detection reliability
- [ ] Support batch processing of multiple images
- [ ] Implement mixed noise detection/removal (e.g., Gaussian + Salt & Pepper)

### Advanced Denoising
- [ ] Explore additional noise types (Poisson, JPEG artifacts, compression noise)
- [ ] Implement automated parameter tuning using optimization algorithms
  - Grid search or Bayesian optimization for uniform/speckle parameters
  - Image quality metrics (PSNR, SSIM) as objective functions
  - Per-image adaptive tuning based on noise characteristics
- [ ] Investigate deep learning approaches (DnCNN, FFDNet) for comparison
- [ ] Add real-time preview with adjustable filter strength slider

### Quality Improvements  
- [ ] Develop better metrics for speckle and uniform noise parameter selection
- [ ] Add noise level estimation display
- [ ] Implement quality assessment scoring (no-reference IQA)
- [ ] Support color image denoising without RGB→grayscale conversion

---

## 📚 Documentation

- **[PRESENTATION_DOCS.md](PRESENTATION_DOCS.md)** - Detailed presentation documentation
- **[TRAINING_DATA_GUIDE.md](TRAINING_DATA_GUIDE.md)** - Guide for generating training data
- **[NOISE_DETECTION_METHODOLOGY.md](NOISE_DETECTION_METHODOLOGY.md)** - ML detection approach

---

## 👥 Contributors

Lam Nguyen - [@LamNguyen181198](https://github.com/LamNguyen181198)

---

## 📄 License

This project is for educational purposes as part of an Image Processing course final project.
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
