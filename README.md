# Image Processing Final Project: ML-Based Noise Detection

This project provides a **complete workflow** for generating noisy images and detecting noise types using **Machine Learning** instead of traditional threshold-based methods.

## Project Status: 🎯 Training Data Ready

**Current Phase:** Training data preparation complete. Ready for ML model training.

---

## System Overview

### ✅ **Noise Generation (MATLAB)**

Generates realistic noisy images with controlled parameters:

* **Gaussian noise** - Additive white Gaussian (σ = 5-20)
* **Salt & Pepper noise** - Impulse noise (density 0.02-0.5)
* **Poisson noise** - Photon shot noise (peak values: 2-50)
* **Speckle noise** - Multiplicative noise (variance 0.1-0.5)
* **Uniform noise** - Additive uniform (range 20-50)
* **JPEG artifacts** - Compression artifacts (quality 30-70)

### ✅ **Feature Extraction (MATLAB)**

Extracts **23 numerical features** from each image:
- Variance-mean relationships (6 features)
- Histogram shape analysis (3 features)
- Statistical moments (3 features)
- Global statistics (2 features)
- Impulse detection (3 features)
- Frequency domain (6 features)

### ✅ **Training Dataset**

**Generated Dataset:**
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
