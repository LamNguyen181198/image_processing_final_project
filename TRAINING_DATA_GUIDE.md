# Training Data Preparation Guide

## Overview
This system integrates with your existing noise generation pipeline to create labeled training data for ML-based noise detection. The training data has been successfully generated with **71 labeled images** and **23 features per image**.

## Status: ✅ COMPLETE

**Dataset Summary:**
- Total images: 71 (1 clean + 70 noisy)
- Training set: 56 samples (78.9%)
- Test set: 15 samples (21.1%)
- Features per image: 23
- Noise types: 7 (gaussian, salt_pepper, poisson, speckle, uniform, jpeg_artifact, clean)

---

## System Components

### 1. **extract_features.m** (MATLAB)
- Extracts 23 numerical features from each image
- Features include: variance-mean relationships, histogram shape, statistical moments, frequency domain properties
- Used by both the training pipeline and future ML classifier
- **Status**: ✅ Tested and working

### 2. **prepare_training_data.py** (Python)
- Automated end-to-end training data generation
- Uses your existing `noise_gen.py` to create noisy images
- Extracts features using MATLAB and creates labeled CSV dataset
- **Status**: ✅ Successfully generated 71 images with features

### 3. **features_to_csv.m** (MATLAB - Alternative)
- Pure MATLAB batch feature extraction
- Direct alternative if you prefer working in MATLAB
- **Status**: ✅ Used to extract features from all 71 images

### 4. **split_train_test.py** (Python) - NEW
- Splits dataset into training and test sets
- Supports stratified splitting (maintains label distribution)
- Configurable test size and random seed
- **Status**: ✅ Created 80/20 train/test split

### 5. **visualize_training_data.py** (Python) - NEW
- Creates visualizations of dataset distribution
- Generates feature correlation heatmaps
- Produces box plots for key features by noise type
- **Status**: ✅ Generated 4 visualization plots

---

## Quick Start

### ✅ COMPLETED - Training Data Generated

The training dataset has been successfully created:

**Location:**
- Full dataset: `training/training_data_features.csv` (71 samples)
- Training set: `training/training_data_features_train.csv` (56 samples)
- Test set: `training/training_data_features_test.csv` (15 samples)
- Visualizations: `training/visualizations/` directory

**Dataset Statistics:**

Training Set (56 samples):
```
salt_pepper:    12 samples
uniform:        11 samples
jpeg_artifact:  10 samples
speckle:         9 samples
poisson:         8 samples
gaussian:        6 samples
```

Test Set (15 samples):
```
gaussian:        4 samples
jpeg_artifact:   3 samples
speckle:         3 samples
poisson:         2 samples
clean:           1 sample
salt_pepper:     1 sample
uniform:         1 sample
```

### Generating New Training Data (Optional)

If you need to regenerate or create additional training data:

Generate training data with one command:

```bash
# Navigate to training directory
cd "C:\Users\ADMIN\Desktop\Image Processing Final Project\image_processing_final_project\training"

# Generate 20 images per noise type + extract features
python prepare_training_data.py ../pre_transform_image/sample1.jpg --num-per-type 20
```

This will:
1. Create `training_data/` folder with 120+ noisy images (20 × 6 noise types)
2. Extract 23 features from each image using MATLAB
3. Generate `training_data_features.csv` with labeled data

Output CSV structure:
```
filename,label,r2_linear,r2_quadratic,variance_coefficient,...
gaussian_01_sigma5.0.png,gaussian,0.234,0.445,0.123,...
salt_pepper_01_density0.1234.png,salt_pepper,0.876,0.823,0.234,...
```

### Option B: MATLAB Only

```matlab
% 1. Generate noisy images manually or use existing ones

% 2. Extract features to CSV
cd('C:\Users\ADMIN\Desktop\Image Processing Final Project\image_processing_final_project\training')
features_to_csv('../noisy_output', 'training_features.csv')
```

---

## Usage Examples

### Split Dataset into Train/Test

```bash
# 80/20 split (default) - ALREADY DONE
python split_train_test.py training_data_features.csv

# 70/30 split
python split_train_test.py training_data_features.csv --test-size 0.3

# Custom random seed for reproducibility
python split_train_test.py training_data_features.csv --seed 123
```

**Output:**
- `training_data_features_train.csv` - Training set
- `training_data_features_test.csv` - Test set

### Visualize Dataset

```bash
# Visualize training set - ALREADY DONE
python visualize_training_data.py training_data_features_train.csv

# Visualize full dataset
python visualize_training_data.py training_data_features.csv

# Visualize test set
python visualize_training_data.py training_data_features_test.csv
```

**Output:** 4 PNG files in `visualizations/` directory

### Generate Different Dataset Sizes

```bash
# Small dataset for quick testing (10 per type = 60+ images)
python prepare_training_data.py ../pre_transform_image/sample1.jpg --num-per-type 10

# Medium dataset (30 per type = 180+ images)
python prepare_training_data.py ../pre_transform_image/sample1.jpg --num-per-type 30

# Large dataset for better ML performance (50 per type = 300+ images)
python prepare_training_data.py ../pre_transform_image/sample1.jpg --num-per-type 50
```

### Use Custom Output Directory

```bash
python prepare_training_data.py ../pre_transform_image/sample1.jpg \
    --num-per-type 25 \
    --output-dir my_custom_training_set
```

### Extract Features from Existing Images

If you already generated noisy images:

```bash
python prepare_training_data.py \
    --skip-generation \
    --images-dir noisy_output
```

---

## Feature Descriptions

The system extracts **23 features** per image, organized into 6 categories:

### 1. Variance-Mean Relationship (6 features)
| Feature | Description | Useful For |
|---------|-------------|------------|
| `r2_linear` | R² fit quality for linear var~mean model | Poisson noise (linear relationship) |
| `r2_quadratic` | R² fit quality for quadratic var~mean model | Speckle noise (quadratic relationship) |
| `variance_coefficient` | Coefficient of variation in local variances | Additive vs multiplicative noise |
| `linear_slope` | Slope of linear var~mean fit | Poisson intensity dependency |
| `linear_intercept` | Intercept of linear fit | Baseline noise level |
| `quadratic_a` | Quadratic coefficient (a·x²) | Speckle characteristic |

### 2. Histogram Shape (3 features)
| Feature | Description | Useful For |
|---------|-------------|------------|
| `has_central_peak` | Peak in center 40-60% of histogram | Gaussian/Uniform detection |
| `histogram_flatness` | Standard deviation / mean of histogram | Uniform noise (low flatness) |
| `bimodal_extreme_ratio` | Ratio of extreme bins to middle bins | Salt & Pepper (bimodal) |

### 3. Statistical Moments (3 features)
| Feature | Description | Useful For |
|---------|-------------|------------|
| `kurtosis` | "Peakedness" of noise distribution | Gaussian (≈3) vs Uniform (≈-1.2) |
| `skewness` | Asymmetry of distribution | Symmetric vs asymmetric noise |
| `noise_variance` | Variance of noise residual | Overall noise magnitude |

### 4. Global Statistics (2 features)
| Feature | Description | Useful For |
|---------|-------------|------------|
| `var_mean_ratio` | Global variance / mean | Poisson detection (should ≈ mean) |
| `var_mean_squared_ratio` | Variance / mean² | Speckle detection |

### 5. Impulse Detection (3 features)
| Feature | Description | Useful For |
|---------|-------------|------------|
| `salt_pepper_score` | Ratio of extreme (black/white) pixels | Salt & Pepper noise |
| `impulse_ratio` | Pixels different from median filtered | Impulse noise strength |
| `median_diff_variance` | Variance of difference from median | Impulse detection |

### 6. Frequency Domain (6 features)
| Feature | Description | Useful For |
|---------|-------------|------------|
| `dct_dc_energy` | DC coefficient energy (low freq) | Overall image brightness |
| `dct_ac_energy` | AC coefficient energy (high freq) | JPEG compression artifacts |
| `edge_variance` | Variance in edge strength | JPEG blocking artifacts |
| `peak_intensity` | Maximum pixel value | Dynamic range |
| `min_intensity` | Minimum pixel value | Dynamic range |
| `entropy` | Information content | Overall complexity |

---

## Visualizations

The `visualize_training_data.py` script generates 4 key visualizations:

### 1. Label Distribution (`label_distribution.png`)
Bar chart showing the count of samples for each noise type. Helps verify dataset balance.

**Current distribution:**
- JPEG artifacts: 13 samples (most common - mix of old and new generation)
- Salt & Pepper: 13 samples
- Speckle: 12 samples
- Uniform: 12 samples
- Gaussian: 10 samples
- Poisson: 10 samples
- Clean: 1 sample

### 2. Feature Correlation Heatmap (`feature_correlation.png`)
Shows correlations between the top 15 features. Helps identify:
- Redundant features (high correlation)
- Independent feature groups
- Feature engineering opportunities

### 3. Feature Box Plots (`feature_boxplots.png`)
Six box plots showing distribution of key discriminative features by noise type:
- **kurtosis**: Separates Gaussian (high) from Uniform (negative)
- **skewness**: Identifies asymmetric noise patterns
- **noise_variance**: Overall noise strength
- **salt_pepper_score**: Clearly identifies salt & pepper noise
- **impulse_ratio**: Detects impulse-based noise
- **var_mean_ratio**: Distinguishes Poisson noise

### 4. Variance-Mean Scatter Plot (`variance_mean_scatter.png`)
2D scatter plot of R² linear vs R² quadratic fits, color-coded by noise type. Shows:
- Poisson: High linear R², low quadratic R²
- Speckle: High quadratic R², moderate linear R²
- Additive (Gaussian/Uniform): Low both R² values
- Salt & Pepper: Clustered in specific region

**To generate visualizations:**
```bash
python visualize_training_data.py training_data_features_train.csv
```

Output saved to: `visualizations/` directory

---

## Next Steps

### ✅ Completed
- [x] Feature extraction system created
- [x] Training data generated (71 images)
- [x] Features extracted (23 per image)
- [x] Dataset split into train/test
- [x] Visualizations created

### 🎯 Ready For
1. **Train ML Model** - Use the CSV files to train Random Forest or SVM classifier
2. **Model Evaluation** - Test accuracy on the test set
3. **Integration** - Replace threshold-based `detect_noise_type.m` with trained model

---

## Files Generated

```
image_processing_final_project/
├── noisy_output/                          # 71 generated noisy images
│   ├── clean_original.png                 # Clean reference
│   ├── gaussian_01_sigma18.0.png          # 10 Gaussian variations
│   ├── salt_pepper_01_density0.0905.png   # 13 Salt&Pepper variations
│   ├── poisson_01_peak50.png              # 10 Poisson variations
│   ├── speckle_01_var0.100.png            # 12 Speckle variations
│   ├── uniform_01_range20.0.png           # 12 Uniform variations
│   └── jpeg_artifact_01_q36.png           # 13 JPEG artifact variations
│
├── training_data_features.csv             # Full dataset (71 samples × 23 features)
├── training_data_features_train.csv       # Training set (56 samples)
├── training_data_features_test.csv        # Test set (15 samples)
│
├── visualizations/                        # Data visualization plots
│   ├── label_distribution.png             # Bar chart of noise type counts
│   ├── feature_correlation.png            # Heatmap of feature correlations
│   ├── feature_boxplots.png               # Box plots for key features
│   └── variance_mean_scatter.png          # Scatter plot of var-mean relationship
│
├── prepare_training_data.py               # Main training data pipeline
├── split_train_test.py                    # Train/test splitting tool
├── visualize_training_data.py             # Visualization generator
│
└── noise_detecting/
    ├── extract_features.m                 # Feature extraction (23 features)
    └── features_to_csv.m                  # MATLAB batch processing
```

---

## Technical Details

### Feature Extraction Performance
- **Processing time**: ~1-2 seconds per image (MATLAB)
- **Total extraction time**: ~2 minutes for 71 images
- **Memory usage**: Minimal (processes one image at a time)

### Dataset Characteristics
- **Class imbalance**: Moderate (1-13 samples per class)
  - Note: "clean" class has only 1 sample - may need more for better generalization
- **Feature range**: Mixed (some features [0,1], others unbounded)
  - Recommendation: Use StandardScaler or MinMaxScaler before ML training
- **Missing values**: None (all features computed successfully)
- **Outliers**: Present in some features (normal for noise data)

### Train/Test Split Strategy
- **Method**: Random split (stratified splitting not possible due to single "clean" sample)
- **Ratio**: 78.9% train / 21.1% test
- **Random seed**: 42 (reproducible)
- **Preservation**: All noise types represented in test set

### Recommendations for ML Training
1. **Feature scaling**: Use `StandardScaler` to normalize features
2. **Handle class imbalance**: Consider:
   - Generating more "clean" images
   - Using class weights in classifier
   - SMOTE (Synthetic Minority Over-sampling)
3. **Cross-validation**: Use 5-fold CV on training set for model selection
4. **Feature selection**: Consider removing highly correlated features
5. **Model choice**: 
   - Random Forest (recommended): Handles mixed feature ranges well
   - SVM with RBF kernel: Good for non-linear boundaries
   - Gradient Boosting: May overfit with small dataset

---

## Troubleshooting

### MATLAB Not Found
```
Error: 'matlab' is not recognized as an internal or external command
```
**Solution**: Add MATLAB to your system PATH or specify full path to MATLAB executable

### No Images Generated
Check that `noise_gen.py` is working:
```bash
python noise_gen/noise_gen.py pre_transform_image/sample1.jpg -n 5 -t gaussian
```

### Feature Extraction Fails
Test MATLAB feature extraction directly:
```matlab
cd noise_detecting
features = extract_features('../pre_transform_image/sample1.jpg')
```

---

## File Structure

```
image_processing_final_project/
├── training/                              # Training data and ML scripts
│   ├── prepare_training_data.py           # Main training data pipeline
│   ├── split_train_test.py                # Train/test splitting
│   ├── visualize_training_data.py         # Visualization generator
│   ├── extract_features.m                 # Feature extraction (23 features)
│   ├── features_to_csv.m                  # MATLAB batch processing
│   ├── training_data_features.csv         # Full dataset (71 samples)
│   ├── training_data_features_train.csv   # Training set (56 samples)
│   ├── training_data_features_test.csv    # Test set (15 samples)
│   ├── visualizations/                    # 4 distribution plots
│   └── README.md                          # Training directory guide
├── noise_gen/
│   ├── noise_gen.py                       # Noise generation wrapper
│   └── generate_noisy_images.m            # MATLAB noise generation
├── noise_detecting/
│   └── detect_noise_type.m                # Old threshold-based detector
├── pre_transform_image/
│   └── sample1.jpg                        # Clean input image
├── noisy_output/                          # Generated noisy images (71 total)
├── README.md                              # Main documentation
├── TRAINING_DATA_GUIDE.md                 # This file
├── PROGRESS_SUMMARY.md                    # Work completed summary
└── QUICK_REFERENCE.md                     # Quick reference card
```

---

## Command Reference

### Complete Workflow (Already Executed)
```bash
# Navigate to training directory
cd training

# Step 1: Generate training data (10 images per noise type)
python prepare_training_data.py ../pre_transform_image/sample1.jpg --num-per-type 10

# Step 2: Split into train/test sets
python split_train_test.py training_data_features.csv --test-size 0.2

# Step 3: Visualize the data
python visualize_training_data.py training_data_features_train.csv
```

### Alternative: MATLAB-Only Workflow
```matlab
% Navigate to training directory
cd('C:\Users\ADMIN\Desktop\Image Processing Final Project\image_processing_final_project\training')

% Extract features to CSV
features_to_csv('../noisy_output', 'training_features.csv')
```

### Quick Data Inspection
```bash
# View first few rows
Get-Content training_data_features.csv | Select-Object -First 5

# Count samples per noise type
python -c "import pandas as pd; print(pd.read_csv('training_data_features.csv')['label'].value_counts())"

# Get basic statistics
python -c "import pandas as pd; df = pd.read_csv('training_data_features_train.csv'); print(df.describe())"
```

---
