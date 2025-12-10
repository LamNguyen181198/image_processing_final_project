# Training Directory

This directory contains all files related to training data preparation and machine learning model training.

## 📁 Contents

### Python Scripts
- **prepare_training_data.py** - Generate noisy images and extract features
- **split_train_test.py** - Split dataset into train/test sets
- **visualize_training_data.py** - Create data visualizations

### MATLAB Functions
- **extract_features.m** - Extract 23 features from a single image
- **features_to_csv.m** - Batch process images and save to CSV

### Datasets
- **training_data_features.csv** - Full dataset (71 samples)
- **training_data_features_train.csv** - Training set (56 samples)
- **training_data_features_test.csv** - Test set (15 samples)

### Visualizations
- **visualizations/** - Directory containing 4 visualization plots
  - label_distribution.png
  - feature_correlation.png
  - feature_boxplots.png
  - variance_mean_scatter.png

---

## 🚀 Quick Start

### 1. Generate Training Data
```bash
# From project root
cd training
python prepare_training_data.py ../pre_transform_image/sample1.jpg --num-per-type 10
```

### 2. Split Dataset
```bash
python split_train_test.py training_data_features.csv --test-size 0.2
```

### 3. Visualize Data
```bash
python visualize_training_data.py training_data_features_train.csv
```

### 4. Extract Features (MATLAB Alternative)
```matlab
% From training directory
cd('C:\Users\ADMIN\Desktop\Image Processing Final Project\image_processing_final_project\training')
features_to_csv('../noisy_output', 'training_data_features.csv')
```

---

## 📊 Dataset Summary

**Current Dataset:**
- Total: 71 images with 23 features each
- Training: 56 samples (78.9%)
- Test: 15 samples (21.1%)
- Noise types: 7 classes

**Training Set Distribution:**
```
salt_pepper:    12
uniform:        11
jpeg_artifact:  10
speckle:         9
poisson:         8
gaussian:        6
```

**Test Set Distribution:**
```
gaussian:        4
jpeg_artifact:   3
speckle:         3
poisson:         2
clean, salt_pepper, uniform: 1 each
```

---

## 🔧 Feature Extraction

The system extracts **23 numerical features** organized into 6 categories:

1. **Variance-Mean Relationships** (6 features)
2. **Histogram Shape** (3 features)
3. **Statistical Moments** (3 features)
4. **Global Statistics** (2 features)
5. **Impulse Detection** (3 features)
6. **Frequency Domain** (6 features)

See `../TRAINING_DATA_GUIDE.md` for detailed feature descriptions.

---

## 🎯 Next Steps

After training data is prepared:

1. Train ML classifier (Random Forest/SVM)
2. Evaluate model on test set
3. Save trained model
4. Create inference pipeline
5. Replace old threshold-based detector

---

## 📝 Usage Notes

- All Python scripts work from this directory
- MATLAB functions can be called from any directory (they use absolute paths)
- Generated noisy images are stored in `../noisy_output/`
- Visualizations are saved in `visualizations/` subdirectory

---

**For detailed documentation, see:**
- Main documentation: `../README.md`
- Training guide: `../TRAINING_DATA_GUIDE.md`
- Progress summary: `../PROGRESS_SUMMARY.md`
