# Quick Reference Card

## 📂 File Locations

**Training Data:**
- Full dataset: `training/training_data_features.csv` (71 samples)
- Training set: `training/training_data_features_train.csv` (56 samples)
- Test set: `training/training_data_features_test.csv` (15 samples)

**Generated Images:**
- Location: `noisy_output/` (71 PNG files)

**Visualizations:**
- Location: `training/visualizations/` (4 PNG files)

**Documentation:**
- Main: `README.md`
- Training guide: `TRAINING_DATA_GUIDE.md`
- Progress: `PROGRESS_SUMMARY.md`
- This card: `QUICK_REFERENCE.md`

---

## 🚀 Common Commands

### Generate New Training Data
```bash
# From training directory
cd training

# Generate 10 images per noise type
python prepare_training_data.py ../pre_transform_image/sample1.jpg --num-per-type 10

# Generate 30 images per noise type (larger dataset)
python prepare_training_data.py ../pre_transform_image/sample1.jpg --num-per-type 30
```

### Split Dataset
```bash
# 80/20 split (default)
python split_train_test.py training_data_features.csv

# 70/30 split
python split_train_test.py training_data_features.csv --test-size 0.3
```

### Visualize Data
```bash
# Visualize training set
python visualize_training_data.py training_data_features_train.csv

# Visualize test set
python visualize_training_data.py training_data_features_test.csv
```

### Extract Features (MATLAB Only)
```matlab
cd training
features_to_csv('../noisy_output', 'training_features.csv')
```

---

## 📊 Dataset Summary

**Total:** 71 images  
**Train:** 56 (78.9%)  
**Test:** 15 (21.1%)  
**Features:** 23 per image  

**Train Distribution:**
- salt_pepper: 12
- uniform: 11
- jpeg_artifact: 10
- speckle: 9
- poisson: 8
- gaussian: 6

**Test Distribution:**
- gaussian: 4
- jpeg_artifact: 3
- speckle: 3
- poisson: 2
- clean, salt_pepper, uniform: 1 each

---

## 🔍 Feature Categories (23 total)

1. **Variance-Mean** (6): r2_linear, r2_quadratic, variance_coefficient, linear_slope, linear_intercept, quadratic_a
2. **Histogram** (3): has_central_peak, histogram_flatness, bimodal_extreme_ratio
3. **Moments** (3): kurtosis, skewness, noise_variance
4. **Global** (2): var_mean_ratio, var_mean_squared_ratio
5. **Impulse** (3): salt_pepper_score, impulse_ratio, median_diff_variance
6. **Frequency** (6): dct_dc_energy, dct_ac_energy, edge_variance, peak_intensity, min_intensity, entropy

---

## 🎯 Next Steps Checklist

- [ ] Review visualizations in `visualizations/`
- [ ] Check training data quality in CSV files
- [ ] Install ML packages: `pip install scikit-learn`
- [ ] Train Random Forest classifier
- [ ] Evaluate on test set
- [ ] Create inference pipeline
- [ ] Replace old `detect_noise_type.m`

---

## 🐛 Troubleshooting

**MATLAB not found?**
→ Add MATLAB to system PATH

**Import errors?**
→ `pip install pandas scikit-learn matplotlib seaborn`

**Feature extraction fails?**
→ Test: `matlab -batch "features = extract_features('image.png')"`

**Unicode errors on Windows?**
→ Already fixed in latest version (uses [OK]/[X] instead of ✓/✗)

---

## 📚 Where to Find Details

| Topic | Document |
|-------|----------|
| Project overview | `README.md` |
| Feature descriptions | `TRAINING_DATA_GUIDE.md` (Section: Feature Descriptions) |
| Usage examples | `TRAINING_DATA_GUIDE.md` (Section: Usage Examples) |
| What was done | `PROGRESS_SUMMARY.md` |
| Visualizations explained | `TRAINING_DATA_GUIDE.md` (Section: Visualizations) |
| File structure | `README.md` (Section: 📁 Project Files) |
| Next steps | All docs (Section: Next Steps) |

---

**Last Updated:** December 10, 2025  
**Status:** Phase 1 Complete - Ready for ML Training
