# MATLAB Noise Generation and Detection Project

This project provides a **complete workflow** for generating noisy images in MATLAB and detecting the type of noise using a hybrid MATLAB + Python pipeline.

It now supports:

### ✅ **Noise Generation (MATLAB)**

* Gaussian noise
* Salt & Pepper noise
* Poisson (photon shot) noise
* Speckle noise
* Uniform noise
* JPEG compression artifacts

### ✅ **Noise Type Detection (MATLAB)**

Supports classification of:

* `gaussian`
* `salt_pepper`
* `jpeg_artifact`
* `none`

Detection uses:

* High-frequency variance / kurtosis (Gaussian)
* Blockiness + DCT peakiness (JPEG)
* Robust 3-metric impulse detection (Salt & Pepper)

### ✅ **Batch Testing + Visualization (Python)**

Includes a Python script to:

* Run MATLAB detection on every generated image
* Extract ground-truth noise type from filenames
* Produce a CSV report
* Generate accuracy plots + confusion matrix

---

# 📦 Requirements

## MATLAB

* MATLAB R2022a or newer
* **Image Processing Toolbox** (recommended)

If you do *not* have IPT, you can replace `im2double` / `im2uint8` with manual scaling.

## Python

Requires:

```
pandas
matplotlib
seaborn
```

All other modules used are standard library.

---

# 🚀 Running Noise Generation (MATLAB)

```
generate_noisy_images('path/to/image.jpg', 'noisy_output', 5, 'all');
```

This generates noisy variants of the image and saves them into `noisy_output/`.

Noise types available:

```
'all'
'gaussian'
'salt_pepper'
'poisson'
'speckle'
'uniform'
'jpeg_artifacts'
```

Generated filenames include metadata (e.g., `salt_pepper_03_density0.1829.png`).

---

# 🔍 Running Noise Detection (MATLAB)

```
out = detect_noise_type('noisy_output/salt_pepper_01_density0.12.png');
```

Returns one of:

* `gaussian`
* `salt_pepper`
* `jpeg_artifact`
* `none`

This version includes:

* Improved low-density salt-and-pepper detection
* Multi-metric voting (impulse ratio + extreme pixels + residual ratio)
* Protection against false JPEG/Gaussian triggers

---

# 🧪 Batch Testing (Python)

Run:

```
python batch_test.py noisy_output
```

This will:

* Run MATLAB detection for each PNG in the directory
* Compare detected vs ground truth
* Save `detection_results.csv`
* Generate `detection_results.png` (accuracy plots + confusion matrix)

---

# 📊 Visualization Output

The generated PNG includes:

* Overall accuracy
* Accuracy by noise type
* Confusion matrix
* Most common misclassifications

Useful for debugging threshold selection.

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
