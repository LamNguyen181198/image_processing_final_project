# MATLAB Noise Generation Project

This project provides a **Python wrapper** to generate multiple noisy images using MATLAB. It supports Gaussian, salt & pepper, Poisson, speckle, uniform, and JPEG artifact noise types.

---

## **Requirements**

* **MATLAB R2025b** (or compatible version)
* **Image Processing Toolbox** (required for `im2double` and `im2uint8`)
* Python 3.x
* Packages: `argparse`, `pathlib`, `shutil`, `subprocess`, `sys` (all standard)

---

## **Installing Image Processing Toolbox**

### **1. Check if Toolbox is Installed**

Open MATLAB and run:

```matlab
ver
license('test', 'Image_Toolbox')  % Returns 1 if installed, 0 otherwise
```

---

### **2. Install via MATLAB Add-On Explorer (GUI)**

1. Open MATLAB.
2. Go to **Home → Add-Ons → Get Add-Ons**.
3. Search for **Image Processing Toolbox**.
4. Click **Install** and follow prompts (requires a MathWorks account and license).

---

### **3. Install via MATLAB Command Line (Optional)**

1. Download the `.mltbx` installer from [MathWorks](https://www.mathworks.com/products/image.html).
2. In MATLAB, run:

```matlab
matlab.addons.install('C:\Path\To\Image_Processing_Toolbox.mltbx')
```

---

### **4. Verify Installation**

After installation, run:

```matlab
license('test', 'Image_Toolbox')  % Should return 1
```

---

## **Running the Python Wrapper**

```bash
python noise_gen/noise_gen.py <input_image> -n <num_images> -t <noise_type> --matlab-path "<path_to_matlab_exe>"
```

* `<input_image>`: Path to the clean image
* `-n <num_images>`: Number of images per noise type (default: 5)
* `-t <noise_type>`: Noise type (`all`, `gaussian`, `salt_pepper`, `poisson`, `speckle`, `uniform`; default: `all`)
* `--matlab-path`: Optional path to MATLAB executable (default searches system PATH)

**Example:**

```bash
python noise_gen/noise_gen.py ./pre_transform_image/sample1.jpg -n 5 -t gaussian --matlab-path "C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
```

---

## **Alternative (No Toolbox Required)**

If you do not have the Image Processing Toolbox, the MATLAB functions can be modified to replace `im2double` and `im2uint8` with native conversions:

```matlab
img_double = double(img) / 255;        % instead of im2double
noisy = uint8(noisy_double * 255);     % instead of im2uint8
```

This allows the code to run on **any MATLAB installation**.

---

## **Output**

* All noisy images are saved in `noisy_output` (relative to the input image location by default).
* The original clean image is saved as `clean_original.png`.

---

## **Notes**

* Make sure MATLAB can run in **batch mode** (non-interactive) to avoid hanging when called from Python.
* Use flags `-nojvm -nodisplay -nosplash -batch` for headless execution.
* Ensure your license includes the Image Processing Toolbox if using `im2double` and `im2uint8`.

---

**Author:** [Your Name]
**Date:** 2025
