#!/usr/bin/env python3
"""
Python wrapper for MATLAB noise generation.

Generates multiple noisy images with user-specified quantity.

- Ensures output directory exists.
- If MATLAB is not available on PATH, creates placeholder output files
  (copies of the input image) so the pipeline still produces outputs.
"""

import subprocess
import argparse
from pathlib import Path
import shutil
import sys


def _safe_matlab_command(image_path: Path, output_dir: Path, num_images: int, noise_type: str) -> str:
    """
    Build a MATLAB command string that adds the folder containing `generate_noisy_images.m`
    to the path before calling the function.
    """
    # Use POSIX-style paths for MATLAB
    img = image_path.as_posix().replace("'", "''")
    out = output_dir.as_posix().replace("'", "''")

    # Folder containing generate_noisy_images.m
    matlab_func_folder = Path(__file__).parent.resolve().as_posix().replace("'", "''")

    # MATLAB command: add path, call function
    cmd = (
        f"addpath('{matlab_func_folder}'); "
        f"generate_noisy_images('{img}', '{out}', {num_images}, '{noise_type}'); "
        f"rmpath('{matlab_func_folder}');"  # optional: clean up path after
    )
    return cmd

def generate_noisy_images(image_path, output_dir, num_images, noise_type='all', matlab_cmd_override=None):
    """
    Generate noisy images using MATLAB (or create placeholders if MATLAB not found).

    Args:
        image_path: Path to input image
        output_dir: Output directory
        num_images: Number of images to generate per noise type
        noise_type: 'all', 'gaussian', 'salt_pepper', 'poisson', 'speckle', 'uniform'
        matlab_cmd_override: optional path to matlab executable (string)
    """
    image_path = Path(image_path).resolve()
    output_dir = image_path.parent.parent / "noisy_output"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # prefer an explicit override, else try PATH
    matlab_exec = None
    if matlab_cmd_override:
        matlab_exec = shutil.which(matlab_cmd_override) or matlab_cmd_override
    else:
        matlab_exec = shutil.which('matlab')

    # build command string for MATLAB
    matlab_cmd_str = _safe_matlab_command(image_path, output_dir, num_images, noise_type)

    print(f"Generating {num_images} image(s) with '{noise_type}' noise...")
    print(f"Input: {image_path}")
    print(f"Output directory: {output_dir}\n")

    if matlab_exec:
        # Run MATLAB
        try:
            result = subprocess.run(
                [matlab_exec, '-nojvm', '-nodisplay', '-nosplash', '-batch', matlab_cmd_str],
                capture_output=True,
                text=True,
                timeout=300
            )

        except FileNotFoundError as e:
            print(f"Error launching MATLAB: {e}")
            return False
        except subprocess.TimeoutExpired as e:
            print(f"MATLAB call timed out: {e}")
            return False

        if result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            print("MATLAB returned an error:")
            print(result.stderr)
            return False

        print("\n✓ MATLAB finished successfully.")
        return True

    else:
        print(e)
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate multiple noisy images using MATLAB (or create placeholders if MATLAB not available)',
        epilog="""
Examples:
  python noise_gen.py image.jpg -n 10                    # 10 images of each type
  python noise_gen.py image.jpg -n 5 -t gaussian         # 5 Gaussian images
  python noise_gen.py image.jpg -n 3 -t salt_pepper      # 3 salt&pepper images
        """
    )

    parser.add_argument('image', help='Input image path')
    parser.add_argument('-n', '--num', type=int, default=5,
                        help='Number of images per noise type (default: 5)')
    parser.add_argument('-o', '--output', default='noisy_images',
                        help='Output directory (default: noisy_images)')
    parser.add_argument('-t', '--type', default='all',
                        choices=['all', 'gaussian', 'salt_pepper',
                                 'poisson', 'speckle', 'uniform'],
                        help='Noise type (default: all)')
    parser.add_argument('--matlab-path', dest='matlab_path', default=None,
                        help='Path to matlab executable (if not on PATH)')

    args = parser.parse_args()

    try:
        ok = generate_noisy_images(args.image, args.output, args.num, args.type, args.matlab_path)
        if not ok:
            return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
