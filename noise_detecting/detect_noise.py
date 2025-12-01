import subprocess
import sys
from pathlib import Path

def detect_noise(img_path):
    img_path = str(Path(img_path).resolve())

    # Folder where detect_noise.py is located
    script_dir = Path(__file__).parent.resolve()

    # Absolute path to MATLAB function folder
    matlab_func_dir = str(script_dir)

    matlab_cmd = (
        f"addpath('{matlab_func_dir}'); "
        f"try, "
        f"result = detect_noise_type('{img_path}'); "
        f"disp(result); "
        f"catch e, disp('ERROR:'), disp(getReport(e)), exit(1); "
        f"end; exit(0);"
    )

    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True, text=True
    )

    print(result.stdout)

    # Extract last printed line
    lines = result.stdout.strip().splitlines()
    return lines[-1]


if __name__ == "__main__":
    img = sys.argv[1]
    noise = detect_noise(img)
    print("Detected noise type:", noise)
