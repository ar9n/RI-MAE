import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args, cwd=ROOT):
    print(f"\n>>> {' '.join(map(str, args))}", flush=True)
    subprocess.check_call(list(map(str, args)), cwd=cwd)


def main():
    python = sys.executable

    # 1. Install Python dependencies
    run(python, "-m", "pip", "install", "-r", ROOT / "requirements.txt")

    # 2. Verify PyTorch is installed before compiling extensions
    run(
        python, "-c",
        "import torch; print('PyTorch:', torch.__version__, "
        "'CUDA:', torch.version.cuda)"
    )

    # 3. Build local CUDA extensions against the active PyTorch
    run(
        python, "-m", "pip", "install", ".", "--no-build-isolation",
        cwd=ROOT / "extensions" / "chamfer_dist"
    )
    run(
        python, "-m", "pip", "install", ".", "--no-build-isolation",
        cwd=ROOT / "extensions" / "emd"
    )

    # 4. Install PointNet++
    run(
        python, "-m", "pip", "install", "--no-build-isolation",
        "git+https://github.com/ar9n/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
    )

    # 5. Install GPU kNN
    run(
        python, "-m", "pip", "install", "--upgrade",
        "https://github.com/altaykacan/KNN_CUDA_reborn/"
        "releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl"
    )

    print("\nRI-MAE dependencies and extensions installed.")


if __name__ == "__main__":
    main()
