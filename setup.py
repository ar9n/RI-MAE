from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CHAMFER_DIR = "extensions/chamfer_dist"
EMD_DIR = "extensions/emd"

ext_modules = [
    CUDAExtension(
        name="chamfer",
        sources=[
            f"{CHAMFER_DIR}/chamfer_cuda.cpp",
            f"{CHAMFER_DIR}/chamfer.cu",
        ],
    ),
    CUDAExtension(
        name="emd_cuda",
        sources=[
            f"{EMD_DIR}/cuda/emd.cpp",
            f"{EMD_DIR}/cuda/emd_kernel.cu",
        ],
        extra_compile_args={
            "cxx": ["-g"],
            "nvcc": ["-O2"],
        },
    ),
]

setup(
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension,
    },
)
