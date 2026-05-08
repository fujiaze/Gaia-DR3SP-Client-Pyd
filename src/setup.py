"""
Build Gaia Spectra Store as a compiled Python extension (.pyd = DLL on Windows).

Usage:
    pip install -e .                    # dev install (editable, .py fallback)
    python setup.py build_ext --inplace  # compile .pyd in-place
    pip wheel . -w dist/                 # build wheel for distribution
"""
import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages

extensions = [
    Extension(
        "gaia_spectra_store._healpix",
        ["gaia_spectra_store/_healpix.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["/O2"] if "win" in __import__("sys").platform else ["-O3"],
    ),
    Extension(
        "gaia_spectra_store.xpsd_client",
        ["gaia_spectra_store/xpsd_client.py"],
        include_dirs=[np.get_include()],
        extra_compile_args=["/O2"] if "win" in __import__("sys").platform else ["-O3"],
    ),
]

setup(
    name="gaia-spectra-store",
    version="2.0.0",
    description="Gaia DR3/SP XPSD Direct Client — 2.19亿星 · 毫秒查询 · 零转换 · .pyd编译",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "embedsignature": True,
        },
        annotate=True,
    ),
    install_requires=[
        "numpy>=1.24",
        "msgpack>=1.0",
        "tqdm>=4.60",
    ],
    python_requires=">=3.8",
)
