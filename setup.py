#!/usr/bin/env python3

import setuptools
import versioneer
from pathlib import Path

install_deps = [
    "tifffile",
    "numpy>=1.24.3,<2.0",
    "numba>=0.57.0",
    "scipy>=1.9.0",
    "fastplotlib[notebook]",
    "scanreader @ git+https://github.com/atlab/scanreader.git@master#egg=scanreader",
    "matplotlib",
    "lbm-mc",
    "fabric",
    "dask",
    "zarr",
]

with open(Path(__file__).parent / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lbm_caiman_python",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Light Beads Microscopy 2P Calcium Imaging Pipeline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name Here",
    author_email="your_email@example.com",
    license="BSD-3-Clause",
    url="https://github.com/millerbrainobservatory/LBM-CaImAn-Python",
    keywords="Pipeline Numpy Microscopy ScanImage multiROI tiff",
    install_requires=install_deps,
    packages=setuptools.find_packages(exclude=["data", "data.*"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
    ],
    entry_points={
        "console_scripts": [
            "lcp = lbm_caiman_python.__main__:main",
            "sr = lbm_caiman_python.assembly:main",
            "transfer = lbm_caiman_python.transfer:main",
            "run_slurm = lbm_caiman_python.run_slurm:main",
        ]
    },
)

