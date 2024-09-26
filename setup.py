#!/usr/bin/env python3
# twine upload dist/rbo-lbm-x.x.x.tar.gz
# twine upload dist/rbo-lbm.x.x.tar.gz -r test
# pip install --index-url https://test.pypi.org/simple/ --upgrade rbo-lbm

import setuptools

install_deps = [
    # "importlib-metadata",
    # "natsort",
    # "rastermap>=0.9.0",
    "tifffile",
    # "torch>=1.13.1",
    "numpy>=1.24.3",
    "numba>=0.57.0",
    "matplotlib",
    "scipy>=1.9.0",
    "dask",
    # "scanimage-tiff-reader>=1.4.1",
]

gui_deps = [
    "qtpy",
    "pyqt6",
    "pyqt6.sip",
    "pyqtgraph",
]

io_deps = [
    # "paramiko", # ssh
    # "h5py",
    "opencv-python-headless",
    "zarr",
    # "xmltodict",
]

notebook_deps = ["jupyterlab"]

all_deps = gui_deps + io_deps

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lbm_caiman_python",
    version="0.7.0",
    description="Light Beads Microscopy 2P Calcium Imaging Pipeline.",
    long_description=long_description,
    author="Flynn OConnell",
    author_email="foconnell@rockefeller.edu",
    license="",
    url="https://github.com/millerbrainobservatory/LBM-CaImAn-Python",
    keywords="Pipeline Numpy Microscopy ScanImage multiROI tiff",
    # packages=setuptools.find_packages(),
    install_requires=install_deps,
    extras_require={
        "docs": [
            "sphinx>=3.0",
            "sphinxcontrib-apidoc",
            "sphinx_rtd_theme",
            "sphinx-prompt",
            "sphinx-autodoc-typehints",
        ],
        "gui": gui_deps,
        "io": io_deps,
        "notebook": notebook_deps,
        "all": all_deps,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English" "Topic :: Scientific/Engineering",
    ],
)
