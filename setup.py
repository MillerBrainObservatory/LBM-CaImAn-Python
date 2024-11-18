#!/usr/bin/env python3
# twine upload dist/rbo-lbm-x.x.x.tar.gz
# twine upload dist/rbo-lbm.x.x.tar.gz -r test
# pip install --index-url https://test.pypi.org/simple/ --upgrade rbo-lbm

import setuptools
from pathlib import Path

install_deps = [
    "tifffile",
    "numpy>=1.24.3",
    "numba>=0.57.0",
    "scipy>=1.9.0",
    "matplotlib",
    "dask",
    "scanreader",
    "zarr",
    "jupyterlab",
]

docs_deps = [
    "sphinx",
    "sphinx-gallery",
    "pydata-sphinx-theme",
    "jupyter-rfb>=0.4.1",  # required so ImageWidget docs show up
    "sphinx-copybutton",
    "sphinx-design",
    "matplotlib",
]

with open(Path(__file__).parent.joinpath("README.md")) as f:
    readme = f.read()

with open(Path(__file__).parent.joinpath("lbm_caiman_python", "VERSION"), "r") as f:
    ver = f.read().split("\n")[0]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lbm_caiman_python",
    version=ver,
    description="Light Beads Microscopy 2P Calcium Imaging Pipeline.",
    long_description=readme,
    author="Flynn OConnell",
    author_email="foconnell@rockefeller.edu",
    license="",
    url="https://github.com/millerbrainobservatory/LBM-CaImAn-Python",
    keywords="Pipeline Numpy Microscopy ScanImage multiROI tiff",
    install_requires=install_deps,
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English" "Topic :: Scientific/Engineering",
    ],
    entry_points = {
        "console_scripts": [
            "lcp = lbm_caiman_python.__main__:main",
        ]
    },
)
