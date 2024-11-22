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
    "fastplotlib[notebook]",
    "matplotlib",
    "dask",
    "zarr",
]

extras_require = {
    "docs": [
        "sphinx>=6.1.3",
        "docutils>=0.19",
        "nbsphinx",
        "numpydoc",
        "sphinx-autodoc2",
        "sphinx_gallery",
        "sphinx-togglebutton",
        "sphinx-copybutton",
        "sphinx_book_theme",
        "pydata_sphinx_theme",
        "sphinx_design",
        "sphinxcontrib-images",
        "sphinxcontrib-video",
        "sphinx_tippy",
        "myst_nb",
    ],
}

with open(Path(__file__).parent / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(Path(__file__).parent.joinpath("lbm_caiman_python", "VERSION"), "r") as f:
    ver = f.read().split("\n")[0]

setuptools.setup(
    name="lbm_caiman_python",
    version=ver,
    description="Light Beads Microscopy 2P Calcium Imaging Pipeline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Flynn OConnell",
    author_email="foconnell@rockefeller.edu",
    license="",
    url="https://github.com/millerbrainobservatory/LBM-CaImAn-Python",
    keywords="Pipeline Numpy Microscopy ScanImage multiROI tiff",
    install_requires=install_deps,
    extras_require=extras_require,
    packages=setuptools.find_packages(),
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
        ]
    },
)
