#!/usr/bin/env python3
# twine upload dist/rbo-lbm-x.x.x.tar.gz
# twine upload dist/rbo-lbm.x.x.tar.gz -r test
# pip install --index-url https://test.pypi.org/simple/ --upgrade rbo-lbm

from setuptools import setup #find_packages

long_description = "Light Beads Microscopy 2P Calcium Imaging Pipeline."

setup(
    name='LBM-CaImAn-Python',
    version='0.1.0',
    description="Light Beads Microscopy 2P Calcium Imaging Pipeline.",
    long_description=long_description,
    author='Flynn OConnell',
    author_email='foconnell@rockefeller.edu',
    license='',
    url='https://github.com/ru-rbo/rbo-lbm',
    keywords='Pipeline Numpy Microscopy ScanImage multiROI tiff',
    packages=['util'],
    install_requires=['numpy>=1.12.0', 'tifffile>=2019.2.22'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'Natural Language :: English'
        'Topic :: Scientific/Engineering',
    ],
)
