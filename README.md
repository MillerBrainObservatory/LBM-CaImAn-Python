## Light Beads Microscopy (LBM) Pipeline: CaImAn-Python

[![Documentation](https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/LBM-CaImAn-Python/)

A pipeline for processing light beads microscopy (LBM) datasets using the [flatironinstitute/caiman](https://github.com/flatironinstitute/CaImAn) pipeline.

For the MATLAB implementation, see [here](https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB)

[![Issues](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://GitHub.com/MillerBrainObservatory/LBM-CaImAn-Python/issues/)

## Overview

This pipeline is unique only in the routines to extract raw data from [ScanImage tiff files](https://docs.scanimage.org/Appendix/ScanImage%2BBigTiff%2BSpecification.html#scanimage-bigtiff-specification), as is outlined below:

![Extraction Diagram]( docs/_static/_images/extraction/extraction_diagram.png)

Once data is extracted to an intermediate filetype `h5`, `.tiff`, `.memmap`, registration, segmentation and deconvolution can all be performed as described in the corresponding pipelines documentation.

The [documentation] for usage, tutorials, tips and tricks. Follow the root `demo_LBM_pipeline.m` file for an example pipeline, or the root `/notebooks` folder for more in-depth exploration of individual pipeline steps.

## Requirements

- caiman
- numpy
- scipy

See the `environment.yml` file at the root of this project for a complete list of package dependencies.

## Algorithms

The following algorithms perform the main computations and are included by default in the pipeline:

- [CNMF](https://github.com/simonsfoundation/NoRMCorre) segmentation and neuronal source extraction.
- [NoRMCorre](https://github.com/flatironinstitute/NoRMCorre) piecewise rigid motion correction.
- [constrained-foopsi](https://github.com/epnev/constrained-foopsi) constrained deconvolution spike inference.

## References

- [Nature Publication](https://www.nature.com/articles/s41592-021-01239-8/)
- [ScanImage-MROI Docs](https://docs.scanimage.org/Premium%2BFeatures/Multiple%2BRegion%2Bof%2BInterest%2B%28MROI%29.html#multiple-region-of-interest-mroi-imaging/)
- [MBO Homepage](https://mbo.rockefeller.edu/)
- [startup matlab files](https://www.mathworks.com/help/matlab/matlab_env/matlab-startup-folder.html)
- [![Publication](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1038/s41592-021-01239-8)
