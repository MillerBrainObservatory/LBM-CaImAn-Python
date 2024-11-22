# LBM-CaImAn-Python Documentation 

For the `MATLAB` implementation of this pipeline, see [here](https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB/).

For current installation instructions, see the project [README](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python/blob/master/README.md)

## Documentation Contents

```{toctree}
---
maxdepth: 2
---
user_guide/index
examples/index
api/index
glossary
```

----------------

## Pipeline Overview

LBM-CaImAn-Pipeline uses [mesmerize-core](https://github.com/nel-lab/mesmerize-core/tree/master) to interface with [CaImAn](https://github.com/flatironinstitute/CaImAn) algorithms for Calcium Imaging data processing.

There are 4 steps in this pipeline:

1. Assembly
    - De-interleave planes
    - Scan-Phase Correction
2. Motion Correction
    - Template creation
    - Rigid registration
    - Piecewise-rigid registration
3. Segmentation
    - Iterative CNMF segmentation
    - Deconvolution
    - Neuron Selection Refinement (COMING SOON)
4. Collation
    - Collate images and metadata into a single volume
    - Lateral offset correction (between z-planes, COMING SOON)

## Comparison with [LBM-CaImAn-MATLAB](https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB/)

### Usage

Beyond the obviously different programming language (MATLAB -> Python), there are a few differences in how these pipelines were constructed.

The MATLAB implementation was essentially 4 functions spread across 4 `.m` files. These functions would be called from a user-made script (for example, [demo_LBM_pipeline.m](https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB/blob/master/demo_LBM_pipeline.m)).

### Performance

The primary pitfal of [LBM-CaImAn-MATLAB](https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB/) are the memory constraints. Though MATLAB is extremely efficient with threaded internal functions, the lack of 3rd-party library support means reading and writing to well-established file-formats (i.e. `.tiff`, `.hdf5`) lacking modern features like [lazy-loading data](https://www.imperva.com/learn/performance/lazy-loading/). As a result, the memory footprint required to process a *N-GB dataset* will be *N-GB of memory*. 

This pipeline utilizes the well-tested and optimized [tifffile](https://pypi.org/project/tifffile/) to selectively load data only when it is needed. That is why processing a `35 GB` file will only consume ~`5 GB` of memory.

## Helpful Resources

- [CaImAn Documentation](https://caiman.readthedocs.io/en/latest/)
- [mesmerize-core Documentation](https://mesmerize-core.readthedocs.io/en/latest/#installation)
- [pandas 10-minute tutorial](https://pandas.pydata.org/docs/user_guide/10min.html)
