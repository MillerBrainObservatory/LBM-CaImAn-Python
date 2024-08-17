# LBM-CaImAn-Python

[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)

[![Documentation](https://img.shields.io/badge/LBM%20Documentation-1f425f.svg)(https://millerbrainobservatory.github.io/LBM-CaImAn-Python/)]

Python implementation of the Light Beads Microscopy (LBM) computational pipeline.

For the `MATLAB` implementation, see [here](https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB/)

## Pipeline Steps:

1. Extraction
    - De-interleave zT
    - Scan Phase-Correlation
2. Registration
    - Template creation
    - Rigid registration
    - Piecewise-rigid registration
3. Segmentation
    - Iterative CNMF segmentation
    - Deconvolution
    - Refinement

# Requirements

- caiman
- numpy
- scipy

```{note}
See the `environment.yml` file at the root of this project for a complete list of package dependencies.
```
