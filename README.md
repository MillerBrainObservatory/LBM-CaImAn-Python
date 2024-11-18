# LBM-CaImAn-Python

Python implementation of the Light Beads Microscopy (LBM) computational pipeline.

For the `MATLAB` implementation, see [here](https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB/)

## Installation

We need to set up a conda installation for `CaImAn` and `mesmerize-core` to be properly installed.

The recommended conda installer is [miniforge](https://github.com/conda-forge/miniforge).
This is a community-driven `conda`/`mamba` installer with pre-configured packages specific to [conda-forge](https://conda-forge.org/).

This helps avoid `conda-channel` conflicts and avoids any issues with the Anaconda TOS.

You can install the installer from the command line in a bash/zsh shell:

``` bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Or download the installer for your operating system [here](https://github.com/conda-forge/miniforge/releases).

```
# if conda is behaving slow, clean/update the **base** environment
conda activate base
conda clean -a
conda update -c conda-forge -all
```

Next, install [`mesmerize-core`](https://github.com/nel-lab/mesmerize-core)
```
conda create -n lbm_py -c conda-forge mesmerize-core

```

If you already have `CaImAn` installed:

```
conda install -n name-of-env-with-caiman mesmerize-core
```

Activate the environment and install `caimanmanager`:

```
conda activate mesmerize-core
caimanmanager install
```

Now you are ready to install lbm_caiman_python and the latest fastplotlib:

```
pip install lbm_caiman_python

# install latest pygfx
pip install git+https://github.com/pygfx/pygfx.git@main
```

## Pipeline Steps:

1. Motion Correction
    - Template creation
    - Rigid registration
    - Piecewise-rigid registration
2. Segmentation
    - Iterative CNMF segmentation
    - Deconvolution
    - Refinement
3. Collation
    - Lateral offset correction (between z-planes)
    - Collate images and metadata into a single volume

# Requirements

- caiman
- mesmerize-core
- numpy
- scipy

```{note}
See the `environment.yml` file at the root of this project for a complete list of package dependencies.
```
