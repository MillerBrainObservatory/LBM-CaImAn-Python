# LBM-CaImAn-Python

[**Installation**](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python#installation) | [**Notebooks**](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python/tree/master/demos/notebooks)
 
Python implementation of the Light Beads Microscopy (LBM) computational pipeline.

For the `MATLAB` implementation, see [here](https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB/)

## Installation

Installation requires an active `conda` installation.

Note: Sometimes conda or mamba will get stuck at a step, such as creating an environment or installing a package. Pressing Enter on your keyboard can sometimes help it continue when it pauses.

1. Install `mamba` into your *base* environment:

:exclamation: This step may take 10 minutes and display several messages like "Solving environment: failed with..." but it should eventually install mamba.

``` bash
conda activate base 
conda install -c conda-forge mamba
```

2. Create a new environment and install [mesmerize-core](https://github.com/nel-lab/mesmerize-core/tree/master)

- Here, we use the `-n` flag to name the environment `lbm` , but you can name it whatever you'd like.
- This step will install Python, mesmerize-core, CaImAn, and all required dependencies for those packages.

``` bash
conda create -n lbm -c conda-forge mesmerize-core
```

If you already have `CaImAn` installed:

``` bash
conda install -n name-of-env-with-caiman mesmerize-core
```

Activate the environment and install `caimanmanager`:
- if you used a name other than `lbm`, be sure to match the name you use here.

``` bash
conda activate lbm
caimanmanager install
```

3. Install [LBM-CaImAn-Python](https://pypi.org/project/lbm-caiman-python/) from pip:

Install the minimal `lbm_caiman_python`:

``` bash

pip install lbm_caiman_python

```

4. Install [scanreader](https://github.com/atlab/scanreader):

``` bash

pip install git+https://github.com/atlab/scanreader.git

```

5. (Optional) Install `mesmerize-viz`:

Several notebooks make use of [mesmerize-viz](https://github.com/kushalkolar/mesmerize-viz) for visualizing registration/segmentation results.

``` bash

pip install https://github.com/kushalkolar/mesmerize-viz.git

```

:exclamation: **Harware requirements** The large CNMF visualizations with contours etc. usually require either a dedicated GPU or integrated GPU with access to at least 1GB of VRAM.

https://www.youtube.com/watch?v=GWvaEeqA1hw

## For Developers

To get the newest version of this package:

``` bash

git clone https://github.com/MillerBrainObservatory/LBM-CaImAn-Python.git

cd LBM-CaImAn-Python

pip install ".[docs]"

```

## Troubleshooting

### Conda Slow / Stalling

if conda is behaving slow, clean the conda installation and update `conda-forge`:

``` bash

conda clean -a

conda update -c conda-forge --all

```

Don't forget to press enter a few times if conda is taking a long time.

### Recommended Conda Distribution

The recommended conda installer is [miniforge](https://github.com/conda-forge/miniforge).

This is a community-driven `conda`/`mamba` installer with pre-configured packages specific to [conda-forge](https://conda-forge.org/).

This helps avoid `conda-channel` conflicts and avoids any issues with the Anaconda TOS.

You can install the installer from a unix command line:

``` bash

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash Miniforge3-$(uname)-$(uname -m).sh

```

Or download the installer for your operating system [here](https://github.com/conda-forge/miniforge/releases).

### Graphics Driver Issues

If you are attempting to use fastplotlib and receive errors about graphics drivers, see the [fastplotlib driver documentation](https://github.com/fastplotlib/fastplotlib?tab=readme-ov-file#gpu-drivers-and-requirements).

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

## Requirements

- caiman
- mesmerize-core
- numpy
- scipy

