# LBM-CaImAn-Python

[**Installation**](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python#installation) | [**Notebooks**](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python/tree/master/demos/notebooks)
 
Python implementation of the Light Beads Microscopy (LBM) computational pipeline.

For the `MATLAB` implementation, see [here](https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB/)

## Pipeline Steps:

1. Image Assembly
    - Extract raw `tiffs` to planar timeseries
2. Motion Correction
    - Template creation
    - Piecewise-rigid or non-rigid registration
3. Segmentation
    - Iterative CNMF segmentation
    - Deconvolution
    - Refinement
4. Collation
    - Lateral offset correction (between z-planes)
    - Collate images and metadata into a single volume

## Requirements

- caiman
- mesmerize-core
- scanreader
- numpy
- scipy
- fastplotlib

:exclamation: **Note:** This package makes heavy use of Fastplotlib for visualizations.

Fastplotlib runs on [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html),
but is not guarenteed to work with Jupyter Notebook or Visual Studio Code notebook environments. 

## Installation

This project is tested on Linux and Windows 10 using `Python 3.9` and `Python 3.10`.

Installation is tested using [miniforge](https://github.com/conda-forge/miniforge). Python [virtual-environments](https://virtualenv.pypa.io/en/latest/) are a not fully developed/tested yet but is left as an optional installation method for python-savvy users.

:exclamation: Anaconda and Miniconda will likely not work due to package conflicts.

### (Option 1). Miniforge (conda)

Note: If conda gets stuck `Solving Environment`, hitting enter can sometimes help.

1. Create a new environment and install [mesmerize-core](https://github.com/nel-lab/mesmerize-core/tree/master)

- Here, we use the `-n` flag to name the environment `lbm` , but you can name it whatever you'd like.
- This step will install Python, mesmerize-core, CaImAn, and all required dependencies for those packages.

``` bash
conda create -n lbm -c defaults -c conda-forge mesmerize-core
```

If you already have `CaImAn` installed, replace `-n lbm` with `-n name-of-env-with-caiman`.

Activate the environment:

- if you used a name other than `lbm`, be sure to match the name you use here.

``` bash
conda activate lbm
```

2. Install [LBM-CaImAn-Python](https://pypi.org/project/lbm-caiman-python/) and [scanreader](https://github.com/atlab/scanreader):

``` bash
pip install lbm_caiman_python
pip install git+https://github.com/atlab/scanreader.git
```

3. Install `caimanmanager`

CaImAn will sometimes look for neural network models, unless you tell it not to with parameters `use_cnn=False` during segmentation.

To install these models, and CaImAn demo data to follow along with their notebooks:

``` bash
caimanmanager install
```

This will create a directory in your home folder `~/caiman_data/`. We recommend doing this step, though it may be safe to skip.

4. (Optional) Install `mesmerize-viz`:

Several notebooks make use of [mesmerize-viz](https://github.com/kushalkolar/mesmerize-viz) for visualizing registration/segmentation results.

``` bash
pip install mesmerize-viz
```

:exclamation: **Harware requirements** The large CNMF visualizations with contours etc. usually require either a dedicated GPU or integrated GPU with access to at least 1GB of VRAM.

[mesmerize-viz youtube video demonstration](https://www.youtube.com/watch?v=GWvaEeqA1hw)

4. Stay up-to-date

LBM-CaImAn-Python is in active development. To update to the latest release:

```python
pip install -U lbm_caiman_python
```


----

### (Option 2). Python Virtual Environments (WIP)

:exclimation: With numpy 2.0 official release, there may be errors for this method. File an issue if you have conflicts.

Ensure you have a system-wide Python installation.

This project and it's dependencies are tested using `Python 3.9` and `Python 3.10`.

**Note:** Make sure you deactivate `conda` environments before proceeding (`conda deactivate`).

Verify `Python` and `pip` installations:

- **Linux/macOS:**
  
```bash

python --version

pip --version
```

- **Windows:**

```bash
py --version

py - m pip --version 

```

:exclamation: Depending on how Python was installed,
you may need to use `python3` or `python3.x` and `pip3` or `pip3.x` instead of `python` and `pip`.

You should see a Python version output like `3.10.x` and a corresponding `pip` version.

If Python is not installed, or an unsupported version is installed (i.e. 3.7),

download and install [python.org](https://www.python.org/) or refer to this [installation guide](https://docs.python-guide.org/starting/installation/).

You will also need [`git`](https://git-scm.com/):

```bash
git --version
```

#### Create a virtual environment

This is normally in a directory dedicated to virtual environments, but can be anywhere you wish:

```bash
python -m venv ~/venv/lbm_caiman_python
```

Activate the virtual environment:

- **Linux/macOS:**

  ```bash
  source ~/venv/lbm_caiman_python/bin/activate
  ```

- **Windows:**

  ```bash
  source ~/venv/lbm_caiman_python/Scripts/activate
  ```

Upgrade core tools in the virtual environment:

```bash
pip install --upgrade setuptools wheel pip
```

#### Clone and install CaImAn

Create a directory to store the cloned repositories.

Again, this can be anywhere you wish:

```bash

cd ~
mkdir repos
cd repos

```

Use git to clone CaImAn:

```bash
git clone https://github.com/flatironinstitute/CaImAn.git
```

Install CaImAn:

1. **CaImAn:**
   ```bash
   cd CaImAn
   pip install -r requirements.txt
   pip install .
   ```
    :exclamation: **Note:** If you encounter errors during the installation of `CaImAn`, you may need to install Microsoft Visual C++ Build Tools. You can download them from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

2. **Other dependencies:**

    ```bash
    pip install mesmerize-core
    pip install lbm_caiman_python
    pip install git+https://github.com/atlab/scanreader.git
    ```

#### Run ipython to make sure everyting works

``` python

import lbm_caiman_python as lcp
import mesmerize_core as mc
import scanreader as sr

scan = sr.read_scan('path/to/data/*.tif', join_contiguous=True)

```


---

## For Developers

To get the newest version of this package, rather than `pip install lbm_caiman_python`:

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

### virtualenv Troubleshooting

#### Error During `pip install .` (CaImAn) on Linux
If you encounter errors during the installation of `CaImAn`, install the necessary development tools:
```bash
sudo apt-get install python3-dev
```

Don't forget to press enter a few times if conda is taking a long time.

### Recommended Conda Distribution

The recommended conda installer is 

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

