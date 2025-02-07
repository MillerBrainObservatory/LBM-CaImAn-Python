# LBM-CaImAn-Python

[**Installation**](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python#installation) | [**Notebooks**](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python/tree/master/demos/notebooks)

[![Documentation](https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/LBM-CaImAn-Python/)
 
Python implementation of the Light Beads Microscopy (LBM) computational pipeline. The documentation has examples of the rendered notebooks.

For the `MATLAB` implementation, see [here](https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB/)

## Pipeline Steps:

1. Image Assembly
    - Extract raw `tiffs` to planar timeseries
2. Motion Correction
    - Rigid/Non-rigid registration
3. Segmentation
    - Iterative CNMF segmentation
    - Deconvolution
    - Refine neuron selection
4. Collation
    - Collate images and metadata into a single volume
    - Lateral offset correction (between z-planes. WIP)

## Requirements

- caiman
- numpy
- scipy
- fastplotlib

:exclamation: **Note:** This package makes heavy use of fastplotlib for visualizations.

fastplotlib runs on [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html),
but is not guarenteed to work with Jupyter Notebook or Visual Studio Code notebook environments. 

## Installation

This project is tested on Linux and Windows 10. It will likely work on Mac as well.

Installation is tested using [miniforge](https://github.com/conda-forge/miniforge).
Python [virtual-environments](https://virtualenv.pypa.io/en/latest/) are working for **Linux/MacOS only**.

:exclamation: Anaconda and Miniconda will likely cause package conflicts.

### (Option 1). Miniforge (conda)

Note: If conda gets stuck `Solving Environment`, hitting enter can sometimes help.

1. Create a new environment and install [CaImAn](https://github.com/nel-lab/mesmerize-core/tree/master)

Installing CaImAn requires extra steps on Windows:

#### Windows Only

:exclamation: **Note:** If you encounter errors during the installation of `CaImAn`, you may need to install Microsoft Visual C++ Build Tools. You can download them from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

``` bash
% -n is the name of the environment, but nama it whatever you want
conda create -n lcp python=3.11 pip vs2019_win-64 imgui-bundle tifffile=2024.12.12
conda activate lcp
```

Clone and install CaImAn using git:

```bash
git clone https://github.com/flatironinstitute/CaImAn.git
cd CaImAn
pip3 install .
cd ../ 
caimanmanager install -f
```

- if the command `pip3 install` leads to `pip3: command not found`, try `pip install .`

#### Linux/MacOS

Install CaImAn with `conda`:

``` bash

% -n is the name of the environment, but nama it whatever you want
conda create -n lcp -c conda-forge python=3.10 caiman imgui-bundle
conda activate lcp
```

2. Install LBM-CaImAn-Python

``` bash
pip install git+https://github.com/MillerBrainObservatory/LBM-CaImAn-Python.git@master
```

:exclamation: **Harware requirements** The large CNMF visualizations with contours etc. usually require either a dedicated GPU or integrated GPU with access to at least 1GB of VRAM.

----

### (Option 2). Python Virtual Environments (Linux/MacOS)

Ensure you have a system-wide Python installation.

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

```bash
   cd CaImAn
   pip install .
```

Install LBM-CaImAn-Python:

2.

```bash
pip install git+https://github.com/MillerBrainObservatory/LBM-CaImAn-Python.git@master
    ```

#### Run ipython to make sure everyting works

``` python

import lbm_caiman_python as lcp
import lbm_mc as mc

# optional
scan = lcp.read_scan('path/to/data/*.tif', join_contiguous=True)

```

---

## Troubleshooting

### Error during pip install: OSError: [Errno 2] No such file or directory

If you recieve an error during pip installation with the hint:

```bash

HINT: This error might have occurred since this system does not have Windows Long Path support enabled. You can find
 information on how to enable this at https://pip.pypa.io/warnings/enable-long-paths

```

In Windows Powershell, as Administrator:

`New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`

Or:

- Open Group Policy Editor (Press Windows Key and type gpedit.msc and hit Enter key.

- Navigate to the following directory:  

`Local Computer Policy > Computer Configuration > Administrative Templates > System > Filesystem > NTFS.`

- Click Enable NTFS long paths option and enable it.

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

