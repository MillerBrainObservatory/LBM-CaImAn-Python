# LBM-CaImAn-Python

[**Installation**](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python#installation) | [**Usage**](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python#usage)

Python implementation of the Light Beads Microscopy (LBM) computational pipeline.

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

Pixi is the only tool you need — it pulls CaImAn from conda-forge for you, so there is no
separate conda install, no clone, and no C++ compiler. Install [pixi](https://pixi.sh)
(`pip install pixi` or see https://pixi.sh for other methods).

### Install (no clone)

```bash
pixi init my-lcp && cd my-lcp
pixi add caiman "python>=3.12.7,<3.12.10" "numpy>=2.2.5,<2.4" "setuptools<81"
pixi add --pypi lbm-caiman-python
pixi run caimanmanager install   # harmless error if ~/caiman_data already exists; skip it
pixi run lcp <input> <output>
```

Add CaImAn (conda) **before** `--pypi lbm-caiman-python`: CaImAn supplies conda PyTorch,
which then satisfies suite2p's `torch` requirement, so pip does not install a second copy
over it (mixing the two breaks torch's DLLs on Windows). The pins are required too —
CaImAn is built for python 3.12 only (a fresh project otherwise defaults to a newer one),
and it pulls the latest numpy/setuptools, which exceed the `numpy<2.4` / `setuptools<81`
caps that `mbo_utilities` sets.

If you already installed `lbm-caiman-python` and only need CaImAn, run `pixi run lcp setup`
(it runs the `pixi add caiman …` + `caimanmanager install` above; `--force` to reinstall,
`--no-data` to skip the data setup). If a `torch`/DLL error appears, the env picked up a
stray pip torch — delete `.pixi/envs/default` and re-run `pixi install`.

### Install for development (clone)

```bash
git clone https://github.com/MillerBrainObservatory/LBM-CaImAn-Python.git
cd LBM-CaImAn-Python
pixi install
pixi run setup-caiman
```

This installs CaImAn from conda-forge along with all dependencies and the project itself in editable mode.

To verify:

```bash
pixi run python -c "import lbm_caiman_python as lcp; print(lcp.__version__)"
```

:exclamation: **Hardware requirements** The large CNMF visualizations with contours etc. usually require either a dedicated GPU or integrated GPU with access to at least 1GB of VRAM.

---

## Usage

Run the pipeline (assembly → motion correction → CNMF → collation) on a raw ScanImage
tiff file or directory:

```bash
pixi run lcp <input> <output>
```

4D/volumetric input is processed one z-plane at a time. Output is written per plane in
suite2p format (`data.bin`, `ops.npy`, `stat.npy`, `iscell.npy`, `F.npy`, `dff.npy`, …)
with per-plane and volume figures, so results load in mbo studio / lbm_suite2p_python.

Common options:

- `--planes 1 2 3` — process specific planes (1-based)
- `--no-mcorr` / `--no-cnmf` — skip a stage
- `--force-mcorr` / `--force-cnmf` — re-run a stage, ignoring cached results
- `--num-frames N` — limit frames processed
- `--K`, `--gSig`, `--min-SNR`, `--rval-thr`, `--max-shifts`, `--strides`, … — override defaults
- `lcp setup` — install CaImAn into the active pixi/conda env (see [Installation](#installation))

`lcp --help` lists every option with its default value.

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

