# LBM-CaImAn-Python Documentation 

For the `MATLAB` implementation of this pipeline, see [here](https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB/).

For current installation instructions and requirements, see the project [README](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python/blob/master/README.md).

## How to

The recommended way to use this pipeline is by using the [example notebooks](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python/tree/master/demos/notebooks) 
while having the user guide accessible as well. The notebooks act as a walkthrough tutorial that you can change to fit your dataset as you go. Fully rendered versions of the notebooks
are available in the [tutorial section of this documentation](https://millerbrainobservatory.github.io/LBM-CaImAn-Python/examples/index.html).

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
    - (optional) Scan-Phase Correction
2. Motion Correction
    - Rigid/non-rigid Registration (single z-plane)
    - Evaluation metrics
    - (optional) Parameter grid-search
    - Register remaining z-planes
3. Segmentation
    - CNMF (single z-plane)
    - (optional) Deconvolution / spike-extraction
    - Refine neuron selection
    - Segment / deconvolve remianing z-planes
4. Collation
    - Collate images and metadata into a single volume
    - Lateral offset correction (between z-planes, COMING SOON)

----------------
## HPC

Slurm utilities are available in the [utilities repository](https://github.com/MillerBrainObservatory/utilities/tree/master/slurm).

**Installation**

```{code-block} bash
git clone https://github.com/MillerBrainObservatory/utilities.git
```

**Transfering data to the HPC**

```{code-block} bash
rsync -avPh /path/to/local/data username@dtn02-hpc.rockefeller.edu:/lustre/fs4/mbo/scratch/<username>/data/ 
```

**Transfering data to the your local machine**

```{code-block} bash
# you will need the ssh config for rbo in your .ssh/config file
rsync -avPh -e "ssh" ./path/to/data rbo:/path/to/destination

```

**Files**

- `multifile_batch.sbatch` - Submit a job to the HPC
- `tunnel.sbatch` - Create a tunnel to the HPC

### `multi_file_batch.sbatch`

- a user folder in the MBO accounts scratch/
- ssh keys for local->hpc and hpc->local transfers
- (windows only) `rsync` [download](https://www.itefix.net/cwrsync) or [with git bash and this utility](https://scicomp.aalto.fi/scicomp/rsynconwindows/).

Login to the hpc on the rocky-9 login node:

```{code-block} bash
ssh username@login-05-hpc.rockefeller.edu
```

Create a backup of your .bashrc file:

```{code-block} bash
cp ~/.bashrc ~/.bashrc.bak
```

Move the bashrc file in the utilities folder you cloned to replace the one you just backed up:

```{code-block} bash
cp utilities/.bashrc ~/.bashrc
```

This file exports some environment variables that are necessary for the pipeline to run.

You will need your files in the `mbo_data`.

After submitting a job via `multifile_batch.sbatch`, you can monitor the job with `squeue -u $USER` and cancel it with `scancel $JOB_ID`.

The resulting batch files will be synced back to your local machine.

### Transfering POSIX->Windows

You can move your batch contents anywhere. However, if moving them to a different operating system, you may encounter
the followng error:

```{code-block} bash
NotImplementedError: cannot instantiate 'PosixPath' on your system
```

You can use [lbm_caiman_python.load_batch_cross_platform](#load_batch_cross_platform) to load this batch item.

----------------

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
