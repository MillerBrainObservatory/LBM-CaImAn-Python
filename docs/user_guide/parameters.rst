Parameters
##########

For the :ref:`core` functions** in this pipeline, the initial 5 parameters are always the same.

.. note::

    The term "parameter" throughout this guide refers to the inputs to each function.
    For example, running "help convertScanImageTiffToVolume" in the command window will
    show to you and describe the parameters of that function.

The only *required* parameters are your `data_path` and `save_path`.

The rest are optional:

Definitions
================

:code:`data_path` : A filepath leading to the directory that contains your .tiff files.

:code:`save_path` : A filepath leading to the directory where results are saved.

:code:`debug_flag` : Set to 1 to print all files / datasets that would be processed, then stop before any processing occurs.

:code:`overwrite` : Set to 1 to overwrite pre-existing data. Setting to 0 will simply return without processing that file.

:code:`num_cores` : Set to the number of CPU cores to use for parallel computing. Note that even though this is an option in pre-processing, there is actually no parallel computations during this step so the value will be ignored.

To see is demonstrated in the :scpt:`demo_LBM_pipeline` (on github) at the root of this repository.

For information about the parameters unique to each function, see the :ref:`api` or the help documentation for that individual function.

.. _Python: https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB/blob/master/demo_LBM_pipeline.m
