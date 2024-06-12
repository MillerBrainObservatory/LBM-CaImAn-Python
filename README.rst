########################################################################
LBM-CaImAn-Python
########################################################################

|docs|

Python implementation of the Light Beads Microscopy (LBM) computational pipeline.

For the `MATLAB` implementation, see `here <https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB/>`_

Pipeline Steps:
===========================

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

Requirements
=============

- caiman
- numpy
- scipy

.. note::

   See the `environment.yml` file at the root of this project for a complete list of package dependencies.


.. |docs| image:: https://img.shields.io/badge/LBM%20Documentation-1f425f.svg
   :target: https://millerbrainobservatory.github.io/LBM-CaImAn-Python/

.. |DOI| image:: https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg
      :target: https://doi.org/10.1038/s41592-021-01239-8
