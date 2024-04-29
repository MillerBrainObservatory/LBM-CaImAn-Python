Data Guide
==========

Collected by Jason Manley and Jeffrey Demas, Aug 2020 - March 2021; and with Sihao Lu, Dec 2022 - April 2023

Overview
--------

+-------------+-------+------------------+---------+-----------+
| Condition   | r_thr | pixel_resolution | min_snr | frameRate |
+=============+=======+==================+=========+===========+
| hemisphere  | 0.2   | 5                | 1.4     | 4.69      |
+-------------+-------+------------------+---------+-----------+
| 2mm         | 0.4   | 2.75             | 1.4     | 6.45      |
+-------------+-------+------------------+---------+-----------+
| 0p9mm       | 0.4   | 3                | 1.5     | 36.89     |
+-------------+-------+------------------+---------+-----------+
| 0p6mm       | 0.4   | 1                | 1.5     | 9.61      |
+-------------+-------+------------------+---------+-----------+
| 0p3mm       | 0.4   | 0.5              | 1.4     | 9.61      |
+-------------+-------+------------------+---------+-----------+

Data Description
----------------

ScanImage tiff files store the following kinds of data:

.. table::

    ================ ===========================================================
    Metadata         Description
    ================ ===========================================================
    **pixel resolution**         The image data itself
    **metadata**     Frame-invariant metadata such as the microscope configuration_ and
                     `region of interest`_ descriptions.
    **descriptions** Frame-varying metadata such as timestamps_ and `I2C data`_.
    ================ ===========================================================

Data Format
-----------

- `Ym`: Mean intensity projections.
- `T_all`: Matrix of extracted neural timeseries.
- `nx, ny, nz`: Vectors of neuron positions with field curvature correction.
- `t`: Timestamps vector.
- `velocity_events`: Indicates the number of velocity events per time bin.
- `motion`: Motion PC timeseries.
- `motion_pcs`: Weights of motion PCs.
- `fhz`: Framerate.


Metadata for Sample LBM Dataset
--------------------------------

**Sample TIFF File:**

.. code-block:: text

    path_template_files = ['/v-data4/foconnell/data/lbm/raw/mh89_hemisphere_FOV_50_550um_depth_250mW_dual_stimuli_30min_00001_00001.tif']

**Resolution and Data Shape:**

- XResolution: (1073741824, 447392)
- YResolution: (1073741824, 536870)
- Pixel Resolution XY: (144, 1000)
- Rows Per Strip: 5104

**TIFF Pages:**

.. code-block:: text

    <tifffle.Pages> pages @18918
    --> first / keyframes (5104, 145)
    --> dtype = int16

**Input Data Shape:**

.. code-block:: text

    dim1 = 25320
    dim2 = 5104
    dim3 = 145
    n_planes = 30

**Reshape Advice:**

.. warning::

   Avoid using `.width` and `.height` attributes due to inconsistencies across tiff readers.

Resources
---------

The code used in this analysis is available at the Vaziri Lab GitHub repository:

.. _VaziriGithub: https://github.com/vazirilab/scaling_analysis/
