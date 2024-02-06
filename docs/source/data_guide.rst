Spontaneous LBM Data
====================

Collected by Jason Manley and Jeffrey Demas, Aug 2020 - March 2021; and with Sihao Lu, Dec 2022 - April 2023

Data Description
----------------

Configurations
--------------

Spontaneous mouse cortical dynamics were recorded using light beads microscopy (Demas et al. 2021) under various configurations, detailed as follows:

- 3x5x0.5mm FOV, approximately 5um lateral spacing, 4.7 Hz (referred to as "hemisphere" in file names)
- 5.4x6x0.5mm FOV, approximately 5um lateral spacing, 2.2 Hz ("both")
- 1.2x1.2x0.5mm FOV, approximately 2um lateral spacing, 9.6 Hz ("1p2mm")

**Additional Configurations:**

- 3.6x3.6x0.5mm FOV, ~5um lateral spacing, 5.4 Hz ("3p6mm")
- 2x2x0.5mm FOV, ~3um lateral spacing, 6.5 Hz ("2mm")
- 1.8x1.8x0.175mm FOV, ~5um lateral spacing, 18 Hz ("1p8")

Behavior Monitoring
-------------------

During the initial experiments, behavior was simultaneously monitored using two FLIR cameras:

- One camera ('cam-face') was aimed to capture facial movements.
- Another camera ('cam-body') faced the front of the mouse but had slow framerates in many recordings.

For the second round of experiments, three FLIR cameras were used:

- cam0: Positioned beneath the mouse on a transparent treadmill.
- cam1: Captured the right side of the mouse's face for pupil tracking.
- cam2: Focused on the left side of the mouse's face and body.

Synchronization between cameras and imaging data was achieved through LEDs blinking every 5 seconds.

Data Format
-----------

**Raw Data Location:**

Initially stored in Jeff's folders on v-data2 and v-data3, later offloaded to the cloud. Refer to the spreadsheet `spontaneous experiments.xlsx` for detailed locations.

**Processed Data:**

Processed data are available in `/v-data2/jason_manley/EXAMPLE_DATA/*.h5`, containing neural timeseries and aligned behavior traces.

**Variables Include:**

- `Ym`: Mean intensity projections.
- `T_all`: Matrix of extracted neural timeseries.
- `nx, ny, nz`: Vectors of neuron positions with field curvature correction.
- `t`: Timestamps vector.
- `velocity_events`: Indicates the number of velocity events per time bin.
- `motion`: Motion PC timeseries.
- `motion_pcs`: Weights of motion PCs.
- `fhz`: Framerate.

**Behavior and CaImAn Outputs:**

Located in `/v-data2/jason_manley/RECORDING_DATE/RECORDING_DESCRIPTION/`.

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
