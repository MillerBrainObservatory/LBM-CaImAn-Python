====================
Spontaneous LBM Data
====================

Collected by Jason Manley and Jeffrey Demas, Aug 2020 - March 2021; and with Sihao Lu, Dec 2022 - April 2023

----------------
Data description
----------------

Configurations
--------------

Spontaneous mouse cortical dynamics recorded for 30min. - 60min. using light beads microscopy (Demas et al. 2021) in the following configurations:
- 3x5x0.5mm FOV, ~5um lateral spacing, 4.7 Hz (referred to as "hemisphere" in file names)
- 5.4x6x0.5mm FOV, ~5um lateral spacing, 2.2 Hz ("both")
- 1.2x1.2x0.5mm FOV, ~2um lateral spacing, 9.6 Hz ("1p2mm")

Additional one-off configurations:
- 3.6x3.6x0.5mm FOV, ~5um lateral spacing, 5.4 Hz ("3p6mm")
- 2x2x0.5mm FOV, ~3um lateral spacing, 6.5 Hz ("2mm")
- 1.8x1.8x0.175mm FOV, ~5um lateral spacing, 18 Hz ("1p8")

For the initial round of experiments, behavior was monitored simultaneously utilizing two FLIR cameras. One camera ('cam-face') was pointed on one side of the mouse, primarily to capture the facial movements. Another camera ('cam-body') was pointed straight at the front of mouse, but faced issues with slow framerates in many recordings which will make alignment with behavior difficult. They can be synchronized by monitoring the LEDs by the mouse's head which blink every 5 seconds; the first blink marks the start of the recording.

For the second round of experiments, behavior was monitored simultaneosuly with three FLIR cameras: cam0 (from beneath the mouse, which was on a transparent treadmill), cam1 (right side of mouse's face, for pupil tracking), cam2 (left side of mouse's face and body). The three cameras where triggered simultaneously via hardware connections and again synchronized with imaging data by monitoring LEDs.

Data Format
-----------

Raw data for the inital round were located in Jeff's folders on v-data2 and v-data3, but have since been offloaded to the cloud. Some of the raw data for the second round have been offloaded, while the remaining are on /v-data2/jason_manley/. For the raw data locations and more information on the recordings, see spontaneous experiments.xlsx

Single files containing the neural timeseries and aligned behavior traces are contained in /v-data2/jason_manley/EXAMPLE_DATA/*.h5. Note these have not yet been created for any experiments from the second round. These contain the following variables:
- Ym  : WxHx30 mean intensity projections over the entire recording for each of the 30 axial planes (potentially missing for some mice currently)
- T_all : TxN matrix of extracted neural timeseries
- nx, ny, nz : Nx1 vectors of the x, y, and z positions of each neuron (with field curvature correction)
- t : Tx1 vector of timestamps
- velocity_events : Tx1 vector indicated the number of velocity events logged from the treadmill rotary encoder in each time bin
- motion : Tx500 matrix of the first 500 motion PC timeseries, aligned to the neural activity using linear interpolation
- motion_pcs : HxWx500 matrix of the weights of the first 500 motion PCs, reshaped to the original HxW image size
- fhz : framerate

The original behavior and CaImAn output files are contained for each mouse in /v-data2/jason_manley/RECORDING_DATE/RECORDING_DESCRIPTION/, which contain the following folders:
- ./analyses* contains outputs of all analyses from the paper
- ./caiman-output/collated_caiman_output*.mat contains the collated neural traces, with potentially varying minSNR thresholds. Individual plane caiman outputs in ./caiman-output/indiv_planes/
- ./cam-body and ./cam-face contain the body and face camera videos in the initial round; otherwise ./cam0, ./cam1, and ./cam2 contain the behavior videos
- ./facemap-output contains the results of facemap on the cam-face videos for the initial round; ./multicam-aligned contains the aligned facemap outputs of the three behavior videos for the second round
- ./experiment_log.txt (or something similar) contains the arduino logs from our treadmill control software: github.com/vazirilab/Treadmill_control
- ./pulse_config*.mat contains the LBM configuration file
- ./YYYY-MM-DD-HH-MM timestamped folders contain the stimulation log for visual experiments


Metadata for sample LBM dataset
-------------------------------

path_template_files = ['/v-data4/foconnell/data/lbm/raw/mh89_hemisphere_FOV_50_550um_depth_250mW_dual_stimuli_30min_00001_00001.tif']

XResolution = (1073741824, 447392)
YResolution = (1073741824, 536870)
pixelResolutionXY = (144, 1000)
RowsPerStrip = 5104

<tifffle.Pages> pages @18918
--> first / keyframes (5104, 145)
--> dtype = int16

Input data shape:
dim1 = {int} 25320
dim2 = {int} 5104
dim3 = {int} 145
n_planes = 30

Reshape:
dim1 = dim1 / n_planes  .. t
dim2 = n_planes # .. z 
dim3 = dim1  # .. x
dim4 = dim2  # .. y


.. warning:: 
   Avoid using `.width` and `.height` attributes as they are inconsistent accross (and within) tiff readers.
   

Resources
---------

.. [4] Code used in this analysis is available at the internal 
   Vaziri Github Account: VaziriGithub_.
▎
.. [VaziriGithub] https://github.com/vazirilab/scaling_analysis/

