#%% md
# # Light Beads Microscopy Demo Pipeline 
# 
# ## Pipeline Steps
# 
# ### Pre-Processing:
# - Extract ScanImage metadata
# - Correct Bi-Directional Offset for each ROI
# - Calculates and corrects the MROI seams (IN PROGRESS)
# ### Motion Correction
# 
# - Apply the nonrigid motion correction (NoRMCorre) algorithm for motion correction.
# - View pre/most correction movie
# - Use quality metrics to evaluate registration quality
# 
# ### Segmentation
# 
# - Apply the constrained nonnegative matrix factorization (CNMF) source separation algorithm to extract initial estimates of neuronal spatial footprints and calcium traces.
# - Apply quality control metrics to evaluate the initial estimates, and narrow down to the final set of estimates.
# 
# 
#%% md
# ### Imports and general setup
#%%
import logging
import os
from pathlib import Path

import bokeh.plotting as bpl
import cv2
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import psutil
import scanreader
import zarr

import core.io

# Give this notebook access to the root package
# sys.path.append('../../')  # TODO: Take this out when we upload to pypi

try:
    cv2.setNumThreads(0)
except():
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect

#%% md
# ## Set up a few helper functions for plotting, logging and setting up our environment
#%%

# set up logging
logging.basicConfig(format="{asctime} - {levelname} - [{filename} {funcName}() {lineno}] - pid {process} - {message}",
                    filename=None, 
                    level=logging.WARNING, style="{") # this shows you just errors that can harm your program
                    # level=logging.DEBUG, style="{") # this shows you general information that developers use to trakc their program 
                    # (be careful when playing movies, there will be a lot of debug messages)

# set env variables 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
chan_order = [1, 5, 6, 7, 8, 9, 2, 10, 11, 12, 13, 14, 15, 16, 17, 3, 18, 19, 20, 21, 22, 23, 4, 24, 25, 26, 27, 28, 29, 30] # this is specific to our dataset
chan_order = [x - 1 for x in chan_order]
# subtract 1 from chan order
#%% md
# ## Extract data using scanreader, joining contiguous ROI's, and plot our mean image
# 
# Our ScanReader object contains all of the properties needed to keep track of our raw data. 
# - ScanImage metadata is stored alongside header metadata, this ScanImage specific data is what's needed to assemble frames from constituent ROIs.
# - We calculate the frame rate and time/pixels between scans and ROI's using the following metadata:
# 
# ![frame rate calculation](../../docs/img/FrameRate1eq.png)
#%%
datapath = Path('/data2/fpo/data/')                 # string pointing to directory containing your data
htiffs = [x for x in datapath.glob('*.tif')]         # this accumulates a list of every filepath which contains a .tif file

reader = scanreader.read_scan(str(htiffs[0]), join_contiguous=True, lbm=True)  # this should take < 2s, no actual data is being read yet
print(f"Number of Planes: {reader.num_channels}")
print(f"Number of ROIs: {reader.num_fields}")
print(f"Total frames (single .tiff) {reader.num_frames}")
print(f"Total frames (all .tiffs) {reader.num_requested_frames}")
#%%
data = reader[0]  # this reads the data into memory, this will take a while depending on the size of the data
data = data[: , :, chan_order, :]  # reorder based on a channel mapping
#%%

dataset_name = 'highres'

# create empty zarr store to save on disk and write to
z1 = zarr.open('/data2/fpo/data/{dataset_name}.zarr', mode='w', shape=(data.shape), chunks=(data.shape[0], data.shape[1], data.shape[2], 1), dtype='int16')
z1[:] = data

# for plane in range(0, data.shape[2]):
#     zdata = zarr.array(data[:, :, plane, :], chunks=(data.shape[0], data.shape[1], data.shape[3]), dtype='int16')
#     store[plane] = zdata


#%% md
# ## Scan Phase Correction
# 
# 
# 
# 
# ### *Methods:*
# 
# **1) Linear interpolation**
# 
# **2) Phase - cross correlation**
# 
#%%
## Method 1: Linear Interpolation
plane = 5
time_frame = slice(0, 1000)

roi_1 = reader[0]
slice_plane = roi_1[:,:,plane, time_frame]  # grab one of our planes

core.io.h5.save_zstack('testfile.h5', slice_plane, {'chan_order': chan_order, 'fps' : reader.fps})  # save the slice to a file
#%% md
# ### Phase correction via Linear Phase Interpolation 
#%%
phase_angle = core.util.compute_raster_phase(slice_plane[:,:], reader.temporal_fill_fraction)
corrected_li = core.util.correct_raster(slice_plane, phase_angle, reader.temporal_fill_fraction)
slice_plane.dtype
#%%
corr = core.util.return_scan_offset(roi_1, 1)
corrected_pc = core.util.fix_scan_phase(roi_1, corr, 1)
#%%
def hook(plot, element):
    print('plot.state:   ', plot.state)
    print('plot.handles: ', sorted(plot.handles.keys()))
    plot.handles['xaxis'].axis_label_text_color = 'red'
    plot.handles['yaxis'].axis_label_text_color = 'blue'

image1 = hv.Image(slice_plane[:,:,time_frame]).opts(
    title="Original",
    tools=['pan', 'wheel_zoom'],
    autorange='y', 
    cmap='gray', 
    hooks=[hook]
    )

image2 = hv.Image(corrected_li[:,:,400]).opts(
    title="Corrected - Linear Interpolation",
    tools=['pan', 'wheel_zoom'],
    cmap='gray',
    autorange='y', 
    hooks=[hook]
    )
image3 = hv.Image(corrected_pc[:,:,5,400]).opts(
    width=slice_plane.shape[1],
    height=slice_plane.shape[0],
    title="Corrected - XPhase Correlation",
    tools=['pan', 'wheel_zoom'],
    cmap='gray',
    hooks=[hook],
    )
layout = image1 + image2
layout2 = image1 + image3

# Display the layout
# hv.save(layout, '../docs/img/comparison.html')
bpl.show(hv.render(layout))
#%% md
# ## Join Contiguious ROI's
# 
# Setting `join_contiguous=True` will combine ROI's with the following constraints:
# 1) Must be the same size/shape
# 2) Must be located in the same scanning depth
# 3) Must be located in the same slice
# - ROI can be directly left, right, above or below the adjacent ROI's
#%% md
# ## Motion Correction: CaImAn - NORMCorre
# 
# ### Load pre-processed data as a CaImAn `movie`
#%%
movie = cm.movie(slice_plane, start_time=2, fr=reader.fps)
downsampling_ratio = 0.2  # subsample 5x
movie = movie.resize(fz=downsampling_ratio)
# movie.play(gain=1.3, backend='embed_opencv')
#%% md
# ### View correlation metrics
# 
# Create a couple of summary images of the movie, including:
# - maximum projection (the maximum value of each pixel) 
# - correlation image (how correlated each pixel is with its neighbors)
# 
# If a pixel comes from an active neural component it will tend to be highly correlated with its neighbors.
#%%
max_projection_orig = np.max(movie, axis=0)
correlation_image_orig = cm.local_correlations(movie, swap_dim=False)
correlation_image_orig[np.isnan(correlation_image_orig)] = 0 # get rid of NaNs, if they exist
#%%
f, (ax_max, ax_corr) = plt.subplots(1,2)
ax_max.imshow(max_projection_orig, 
              cmap='viridis',
              vmin=np.percentile(np.ravel(max_projection_orig),50), 
              vmax=np.percentile(np.ravel(max_projection_orig),99.5));
ax_max.set_title("Max Projection Orig", fontsize=12);

ax_corr.imshow(correlation_image_orig, 
               cmap='viridis', 
               vmin=np.percentile(np.ravel(correlation_image_orig),50), 
               vmax=np.percentile(np.ravel(correlation_image_orig),99.5));
ax_corr.set_title('Correlation Image Orig', fontsize=12);
#%% md
# ### Parameter Selection
#%%
max_shifts = (6, 6)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
overlaps = (24, 24)  # overlap between patches (size of patch strides+overlaps)
max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
#%%
parameter_dict = {'fnames': tiffs,
                  'fr': fr,
                  'dxy': dxy,
                  'decay_time': decay_time,
                  'strides': strides,
                  'overlaps': overlaps,
                  'max_shifts': max_shifts,
                  'max_deviation_rigid': max_deviation_rigid,
                  'pw_rigid': pw_rigid,
                  'p': p,
                  'nb': gnb,
                  'rf': rf,
                  'K': K, 
                  'gSig': gSig,
                  'gSiz': gSiz,
                  'stride': stride_cnmf,
                  'method_init': method_init,
                  'rolling_sum': True,
                  'only_init': True,
                  'ssub': ssub,
                  'tsub': tsub,
                  'merge_thr': merge_thr, 
                  'bas_nonneg': bas_nonneg,
                  'min_SNR': min_SNR,
                  'rval_thr': rval_thr,
                  'use_cnn': True,
                  'min_cnn_thr': cnn_thr,
                  'cnn_lowest': cnn_lowest}

parameters = params.CNMFParams(params_dict=parameter_dict) # CNMFParams is the parameters class
print(f"You have {psutil.cpu_count()} CPUs available in your current environment")
num_processors_to_use = None
#%%
fnames = tiffs
#%%
##%% start the cluster (if a cluster already exists terminate it)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='multiprocessing', n_processes=None, single_thread=False)
# create a motion correction object

mc = MotionCorrect(fnames, dview=dview, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid, 
                  shifts_opencv=shifts_opencv, nonneg_movie=True,
                  border_nan=border_nan)

#%%
mov=(cm.movie(mot_correct)).play(magnification=2, fr=reader.fps, q_min=0.1, q_max=99.9)
#%%
reader.fps
#%%
cm.stop_server(dview=cluster)