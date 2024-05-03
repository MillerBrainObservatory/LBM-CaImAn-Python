#%% md
# # Light Beads Microscopy Demo Pipeline 
# 
# ### Imports and general setup
#%%
##%%
import sys
from pathlib import Path

sys.path.append('../util/')  # TODO: Take this out when we upload to pypi
import scanreader
import util

import bokeh.plotting as bpl
import cv2
import holoviews as hv
from IPython import get_ipython
import logging
import matplotlib.pyplot as plt, mpld3
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except():
    pass

try:
    if __IPYTHON__:
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass

import caiman as cm

bpl.output_notebook()
hv.notebook_extension('bokeh')


#%% md
# ## Set up a few helper functions for plotting, logging and setting up our environment
#%%
def plot_frame(img, title='', savepath='', **kwargs):
    fig, ax = plt.subplots()
    ax.imshow(img, **kwargs)
    fig.suptitle(f'{title}')
    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, **kwargs)
    plt.show()


# set up logging
logging.basicConfig(format="{asctime} - {levelname} - [{filename} {funcName}() {lineno}] - pid {process} - {message}",
                    filename=None,
                    level=logging.WARNING, style="{")  # this shows you just errors that can harm your program
# level=logging.DEBUG, style="{") # this shows you general information that developers use to trakc their program
# (be careful when playing movies, there will be a lot of debug messages)

# set env variables 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#%% md
# ## Extract data using scanreader, joining contiguous ROI's, and plot our mean image
#%%
datapath = Path('/data2/fpo/lbm/3mm_5mm/')  # string pointing to directory containing your data
tiffs = [x for x in datapath.glob('*.tif')]  # this accumulates a list of every filepath which contains a .tif file

# Two readers to demonstrate joining contiguous ROI's and it's effect on the phase scan correction
reader = scanreader.read_scan(str(tiffs[0]), join_contiguous=False)
print(f'Number of ROIs: {len(reader)}')
#%%
roi_1 = reader[0]
slice_plane = roi_1[:, :, 5, :]
slice_frame = slice_plane[:, :, 400]

phase_angle = util.compute_raster_phase(slice_frame, reader.temporal_fill_fraction)
corrected = util.correct_raster(slice_plane, phase_angle, reader.temporal_fill_fraction)
#%%

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(slice_frame[200:300, 5:105])
ax[0].set_xticks([])
ax[0].set_title("Uncorrected")
ax[1].imshow(corrected[200:300, 5:105, 400])
ax[1].set_yticks([])
ax[1].set_title("Corrected")

fig.tight_layout()
plt.show()
#%%
plane = 21
frame = 100
# as soon as field.shape is called, the .tif is scanned to gather the number of pages
num_roi = len(reader)
fig, ax = plt.subplots(ncols=num_roi * 2)
if num_roi == 1:
    num_roi = [num_roi]

for idx, field in enumerate(reader):
    # process this field
    field_data = field[:, :, plane, frame]
    phase_angle = util.compute_raster_phase(field_data, reader.temporal_fill_fraction)
    corrected = util.correct_raster(field_data, phase_angle, reader.temporal_fill_fraction)

    ax[idx].imshow(field_data)
    ax[idx].set_xticks([])
    ax[idx].set_yticks([])

fig.tight_layout()
plt.show()
#%%

#%%
data_plane_21 = data[:, :, 20, 1000:]
mean_img = np.mean(data_plane_21, axis=2)

fig, ax = plt.subplots()
[ax[i].set_xticks([]) for i, a in enumerate(ax)]
[ax[i].set_yticks([]) for i, a in enumerate(ax)]
ax[0].imshow(mean_img[140:160, 55:75])
ax[0].set_title('Uncorrected')

ax[1].imshow(corrected[140:160, 55:75])
ax[1].set_title('Corrected')
plt.tight_layout()
plt.show()
plot_frame(mean_img, title='Raw Mean Image: Plane 21')
#%%
image_raster_phase = util.compute_raster_phase(mean_img, reader.temporal_fill_fraction)
corrected = util.correct_raster(mean_img, image_raster_phase, reader.temporal_fill_fraction)

fig, ax = plt.subplots(2)
[ax[i].set_xticks([]) for i, a in enumerate(ax)]
[ax[i].set_yticks([]) for i, a in enumerate(ax)]
ax[0].imshow(mean_img[140:160, 55:75])
ax[0].set_title('Uncorrected')
ax[1].imshow(corrected[140:160, 55:75])
ax[1].set_title('Corrected')
plt.tight_layout()
plt.show()
#%%
# Create another reader, without joining contiguous fields
reader_noncontig = scanreader.read_scan(str(tiffs[0]), join_contiguous=False)

#%%
for field in reader_noncontig.fields:
    len(field)

# data = reader_noncontig[0]
#%% md
# Display the contents of our array, first and second frame
#%%
reader = scanreader.read_scan(str(tiffs[0]), join_contiguous=True)
num_rois = len(reader)
slice = reader[:]  # [Y X Z T]
#%%
fig, ax = plt.subplots()
ax.imshow(slice[:, :, 3])  # plot the 3rd frame
mpld3.display()
#%%
scan.shape
#%%

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
ax1.imshow(scan[:, :, 21, 1])
ax2.imshow(scan[:, :, 21, 2])
plt.tight_layout()
plt.show()
#%%
# Now, create a reader that stitches together contiguous ROI's
reader_contig = scanreader.read_scan(str(tiffs[0]), join_contiguous=True)
scan_contig = reader_contig[0]
#%% md
# ## Motion Correction
#%%
movie = cm.movie(slice, start_time=2, fr=reader.fps)
downsampling_ratio = 0.2  # subsample 5x
movie = movie.resize(fz=downsampling_ratio)
movie.play(gain=1.3, backend='embed_opencv')
