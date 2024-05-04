import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append("../util/")  # TODO: Take this out when we upload to pypi
sys.path.append("../exclude/")  # TODO: Take this out when we upload to pypi
import scanreader
import util

import bokeh.plotting as bpl
import holoviews as hv
from IPython import get_ipython
import logging
import matplotlib.pyplot as plt

try:
    cv2.setNumThreads(0)
except ():
    pass

try:
    if __IPYTHON__:
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")
except NameError:
    pass

import caiman as cm

# bpl.output_notebook()
# hv.notebook_extension('bokeh')


# %% md
## Set up a few helper functions for plotting, logging and setting up our environment
# %%
def plot_frame(img, title="", savepath="", **kwargs):
    fig, ax = plt.subplots()
    ax.imshow(img, **kwargs)
    fig.suptitle(f"{title}")
    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, **kwargs)
    plt.show()


# set up logging
logging.basicConfig(
    format="{asctime} - {levelname} - [{filename} {funcName}() {lineno}] - pid {process} - {message}",
    filename=None,
    level=logging.WARNING,
    style="{",
)  # this shows you just errors that can harm your program
# level=logging.DEBUG, style="{") # this shows you general information that developers use to trakc their program
# (be careful when playing movies, there will be a lot of debug messages)

# set env variables
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
chan_order = np.array(
    [
        1,
        5,
        6,
        7,
        8,
        9,
        2,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        3,
        18,
        19,
        20,
        21,
        22,
        23,
        4,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
    ]
)  # this is specific to our dataset
# %% md
## Extract data using scanreader, joining contiguous ROI's, and plot our mean image



# %%
datapath = Path("/data2/fpo/data/")  # string pointing to directory containing your data
tiffs = [
    x for x in datapath.glob("*.tif")
]  # this accumulates a list of every filepath which contains a .tif file

reader = scanreader.read_scan(str(tiffs[0]), join_contiguous=False)
print(f"Number of Planes: {reader.num_channels}")
print(f"Number of ROIs: {reader.num_fields}")
print(f"Total frames (single .tiff) {reader.num_frames}")
print(f"Total frames (all .tiffs) {reader.num_requested_frames}")
# %% md
## Scan Phase Correction

# %%
roi_1 = reader[0]
slice_plane = roi_1[:, :, 5, :]
phase_angle = util.compute_raster_phase(
    slice_plane[:, :, 400], reader.temporal_fill_fraction
)
corrected_li = util.correct_raster(
    slice_plane, phase_angle, reader.temporal_fill_fraction
)
# %%
from bokeh.models import Range1d


def bounds_hook(plot, elem, xbounds=None, ybounds=None):
    x_range = plot.handles["plot"].x_range
    y_range = plot.handles["plot"].y_range
    if xbounds is not None:
        x_range.bounds = xbounds
    else:
        x_range.bounds = x_range.start, x_range.end
    if ybounds is not None:
        y_range.bounds = ybounds
    else:
        y_range.bounds = y_range.start, y_range.end


aspect_ratio = slice_plane.shape[1] / slice_plane.shape[0]

range_x = Range1d(start=0, end=slice_plane.shape[1])
range_y = Range1d(start=0, end=slice_plane.shape[0])

plot_width = 600
plot_height = int(plot_width / aspect_ratio)

image1 = hv.Image(slice_plane[:, :, 400]).opts(
    width=plot_width,
    height=plot_height,
    title="Original Image",
    tools=["hover", "pan", "wheel_zoom"],
    cmap="gray",
    hooks=[bounds_hook],
)

image2 = hv.Image(corrected_li[:, :, 400]).opts(
    width=plot_width,
    height=plot_height,
    title="Corrected Image",
    tools=["hover", "pan", "wheel_zoom"],
    cmap="gray",
    hooks=[bounds_hook],
)

# Combine the images into a layout
layout = image1 + image2

# Display the layout
# hv.save(layout, '../docs/img/comparison.html')
bpl.show(hv.render(layout))
# %%
print(image1)
# %%
ph = util.return_scan_offset(roi_1, 1)
corrected_pc = util.fix_scan_phase(roi_1, ph, 1)
# %%
image1 = hv.Image(roi_1[:, :, 4, 100]).opts(
    width=plot_width,
    height=plot_height,
    title="Original Image",
    tools=["hover", "pan", "wheel_zoom"],
    cmap="gray",
    hooks=[bounds_hook],
)

image2 = hv.Image(corrected_pc[:, :, 4, 100]).opts(
    width=plot_width,
    height=plot_height,
    title="Corrected Image",
    tools=["hover", "pan", "wheel_zoom"],
    cmap="gray",
    hooks=[bounds_hook],
)

layout = image1 + image2

# Display the layout
# hv.save(layout, '../docs/img/comparison.html')
bpl.show(hv.render(layout))
# %% md
## Join Contiguious ROI's

# Setting `join_contiguous=True` will combine ROI's with the following constraints:
# 1) Must be the same size/shape
# 2) Must be located in the same scanning depth
# 3) Must be located in the same slice
#                                - ROI can be directly left, right, above or below the adjacent ROI's
# %%
# Create another reader, without joining contiguous fields
contig = scanreader.read_scan(str(tiffs[0]), join_contiguous=True)
num_roi = len(contig)  # we now have a single ROI due to the merging
data = contig[0]
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(data[:, :, 5, 400])
ax[1].imshow(data[100:160, 125:165, 5, 400])
plt.tight_layout()
plt.show()
# %%
do_stats = False
if do_stats:
    result_dict = {}

# %% md
## Motion Correction: CaImAn - NORMCorre
# %%

slice = reader[0][:, :, 5, :]
movie = cm.movie(slice, start_time=2, fr=reader.fps)
downsampling_ratio = 0.2  # subsample 5x
movie = movie.resize(fz=downsampling_ratio)
# movie.play(gain=1.3, backend='embed_opencv')
# %% md
max_projection_orig = np.max(movie, axis=0)
correlation_image_orig = cm.local_correlations(movie, swap_dim=False)
correlation_image_orig[
    np.isnan(correlation_image_orig)
] = 0  # get rid of NaNs, if they exist
# %%
f, (ax_max, ax_corr) = plt.subplots(1, 2, figsize=(12, 6))
ax_max.imshow(
    max_projection_orig,
    cmap="viridis",
    vmin=np.percentile(np.ravel(max_projection_orig), 50),
    vmax=np.percentile(np.ravel(max_projection_orig), 99.5),
)
ax_max.set_title("Max Projection Orig", fontsize=12)

ax_corr.imshow(
    correlation_image_orig,
    cmap="viridis",
    vmin=np.percentile(np.ravel(correlation_image_orig), 50),
    vmax=np.percentile(np.ravel(correlation_image_orig), 99.5),
)
ax_corr.set_title("Correlation Image Orig", fontsize=12)
# %%
