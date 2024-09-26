#%% md
# # LBM Step 2: Registration
# 
# ## Registration: Correct for rigid/non-rigid movement
# 
# - Apply the nonrigid motion correction (NoRMCorre) algorithm for motion correction.
# - View pre/most correction movie
# - Use quality metrics to evaluate registration quality
#%%
%gui qt
from pathlib import Path
import os
import sys
import numpy as np
import zarr
import pandas as pd

import logging
import mesmerize_core as mc
from mesmerize_viz import *

from mesmerize_core.caiman_extensions.cnmf import cnmf_cache
from caiman.summary_images import correlation_pnr

import napari

import matplotlib.pyplot as plt

sys.path.append('../..')  # TODO: Take this out when we upload to pypi
import lbm_caiman_python as lcp

try:
    import cv2
    cv2.setNumThreads(0)
except():
    pass

logging.basicConfig()

os.environ['WAYLAND_DISPLAY'] = ''
os.environ['RUST_LOG'] = 'info'
os.environ['WINIT_UNIX_BACKEND'] = 'x11'

os.environ["QT_PLATFORM_PLUGIN"] = "xcb"
os.environ["CONDA_PREFIX_1"] = ""

if os.name == "nt":
    # disable the cache on windows, this will be automatic in a future version
    cnmf_cache.set_maxsize(0)

pd.options.display.max_colwidth = 120
#%% md
# ## Logging
#%%
# set up logging
debug = True

logger = logging.getLogger("caiman")
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
log_format = logging.Formatter("%(relativeCreated)12d [%(filename)s:%(funcName)10s():%(lineno)s] [%(process)d] %(message)s")
handler.setFormatter(log_format)
logger.addHandler(handler)

# set env variables
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

if debug:
   logging.getLogger("caiman").setLevel(logging.DEBUG)
#%% md
# ## (optional): View hardware information
#%%
# !pip install cloudmesh-cmd5
!cms help # dont forget to call it after the install as it sets some defaults
!cms sysinfo
#%% md
# ## User input: input data path and plane number
# 
# the same path as [pre_processing](./pre_processing.ipynb)
# parent_dir = Path().home() / 'caiman_data' / 'animal_01' / 'session_01'
#%%
# File locations:
# - batch.pickle parent directory
# - input movie

# parent_path = Path().home() / "caiman_data_org"
parent_path = Path().home() / "caiman_data" / 'animal_01' / 'session_01'
save_path = parent_path / 'motion_correction'

# for TIFF
tiff_path = parent_path / 'tiff'
tiff_files = [x for x in Path(tiff_path).glob('*.tif*')]

# for ZARR
# movie_path = raw_data_path / 'animal_01' / "session_01" / 'plane_1.zarr'
# movie = zarr.open(movie_path)['mov']

reg_batch_path = parent_path / 'batch.pickle'
df = lcp.lbm_load_batch(reg_batch_path, overwrite=False)

df=df.caiman.reload_from_disk()
df
#%%
sorted(tiff_files)
#%% md
# # Default registration parameters
# 
# The parameters are passed **directly** to `caiman`, this means you need to use the same exact names for the parameters and you can use all the parameters that you can use with `caiman` - because it's just passing them to `caiman`.
# 
# The parameters dict for a mesmerize batch item must have the following structure. Put all the parameters in a dict under a key called **main**. The **main** dict is then fed directly to `caiman`.
# 
# ```python
# {"main": {... params directly passed to caiman}}
# ```
#%%
pix_res = 1

mx = 10/pix_res
max_shifts = (int(mx), int(mx))       # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
max_deviation_rigid = 3               # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = True                # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True        # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'                   # replicate values along the boundary (if True, fill in with NaN)

mcorr_params = {
    'main':  # this key is necessary for specifying that these are the "main" params for the algorithm
    {
        'var_name_hdf5': 'mov',
        'max_shifts': max_shifts,
        'strides': [48, 48],
        'overlaps': [24, 24],
        'max_deviation_rigid': 3,
        'border_nan':border_nan,
        'pw_rigid': pw_rigid,
        'gSig_filt': None
    },
}

# # Add a "batch item" to the DataFrame this is the combination of:
# * algorithm to run, `algo`
# * input movie to run the algorithm on, `input_movie_path`
# * parameters for the specified algorithm, `params`
# * a name for you to keep track of things, usually the same as the movie filename, `item_name`
df.caiman.add_item(
    algo='mcorr',
    input_movie_path=tiff_files[0],
    params=mcorr_params,
    item_name=tiff_files[0].stem,  # filename of the movie, but can be anything
)
#%%
df=df.caiman.reload_from_disk()
df
#%% md
# # First registration run: preset with good defaults
# 
# Technical notes: On Linux & Mac it will run in subprocess but on Windows it will run in the local kernel.
#%%
df.iloc[1].caiman.run()
#%% md
# ## Preview Motion Correction
# 
# Before running a grid search for the best parameter set, preview your registration results
#%%
viz = df.mcorr.viz(data_options=["input", "mcorr", "mean", "corr"], start_index=1)
viz.show()
#%%
viewer = napari.Viewer()
viewer.add_image(mcorr_movie, name=f'plane_2_a')
viewer.add_image(mcorr_movie2, name=f'plane_2_b')
#%% md
# # Registration Grid Search (if you need it!)
# 
# More runs with varying parameters, stored on disk in the dataframe batch.pickle
#%%
# copy the mcorr_params2 dict to make some changes
# some variants of max_shifts
from copy import deepcopy

for shifts in [2,32]:
    for strides in [12,64]:
        overlaps = int(strides / 2)
        # deep copy is the safest way to copy dicts
        new_params = deepcopy(mcorr_params)

        # assign the "max_shifts"
        new_params["main"]["pw_rigid"] = True
        new_params["main"]["max_shifts"] = (shifts, shifts)
        new_params["main"]["strides"] = (strides, strides)
        new_params["main"]["overlaps"] = (overlaps, overlaps)

        df.caiman.add_item(
            algo='mcorr',
            input_movie_path=tiff_files[0],
            params=new_params,
            item_name=tiff_files[0].stem,  # filename of the movie, but can be anything
        )

df.caiman.reload_from_disk()
#%%
for i, row in df.iterrows():
    if row["outputs"] is not None: # item has already been run
        continue # skip

    process = row.caiman.run()

    # on Windows you MUST reload the batch dataframe after every iteration because it uses the `local` backend.
    # this is unnecessary on Linux & Mac
    # "DummyProcess" is used for local backend so this is automatic
    if process.__class__.__name__ == "DummyProcess":
        df = df.caiman.reload_from_disk()
#%% md
# 
# # Distinguishing parameter variants
# 
# We can see that there are many parameter variants, but it is not easy to see the differences in parameters between the rows that have the same `item_name`.
# 
# We can use the `caiman.get_params_diffs()` to see the unique parameters between rows with the same `item_name`
#%%
diffs = df.caiman.get_params_diffs(algo="mcorr", item_name=df.iloc[0]["item_name"])
diffs
#%% md
# # Use the varients to organize results to run multiple batch items.
# 
# `df.iterrows()` iterates through rows and returns the numerical index and row for each iteration
#%%
plane = 1

df = df.caiman.reload_from_disk()

# first item is just the raw movie

movies = [df.iloc[0].caiman.get_input_movie()]

subplot_names = ["raw"]

means = [df.iloc[0].caiman.get_projection("mean")]

# get the param diffs to set plot titles
param_diffs = df.caiman.get_params_diffs("mcorr", item_name=df.iloc[0]["item_name"])

# add all the mcorr outputs to the list
for i, row in df.iterrows():

    if row.algo != 'mcorr':
        continue

    # add to the list of movies to plot
    movies.append(row.mcorr.get_output())

    max_shifts = param_diffs.iloc[i]["max_shifts"][0]
    strides = param_diffs.iloc[i]["strides"][0]
    overlaps = param_diffs.iloc[i]["overlaps"][0]

    # subplot title to show dataframe index
    subplot_names.append(f"ix {i}: max_sh: {max_shifts}, str: {strides}, ove: {overlaps}")

    # mean images which we'll use later
    means.append(row.caiman.get_projection("mean"))
#%%
iw_zfish = fpl.ImageWidget(
    data=[movies[0], movies[1]],
    names=['Raw', 'Corrected'],
    cmap="gray",
    histogram_widget=False
)
iw_zfish.show()
#%%
napari.view_image(movies[0], name=subplot_names[0])


# mcorr_movie = df.iloc[0].mcorr.get_output()
# mcorr_movie2 = df.iloc[-1].mcorr.get_output()
# corr, pnr = correlation_pnr(mcorr_movie, swap_dim=False)
# corr2, pnr2 = correlation_pnr(mcorr_movie2, swap_dim=False)
#%%
import napari
viewer = napari.Viewer()
viewer.add_image(mcorr_movie, name=f'plane_2_a')
viewer.add_image(mcorr_movie2, name=f'plane_2_b')
viewer.add_image(corr, name="Mean Correlation")
viewer.add_image(corr2, name="Mean Correlation2")
#%%
napari.view_image(df.iloc[0].mcorr.get_output()[::2, ...])
napari.current_viewer().add_image(df.iloc[0].caiman.get_input_movie()[::2, ...])
#%% md
# ## Correlation metrics
# 
# Create a couple of summary images of the movie, including:
# - maximum projection (the maximum value of each pixel) 
# - correlation image (how correlated each pixel is with its neighbors)
# 
# If a pixel comes from an active neural component it will tend to be highly correlated with its neighbors.
#%%
row_index = i

mean_proj = df.iloc[i].caiman.get_projection("mean")
max_proj = df.iloc[i].caiman.get_projection("max")
std_proj = df.iloc[i].caiman.get_projection("std")
corr_image = df.iloc[i].caiman.get_corr_image()
# viewer.add_image(mcorr_movie, rgb=False, multiscale=False)
# viewer.add_image(input_movie['plane_1'], rgb=False, multiscale=False)

viewer = napari.Viewer()
viewer.add_image(mean_proj)
viewer.add_image(std_proj)
viewer.add_image(max_proj)
viewer.add_image(corr_image)
#%% md
# ## Pixel Shifts
#%%
row_idx = 0

shifts = df.iloc[row_idx].mcorr.get_shifts()
shifts = shifts[1]
shiftsx = [x[0] for x in shifts]
shiftsy = [x[1] for x in shifts]
shiftsx_mean = [np.mean(x) for x in shiftsx]
shiftsy_mean = [np.mean(y) for y in shiftsy]
##%%
xr = list(range(1730))
plt.plot(xr, shiftsx_mean)
plt.title('Mean X-Shifts') # TODO: std error bars, napari layer
plt.xlabel("timestep (frames)")
plt.ylabel("# pixels shufted in X dimension")
plt.show()
#%%
xr = list(range(1730))

plt.title('Mean Y-Shifts') # TODO: std error bars, napari layer
plt.xlabel("timestep (frames)")
plt.ylabel("# pixels shufted in Y dimension")
plt.plot(xr, shiftsy_mean)
#%% md
# # Optional, cleanup DataFrame
# 
# Use the index that works best and all other items.
# 
# Remove batch items (i.e. rows) using `df.caiman.remove_item(<item_uuid>)`. This also cleans up the output data in the batch directory.
# 
# **Note:** On windows calling `remove_item()` will raise a `PermissionError` if you have the memmap file open. The workaround is to shutdown the current kernel and then use `df.caiman.remove_item()`. For example, you can keep another notebook that you use just for cleaning unwanted mcorr items.
# 
# There is currently no way to close a `numpy.memmap`: https://github.com/numpy/numpy/issues/13510
#%% md
# Indices are always reset when you use `caiman.remove_item()`. UUIDs are always preserved.
#%%
df=df.caiman.reload_from_disk()
df
#%%
rows_keep = [2]
for i, row in df.iterrows():
    if i not in rows_keep:
        df.caiman.remove_item(row.uuid, safe_removal=False)
df
#%%
df.caiman.save_to_disk()
#%%

#%% md
# ## Evaluate Results: Optical Flow
#%%
import caiman as cm
#%%
# fnames = [df.iloc[0].mcorr.get_]
# fnames = [str(df.iloc[0].mcorr.get_output_path())]
fnames = str(movie_path)
fnames_rig = str(df.iloc[0].mcorr.get_output_path())

#% compute metrics for the results (TAKES TIME!!)
final_size = np.subtract(movie[1,:,:].shape, 2 * 2) # remove pixels in the boundaries
winsize = 100
swap_dim = False
resize_fact_flow = .2    # downsample for computing ROF
#%%
%%capture

tmpl_orig, correlations_orig, flows_orig, norms_orig, crispness_orig = cm.motion_correction.compute_metrics_motion_correction(
    fnames[0], final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

#%%
tmpl_rig, correlations_rig, flows_rig, norms_rig, crispness_rig = cm.motion_correction.compute_metrics_motion_correction(
    fnames_rig[0], final_size[0], final_size[1],
    swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

# tmpl_els, correlations_els, flows_els, norms_els, crispness_els = cm.motion_correction.compute_metrics_motion_correction(
#     mc.fname_tot_els[0], final_size[0], final_size[1],
#     swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
#%%
fpath = cm.paths.fname_derived_presuffix(str(fnames), 'metrics', swapsuffix='npz')
fpath
#%%
##%% plot the results of Residual Optical Flow
fls = [cm.paths.fname_derived_presuffix(str(fnames), 'metrics', swapsuffix='npz'), cm.paths.fname_derived_presuffix(str(fnames_rig), 'metrics', swapsuffix='npz')]

plt.figure(figsize = (20,10))
for cnt, fl, metr in zip(range(len(fls)), fls, ['raw','corrected',]):
    print('fl')
    if Path(fl).suffix == '.npz':
        with np.load(str(fl)) as ld:
            print(str(np.mean(ld['norms'])) + '+/-' + str(np.std(ld['norms'])) +
                ' ; ' + str(ld['smoothness']) + ' ; ' + str(ld['smoothness_corr']))
            plt.subplot(len(fls), 3, 1 + 3 * cnt)
            plt.ylabel(metr)
            try:
                mean_img = np.mean(cm.load(fl[:-12] + '.tif'))[12:-12, 12:-12]
            except:
                try:
                    mean_img = np.mean(
                        cm.load(fl[:-12] + '.tif'), 0)[12:-12, 12:-12]
                except:
                    try:
                        mean_img = np.mean(
                            cm.load(fl[:-12] + '.hdf5'), 0)[12:-12, 12:-12]
                    except:
                        try:
                            mean_img = np.mean(cm.load(fl[:-12] + '.zarr'), 0)[12:-12, 12:-12]
                        except:
                            print(fl[:-12] + '.zarr')



            lq, hq = np.nanpercentile(mean_img, [.5, 99.5])
            plt.imshow(mean_img, vmin=lq, vmax=hq)
            plt.title('Mean Optical Flow')
            plt.subplot(len(fls), 3, 3 * cnt + 2)
            plt.imshow(ld['img_corr'], vmin=0, vmax=.35)
            plt.title('Corr image')
            plt.subplot(len(fls), 3, 3 * cnt + 3)
            flows = ld['flows']
            plt.imshow(np.mean(
            np.sqrt(flows[:, :, :, 0]**2 + flows[:, :, :, 1]**2), 0), vmin=0, vmax=0.3)
            plt.colorbar()


            lq, hq = np.nanpercentile(mean_img, [.5, 99.5])
            plt.imshow(mean_img, vmin=lq, vmax=hq)
            plt.title('Mean Optical Flow')
            plt.subplot(len(fls), 3, 3 * cnt + 2)
            plt.imshow(ld['img_corr'], vmin=0, vmax=.35)
            plt.title('Corr image')
            plt.subplot(len(fls), 3, 3 * cnt + 3)
            flows = ld['flows']
            plt.imshow(np.mean(
            np.sqrt(flows[:, :, :, 0]**2 + flows[:, :, :, 1]**2), 0), vmin=0, vmax=0.3)
            plt.colorbar()
#%%
plt.figure(figsize = (20,20))
plt.subplot(211); plt.plot(correlations_orig); plt.plot(correlations_rig);
plt.legend(['Original','Rigid','PW-Rigid'])
plt.subplot(223); plt.scatter(correlations_orig, correlations_rig); plt.xlabel('Original');
plt.ylabel('Rigid'); plt.plot([0.3,0.7],[0.3,0.7],'r--')
axes = plt.gca(); axes.set_xlim([0.3,0.7]); axes.set_ylim([0.3,0.7]); plt.axis('square');
#%%
