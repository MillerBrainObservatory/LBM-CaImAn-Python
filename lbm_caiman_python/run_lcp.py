import os
import sys

import logging

import dask.array
import mesmerize_core as mc
import napari
from mesmerize_core.caiman_extensions.cnmf import cnmf_cache
from caiman.summary_images import correlation_pnr
import matplotlib.pyplot as plt

import sys
from pathlib import Path
import os
import zarr
import numpy as np
import pandas as pd
import lbm_caiman_python as lcp

try:
    import cv2

    cv2.setNumThreads(0)
except:
    pass

logging.basicConfig()
if os.name == "nt":
    # disable the cache on windows, this will be automatic in a future version
    cnmf_cache.set_maxsize(0)

pd.options.display.max_colwidth = 120

## Segmentation Path
parent_path = Path().home() / "caiman_data" / "animal_01" / "session_01"

batch_path = parent_path / 'batch.pickle'
mc.set_parent_raw_data_path(str(parent_path))

# you could alos load the registration batch and
# save this patch in a new dataframe (saved to disk automatically)
try:
    df = mc.load_batch(batch_path)
except (IsADirectoryError, FileNotFoundError):
    df = mc.create_batch(batch_path)

df = df.caiman.reload_from_disk()

debug = True

logger = logging.getLogger("caiman")
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
log_format = logging.Formatter(
    "%(relativeCreated)12d [%(filename)s:%(funcName)10s():%(lineno)s] [%(process)d] %(message)s")
handler.setFormatter(log_format)
logger.addHandler(handler)

if debug:
    logging.getLogger("caiman").setLevel(logging.INFO)


old_batch_path = parent_path / 'batch.pickle'
old_batch = mc.load_batch(old_batch_path)
mcorr_old = old_batch.iloc[0]

rf = 20
k = 600 / rf

# general dataset-dependent parameters
fr = 9.62                   # imaging rate in frames per second
decay_time = 0.4            # length of a typical transient in seconds
dxy = (1., 1.)              # spatial resolution in x and y in (um per pixel)

# motion correction parameters
strides = (48, 48)          # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)         # overlap between patches (width of patch = strides+overlaps)
max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
pw_rigid = True             # flag for performing non-rigid motion correction

# CNMF parameters for source extraction and deconvolution
p = 2                       # order of the autoregressive system (set p=2 if there is visible rise time in data)
gnb = 1                     # number of global background components (set to 1 or 2)
merge_thr = 0.80            # merging threshold, max correlation allowed
bas_nonneg = True           # enforce nonnegativity constraint on calcium traces (technically on baseline)
stride_cnmf = 10            # amount of overlap between the patches in pixels (overlap is stride_cnmf+1)
# K = 780                   # number of components per patch
gSig = np.array([6, 6])     # expected half-width of neurons in pixels (Gaussian kernel standard deviation)
gSiz = None #2*gSig + 1     # Gaussian kernel width and hight
method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data see demo_dendritic.ipynb)
ssub = 1                    # spatial subsampling during initialization
tsub = 1                    # temporal subsampling during intialization

# parameters for component evaluation
min_SNR = 1.4               # signal to noise ratio for accepting a component
rval_thr = 0.80             # space correlation threshold for accepting a component
#%%
params_cnmf = {
    'main': {
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
        'K':  k,
        'gSig': gSig,
        'gSiz': gSiz,
        'stride': stride_cnmf,
        'method_init': method_init,
        'rolling_sum': True,
        'use_cnn': False,
        'ssub': ssub,
        'tsub': tsub,
        'merge_thr': merge_thr,
        'bas_nonneg': bas_nonneg,
        'min_SNR': min_SNR,
        'rval_thr': rval_thr,
    },
    'refit': True
}

df = df.caiman.reload_from_disk()

## DEBUG ##
df.caiman.add_item(
    algo='cnmf',
    input_movie_path=df.iloc[0],
    params=params_cnmf,
    item_name=f'batch_cnmf',
)

df.iloc[-1].caiman.run(backend='local', wait=False)

#### Load Segmentation Results

mcorr_movie = df.iloc[0].mcorr.get_output()
cnmf_model = df.iloc[-1].cnmf.get_output()
contours = df.iloc[-1].cnmf.get_contours()

good_masks = df.iloc[-1].cnmf.get_masks('good')
bad_masks = df.iloc[-1].cnmf.get_masks('bad')

combined_masks = np.argmax(good_masks, axis=-1) + 1  # +1 to avoid zero for the background

all_masks = dask.array.stack([mask[..., i] for i, mask in enumerate(good_masks)])

correlation_image = df.iloc[-1].caiman.get_corr_image()

viewer = napari.Viewer()
viewer.add_image(correlation_image, name='Correlation')
viewer.add_labels(combined_masks, name='Combined')
viewer.add_labels(contours[0], name='List')
napari.run()
x = 5
