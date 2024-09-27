from pathlib import Path
from copy import deepcopy
import os
import zarr
import dask.array as da
from functools import partial
import glfw
import numpy as np
import pandas as pd
import tifffile
from ipywidgets import IntSlider, VBox
import fastplotlib as fpl

from caiman.motion_correction import high_pass_filter_space
from caiman.summary_images import correlation_pnr

import mesmerize_core as mc
from mesmerize_core.arrays import LazyTiff
from mesmerize_viz import *
#%%
from mesmerize_core.caiman_extensions.cnmf import cnmf_cache

if os.name == "nt":
    # disable the cache on windows, this will be automatic in a future version
    cnmf_cache.set_maxsize(0)
#%%
parent_path = Path().home() / "caiman_data_org"
movie_path = parent_path / 'animal_01' / 'session_01' / 'plane_1.zarr'

batch_path = parent_path / 'batch.pickle'
mc.set_parent_raw_data_path(str(parent_path))

# you could alos load the registration batch and
# save this patch in a new dataframe (saved to disk automatically)
try:
    df = mc.load_batch(batch_path)
except (IsADirectoryError, FileNotFoundError):
    df = mc.create_batch(batch_path)

df=df.caiman.reload_from_disk()
df
#%%
def read_zarr(path):
    # return zarr.open(path)['mov']
    return da.from_zarr(path, 'mov').compute()

#%%
filt = lambda x: high_pass_filter_space(x, (3, 3))

funcs = {
    0: filt,
    1: filt
}

os.environ['WAYLAND_DISPLAY'] = ''
os.environ['RUST_LOG'] = 'info'
os.environ['WINIT_UNIX_BACKEND'] = 'x11'

# viewer = napari.Viewer()
viz_mcor = df.mcorr.viz(
    input_movie_kwargs={"reader": read_zarr},
    image_widget_kwargs={"frame_apply": funcs}
)
viz_mcor.show()

x = 5