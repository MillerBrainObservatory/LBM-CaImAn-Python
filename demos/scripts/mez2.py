#%%
import os
import sys
from copy import deepcopy
import warnings

from pathlib import Path
import mesmerize_core as mc
import matplotlib.pyplot as plt

import scanreader

import pandas as pd
import dask.array as da
from mesmerize_core.caiman_extensions.cnmf import cnmf_cache
import fastplotlib as fpl

import mesmerize_core as mc
import mesmerize_viz
if os.name == "nt":
    # disable the cache on windows
    cnmf_cache.set_maxsize(0)

sys.path.append("../../scanreader/")
warnings.filterwarnings("ignore")
os.environ["CONDA_PREFIX_1"] = ""
os.environ["WGPU_BACKEND"]="gl"

pd.options.display.max_colwidth = 120

## Functions 
def clear_zeros(_scan, rmz_threshold=1e-5):
    non_zero_rows = ~np.all(np.abs(_scan) < rmz_threshold, axis=(0, 2))
    non_zero_cols = ~np.all(np.abs(_scan) < rmz_threshold, axis=(0, 1))
    cleaned = _scan[:, non_zero_rows, :]
    return cleaned[:, :, non_zero_cols]

def trim_scan(scan, amounts_x):
    new_slice_x = [slice(s.start + amounts_x[0], s.stop - amounts_x[1]) for s in scan.fields[0].output_xslices]
    return [i for s in new_slice_x for i in range(s.start, s.stop)]

#%% md
# #### NOTE
# 
# on `cannot open libGL shared library`:
# 
# `sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y`
#%% md
# ### Set up data and save-paths
#%%
parent = Path('/home/mbo/caiman_data/high_res')

mc.set_parent_raw_data_path(parent / 'final')
process_path = mc.get_parent_raw_data_path()
batch_path = mc.get_parent_raw_data_path().joinpath("batch/reg")
save_path = mc.get_parent_raw_data_path().joinpath("results")
raw = [x for x in parent.glob("*.tif*")]
movie_names = [x for x in process_path.glob("*.tif*")]

#reader = scanreader.read_scan(str(raw[0]), join_contiguous=True)
#%% md
# ## Registration Grid Search
#%%
# create a new batch
try:
    df = mc.create_batch(batch_path)
except FileExistsError:
    df = mc.load_batch(batch_path)
    
# set initial params
mcorr_params =\
{
  'main':
    {
        'max_shifts': [4, 4],
        'strides': [48, 48],
        'overlaps': [24, 24],
        'max_deviation_rigid': 3,
        'border_nan': 'copy',
        'pw_rigid': True,
        'gSig_filt': None
    },
}
#%%
# add other param variant to the batch
df.caiman.add_item(
  algo='mcorr',
  item_name=movie_names[0].stem,
  input_movie_path=movie_names[0],
  params=mcorr_params
)

df
#%%
# copy the mcorr_params2 dict to make some changes
new_params = deepcopy(mcorr_params)

# some variants of max_shifts
for shifts in [2, 32]: 
    for strides in [12, 24, 64]:
        overlaps = int(strides / 2)
        # deep copy is the safest way to copy dicts
        new_params = deepcopy(new_params)

        # assign the "max_shifts"
        new_params["main"]["max_shifts"] = (shifts, shifts)
        new_params["main"]["strides"] = (strides, strides)
        new_params["main"]["overlaps"] = (overlaps, overlaps)

        df.caiman.add_item(
          algo='mcorr',
          item_name=movie_names[0].stem,
          input_movie_path=movie_names[0],
          params=new_params
        )
#%%
diffs = df.caiman.get_params_diffs(algo="mcorr", item_name=df.iloc[0]["item_name"])
diffs
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
#%%
df = df.caiman.reload_from_disk()
df
#%%
viz = df.mcorr.viz()
viz.show()
#%%
viz = df.iloc[0,:].mcorr.viz(data_options=["input", "mcorr", "mean", "corr"])
#%%
viz.show()
#%%
viz.close()
#%%
from mesmerize_core import *
import tifffile
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy
import pandas as pd

from fastplotlib import GridPlot, Plot
from ipywidgets.widgets import IntSlider, VBox
from mesmerize_viz.cnmfviewer import CNMFViewer
from mesmerize_viz.mcorrviewer import MCorrViewer
from mesmerize_viz.dataframeviewer import DataframeViewer
from mesmerize_viz.selectviewer import SelectViewer
#%%
mv = MCorrViewer(dataframe=df)
sv = SelectViewer(mv, grid_shape=(2,3))
dfv = DataframeViewer(mv, sv)
#%%
df.iloc[0, 4]
#%%

# first item is just the raw movie
movies = [df.iloc[0].caiman.get_input_movie()]

# subplot titles
subplot_names = ["raw"]

# we will use the mean images later
means = [df.iloc[0].caiman.get_projection("mean")]

# get the param diffs to set plot titles
param_diffs = df.caiman.get_params_diffs("mcorr", item_name=df.iloc[0]["item_name"])

# add all the mcorr outputs to the list
for i, row in df.iterrows():
    # add to the list of movies to plot
    movies.append(row.mcorr.get_output())

    max_shifts = param_diffs.iloc[i]["max_shifts"][0]
    strides = param_diffs.iloc[i]["strides"][0]
    overlaps = param_diffs.iloc[i]["overlaps"][0]
    
    # subplot title to show dataframe index
    subplot_names.append(f"ix {i}: max_sh: {max_shifts}, str: {strides}, ove: {overlaps}")
    
    # mean images which we'll use later
    means.append(row.caiman.get_projection("mean"))

# stack movies using Dask
stacked_movies = da.stack([da.from_array(movie) for movie in movies])


#%% md
# View an example movie
#%%
from napari import Viewer
# create the viewer
viewer = Viewer()
# viewer.add_image(df.iloc[0].caiman.get_input_movie(), name='input', colormap='gray')
viewer.add_image(df.iloc[0].mcorr.get_output(), name='output', colormap='gray')
#%%
viewer.close()
#%%
shifts = df.iloc[0].mcorr.get_shifts()
#%% md
# ## Extras
# 
#%%

savep = parent / "tiff" 

savep.mkdir(exist_ok=True)

def sp(reader, plane):
    idx = plane - 1
    savename = savep / f"extracted_plane_{plane}_v1.tiff"
    print(savename)
    trim_x = trim(reader, (7,7))
    image_og = reader[1, idx, :,:]
    image_trim = reader[1, idx, trim_x, :]
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image_og)
    ax[0].axis('off')
    ax[0].set_title('untrimmed')
    ax[1].imshow(image_trim)
    ax[1].axis('off')
    ax[1].set_title('trimmed')
    plt.show()

sp(reader, 1)