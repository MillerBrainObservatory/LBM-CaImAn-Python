from pathlib import Path
import zarr
import mesmerize_core as mc
from mesmerize_viz import *
from fastplotlib import GridPlot, ImageWidget, Plot
from ipywidgets.widgets import IntSlider, VBox
from mesmerize_viz.cnmfviewer import CNMFViewer
from mesmerize_viz.mcorrviewer import MCorrViewer
from mesmerize_viz.dataframeviewer import DataframeViewer
from mesmerize_viz.selectviewer import SelectViewer

try:
    import cv2
    cv2.setNumThreads(0)
except():
    pass

parent_dir = Path().home() / 'caiman_data' / 'animal_01' / 'session_01'
results_path = parent_dir / 'motion_corrected'

results_path.mkdir(exist_ok=True, parents=True)  
zarr.open(parent_dir).info

# %%
mc.set_parent_raw_data_path(parent_dir.parent)

# %%
# create a new batch
try:
# to load existing batches use `load_batch()`
    df = mc.load_batch(parent_dir / 'batch.pickle')
except (IsADirectoryError, FileNotFoundError):
    df = mc.create_batch(parent_dir / 'batch.pickle')
df

# %% [markdown]
# # Registration Parameters

# %%
pix_res = 1

mx = 10/pix_res             
max_shifts = (int(mx), int(mx))       # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
max_deviation_rigid = 3     # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = True             # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True        # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
max_shifts

# %%
# Set parameters for rigid motion correction
mcorr_params =\
{ 'main':
    {
        'var_name_hdf5': 'plane_1',
        'max_shifts': max_shifts,   # max shift in pixels
        'strides': (48, 48),        # create a new patch every x pixels for pw-rigid correction
                                    # NOTE: "stride" for cnmf, "strides" for mcorr
        'overlaps': (24, 24),       # overlap between patches (size of patch strides+overlaps)
        'max_deviation_rigid': 3,   # maximum deviation allowed for patch with respect to rigid shifts
        'pw_rigid': False,          # flag for performing rigid or piecewise rigid motion correction
        'shifts_opencv': True,      # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
        'border_nan': 'copy',       # replicate values along the boundary (if True, fill in with NaN)
        'nonneg_movie': False,
    }
}

# %%
if df.empty:
        # add other param variant to the batch
    df.caiman.add_item(
    algo='mcorr',
    item_name='plane_1',
    input_movie_path=parent_dir,
    params=mcorr_params
    )

df = df.caiman.reload_from_disk()
df


viz = df.mcorr.viz(data_options=["input", "mcorr", "mean", "corr"], start_index=1)
viz.show()

# %%
df.iloc[0].caiman.run()
print(df.iloc[0].outputs['traceback'])

# %%
df.iloc[0]