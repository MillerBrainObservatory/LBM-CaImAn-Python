import os

import numpy as np

from lbm_caiman_python import read_scan

# demo_reg
# %% [markdown]
# # LBM Step 2: Registration
#
# ## Registration: Correct for rigid/non-rigid movement
#
# - Apply the nonrigid motion correction (NoRMCorre) algorithm for motion correction.
# - View pre/most correction movie
# - Use quality metrics to evaluate registration quality

# %%
from pathlib import Path
import zarr
# import mesmerize_core as mc
import fastplotlib as fpl

try:
    import cv2

    cv2.setNumThreads(0)
except():
    pass

os.environ['WGPU_BACKEND_TYPE'] = 'D3D12'
os.environ["CONDA_PREFIX_1"] = ""
os.environ["QT_PLATFORM_PLUGIN"] = "xcb"

parent_dir = Path().home() / 'caiman_data' / 'animal_01' / 'session_01' / 'save_gui.zarr'
results_path = parent_dir / 'registration'

results_path.mkdir(exist_ok=True, parents=True)
img = zarr.open(parent_dir)['plane_1'][2, ...].astype(np.float32)

# fastplotlib and wgpu will auto-detect if Qt is imported and then use the Qt canvas and Qt output context
fig = fpl.Figure()

fig[0, 0].add_image(img)

# must call fig.show() to start rendering loop and show the QWidget containing the fastplotlib figure
qwidget = fig.show()

# set QWidget initial size from image width and height
qwidget.resize(*img.shape[:2])

# execute Qt app
# if this is part of a larger Qt QApplication, you can also call app.exec() where app is the QApplication instance
fpl.run()

# %%
# mc.set_parent_raw_data_path(parent_dir.parent)
#
# # %%
# # create a new batch
# try:
# # to load existing batches use `load_batch()`
#     df = mc.load_batch(results_path / 'registration.pickle')
# except (IsADirectoryError, FileNotFoundError):
#     df = mc.create_batch(results_path / 'registration.pickle')
# df
#
# # %% [markdown]
# # # Registration Parameters
#
# # %%
# pix_res = 1
#
# mx = 10/pix_res
# max_shifts = (int(mx), int(mx))       # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
# max_deviation_rigid = 3     # maximum deviation allowed for patch with respect to rigid shifts
# pw_rigid = True             # flag for performing rigid or piecewise rigid motion correction
# shifts_opencv = True        # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
# border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
# max_shifts
#
# # %%
# # Set parameters for rigid motion correction
# mcorr_params =\
# { 'main':
#     {
#         'var_name_hdf5': 'plane_1',
#         'max_shifts': max_shifts,   # max shift in pixels
#         'strides': (48, 48),        # create a new patch every x pixels for pw-rigid correction
#                                     # NOTE: "stride" for cnmf, "strides" for mcorr
#         'overlaps': (24, 24),       # overlap between patches (size of patch strides+overlaps)
#         'max_deviation_rigid': 3,   # maximum deviation allowed for patch with respect to rigid shifts
#         'pw_rigid': False,          # flag for performing rigid or piecewise rigid motion correction
#         'shifts_opencv': True,      # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
#         'border_nan': 'copy',       # replicate values along the boundary (if True, fill in with NaN)
#     }
# }
#
# # %%
# if df.empty:
#         # add other param variant to the batch
#     df.caiman.add_item(
#     algo='mcorr',
#     item_name='plane_1',
#     input_movie_path=parent_dir,
#     params=mcorr_params
#     )
#
# df = df.caiman.reload_from_disk()
# df
#
# # %%
# df.iloc[0].caiman.run()
# print(df.iloc[0].outputs['traceback'])
#
# # %%
# df.iloc[0]
#
# # %% [markdown]
# # ### View rigid template
#
# # %%
# # load motion corrected movie
# if __name__ == "__main__":
#
#     path = Path().home() / 'caiman_data' / 'animal_01' / 'session_01'
#     savedir = path / 'pre_processed'
#     reader = read_scan(path, trim_roi_x=(5, 5), trim_roi_y=(17, 0))
#
#
#     x = 5
