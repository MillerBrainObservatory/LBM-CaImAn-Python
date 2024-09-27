#%% md
# # LBM Registration with Mesmerize
# 
# **The visualizations in this notebook will run in [jupyter lab](https://github.com/jupyterlab/jupyterlab#installation), not jupyter notebook. Google colab is not supported either. VS Code notebooks _might_ work but that has not been tested.** See the fastplotlib supported frameworks for more info: https://github.com/fastplotlib/fastplotlib/#supported-frameworks 
#%%
from pathlib import Path
from copy import deepcopy
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import napari

import mesmerize_core as mc
#%%
from mesmerize_core.caiman_extensions.cnmf import cnmf_cache

#%%
if os.name == "nt":
    # disable the cache on windows, this will be automatic in a future version
    cnmf_cache.set_maxsize(0)
#%%
# Mac users!
# temporary patch for Mac, won't be necessary in next release
# Thanks Ryan Ly for the PR! :D I need to dig into it more before merging
# conda_prefix_1_str = os.environ['CONDA_PREFIX'].replace(os.path.join(' ', 'envs', 'mescore')[1:], '')
# os.environ['CONDA_PREFIX_1'] = conda_prefix_1_str
#%%
# This is just a pandas table display formatting option
pd.options.display.max_colwidth = 120
#%% md
# # Paths
# 
# `mesmerize-core` helps manage the outputs of caiman algorithms and organizes "parameter variants" - the output of a given combination of input data and algorithm parameters. In order to run the algorithms you must tell `mesmerize-core` where your _input data_ are located and decide on a **top level raw data directory**. For example consider the following directory structure of experimental data (you may organize your raw data however you wish, this is just an example). We can see that all the experimental data lies under `/data/group_name/my_name/exp_data`. Therefore we can use this `exp_data` dir as a `parent raw data path`. `mesmerize-core` will then only store the _relative_ paths to the raw data files, this allows you to move datasets between computers and filesystems. `mesmerize-core` does not store any hard file paths, only relative paths.
# 
# ```
# /data/group_name/my_name
#                         └── exp_data
#                             ├── axonal_imaging
#                             │   ├── mouse_1
#                             │   │   ├── exp_a.tiff
#                             │   │   ├── exp_b.tiff
#                             │   │   └── exp_c.tiff
#                             │   ├── mouse_2
#                             │   │   ├── exp_a.tiff
#                             │   │   └── exp_b.tiff
#                             │   └── mouse_3
#                             └── hippocampus_imaging
#                                 ├── mouse_1
#                                 │   ├── exp_a.tiff
#                                 │   ├── exp_b.tiff
#                                 │   └── exp_c.tiff
#                                 ├── mouse_2
#                                 └── mouse_3
# ```
# 
# **For this demo set the `caiman_data` dir as the parent raw data path**
# 
# Sidenote: We recommend using [pathlib](https://docs.python.org/3/library/pathlib.html) instead of manually managing paths as strings. `pathlib` is just a part of the Python standard library, it makes it much easier to deal with paths and saves a lot of time in the long-run! It also makes your paths compatible across operating systems. Therefore even if you are on Windows you can use the regular `/` for paths, you do not have to worry about the strangeness of `\\` and `\`
#%%
data_path = Path(r"/mnt/c/Users/RBO/caiman_data")
mc.set_parent_raw_data_path(data_path)
#%% md
# ### Batch path, this is where caiman outputs will be organized
# 
# This can be anywhere, it does not need to be under the parent raw data path.
#%%
batch_path = data_path / "mesmerize-batch/batch.pickle"
#%% md
# # Create a new batch
# 
# This creates a new pandas `DataFrame` with the columns that are necessary for mesmerize. In mesmerize we call this the **batch DataFrame**. You can add additional columns relevant to your experiment, but do not modify columns used by mesmerize.
# 
# Note that when you create a DataFrame you will need to use `load_batch()` to load it later. You cannot use `create_batch()` to overwrite an existing batch DataFrame
#%%
# create a new batch
try:
    df = mc.create_batch(batch_path)
except:
    # to load existing batches use `load_batch()`
    df = mc.load_batch(batch_path)
#%% md
# # Let's add stuff to the DataFrame!
# 
# First get an input movie. An input movie must be somewhere under `parent raw data path`. It does not have to be directly under `parent raw data path`, it can be deeply nested anywhere under it.
#%%
filepath = [x for x in data_path.glob("*tif*")][0]
movie_path = mc.get_parent_raw_data_path().joinpath(str(filepath))
movie_path.is_file()
#%% md
# # Motion correction parameters
# 
# The parameters are passed **directly** to `caiman`, this means you need to use the same exact names for the parameters and you can use all the parameters that you can use with `caiman` - because it's just passing them to `caiman`.
# 
# 
# The parameters dict for a mesmerize batch item must have the following structure. Put all the parameters in a dict under a key called **main**. The **main** dict is then fed directly to `caiman`.
# 
# ```python
# {"main": {... params directly passed to caiman}}
# ```
#%%
# We will start with one version of parameters
mcorr_params1 = \
    {
        'main':  # this key is necessary for specifying that these are the "main" params for the algorithm
            {
                'max_shifts': [24, 24],
                'strides': [48, 48],
                'overlaps': [24, 24],
                'max_deviation_rigid': 3,
                'border_nan': 'copy',
                'pw_rigid': True,
                'gSig_filt': None
            },
    }
#%% md
# # Add a "batch item" to the DataFrame this is the combination of:
# * algorithm to run, `algo`
# * input movie to run the algorithm on, `input_movie_path`
# * parameters for the specified algorithm, `params`
# * a name for you to keep track of things, usually the same as the movie filename, `item_name`
#%%
# add an item to the DataFrame
df.caiman.add_item(
    algo='mcorr',
    input_movie_path=movie_path,
    params=mcorr_params1,
    item_name=movie_path.stem,  # filename of the movie, but can be anything
)

df
#%% md
# We can now see that there is one item in the DataFrame. What we called a "item" in `mesmerize-core` DataFrames is technically called a pandas `Series` or row.
#%% md
# # Run an item
# 
# There is only one item in this DataFrame and it is located at index `0`. You can run a row using `df.iloc[index].caiman.run()`
# 
# Technical notes: On Linux & Mac it will run in subprocess but on Windows it will run in the local kernel.
#%%
df.iloc[0].caiman.run()
#%% md
# # Reload DataFrame from disk
# 
# After running one or any number of items in `mesmerize-core` you must call `df = df.caiman.reload_from_disk()`. This loads the DataFrame with references to the output files in the batch directory.
#%%
df = df.caiman.reload_from_disk()
#%%
df
#%% md
# # Outputs
# 
# We can see that the outputs column has been populated. The entries in this column do not have to be accessed directly. The `mesmerize-core` API allows you to fetch these outputs.
# 
# ```python
# index = 0 # we will fetch stuff from index 0 which we just ran
# 
# # get the motion corrected movie memmap
# mcorr_movie = df.iloc[0].mcorr.get_output()
# mcorr_movie.shape
# ```
# 
# Lazy loading happens with `tifffile.memmap`:
# 
# ```python
# mcorr_memmap_path = df.iloc[0].mcorr.get_output_path()
# mcorr_memmap_path
# ```
# 
# Load mean projections and the input movie:
# 
# ```python
# # mean projection, max and std projections are also available
# mean_proj = df.iloc[0].caiman.get_projection("mean")
# mean_proj.shape
# 
# # the input movie, note that we use `.caiman` here instead of `.mcorr`
# input_movie = df.iloc[0].caiman.get_input_movie()
# input_movie.shape
# ```
#%% md
# # Parameter variants - this is the purpose of mesmerize-core!
# 
# Let's add another row to the DataFrame. We will use the same input movie but different parameters. This is the basis of how we can perform a _parameter gridsearch_.
#%%
mcorr_params = \
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

# add other param variant to the batch
df.caiman.add_item(
    algo='mcorr',
    item_name=movie_path.stem,
    input_movie_path=movie_path,
    params=mcorr_params
)
df
#%% md
# We can see that there are two batch items for the same input movie.
#%% md
# # Parameter Gridsearch
# 
# Use a `for` loop to add multiple different parameter variants more efficiently.
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
            item_name=movie_path.stem,
            input_movie_path=movie_path,
            params=new_params
        )
#%%
df
#%% md
# # Distinguishing parameter variants
# 
# We can see that there are many parameter variants, but it is not easy to see the differences in parameters between the rows that have the same `item_name`.
# 
# We can use the `caiman.get_params_diffs()` to see the unique parameters between rows with the same `item_name`
#%%
diffs = df.caiman.get_params_diffs(algo="mcorr", item_name=df.iloc[0]["item_name"])
diffs
#%% md
# # Run multiple batch items.
# 
# `df.iterrows()` iterates through rows and returns the numerical index and row for each iteration
#%%
for i, row in df.iterrows():
    if row["outputs"] is not None:  # item has already been run
        continue  # skip

    process = row.caiman.run()

    # on Windows you MUST reload the batch dataframe after every iteration because it uses the `local` backend.
    # this is unnecessary on Linux & Mac
    # "DummyProcess" is used for local backend so this is automatic
    if process.__class__.__name__ == "DummyProcess":
        df = df.caiman.reload_from_disk()
#%% md
# # Outputs
# 
# Load the output information into the DataFrame
#%%
df = df.caiman.reload_from_disk()
#%%
df.outputs[0]['traceback']
#%% md
# # Visualization using `mesmerize-viz` 
#%%
mcorr_viz = df.mcorr.viz(
    data_options=["input", "mcorr"],
    image_widget_kwargs={"grid_plot_kwargs": {"size": (1000, 500)}}
    # you can also pass kwargs to the ImageWidget that is created
)
mcorr_viz.show()
#%%
# close when done
mcorr_viz.close()
#%% md
# # Build your own visualizations using `fastplotlib`
# 
# # Use `ImageWidget` to view multiple mcorr results simultaneously
# 
# This type of visualization usually requires your files to be lazy-loadble, and the performance will depend on your hard drive's capabilities.
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

# create the widget
# mcorr_iw_multiple = fpl.ImageWidget(
#     data=movies,  # list of movies
#     window_funcs={"t": (np.mean, 17)}, # window functions as a kwarg, this is what the slider was used for in the ready-made viz
#     grid_plot_kwargs={"size": (900, 700)},
#     names=subplot_names,  # subplot names used for titles
#     cmap="gnuplot2"
# )

# mcorr_iw_multiple.show()
#%% md
# Optionally hide the histogram LUT tool
#%%
# for subplot in mcorr_iw_multiple.gridplot:
#     subplot.docks["right"].size = 0
# #%% md
# # Modify the `window_funcs` at any time. This is what the slider in `mesmerize-viz` does.
# #%%
# mcorr_iw_multiple.window_funcs["t"].window_size = 43
# #%%
# mcorr_iw_multiple.close()
# #%% md
# # # Optional, cleanup DataFrame
# #
# # ix `6` seems to work the best so we will cleanup the DataFrame and remove all other items.
# #
# # Remove batch items (i.e. rows) using `df.caiman.remove_item(<item_uuid>)`. This also cleans up the output data in the batch directory.
# #%%
# # make a list of rows we want to keep using the uuids
# # rows_keep = [df.iloc[6].uuid]
# # rows_keep
# #%% md
# # **Note:** On windows calling `remove_item()` will raise a `PermissionError` if you have the memmap file open. The workaround is to shutdown the current kernel and then use `df.caiman.remove_item()`. For example, you can keep another notebook that you use just for cleaning unwanted mcorr items.
# #
# # There is currently no way to close a `numpy.memmap`: https://github.com/numpy/numpy/issues/13510
# #%%
# # for i, row in df.iterrows():
# #     if row.uuid not in rows_keep:
# #         df.caiman.remove_item(row.uuid)
#
# # df
# #%% md
# # Indices are always reset when you use `caiman.remove_item()`. UUIDs are always preserved.
# #%% md
# # # CNMF
# #
# # Perform CNMF using the mcorr output.
# #
# # Similar to mcorr, put the CNMF params within the `main` key. The `refit` key will perform a second iteration, as shown in the `caiman` `demo_pipeline.ipynb` notebook.
# #%%
# # some params for CNMF
# params_cnmf =\
# {
#     'main': # indicates that these are the "main" params for the CNMF algo
#         {
#             'fr': 30, # framerate, very important!
#             'p': 1,
#             'nb': 2,
#             'merge_thr': 0.85,
#             'rf': 15,
#             'stride': 6, # "stride" for cnmf, "strides" for mcorr
#             'K': 4,
#             'gSig': [4, 4],
#             'ssub': 1,
#             'tsub': 1,
#             'method_init': 'greedy_roi',
#             'min_SNR': 2.0,
#             'rval_thr': 0.7,
#             'use_cnn': True,
#             'min_cnn_thr': 0.8,
#             'cnn_lowest': 0.1,
#             'decay_time': 0.4,
#         },
#     'refit': True, # If `True`, run a second iteration of CNMF
# }
# #%% md
# # # Add CNMF item
# #
# # You can provide the mcorr item row to `input_movie_path` and it will resolve the path of the input movie from the entry in the DataFrame.
# #%%
# good_mcorr_index = 6
#
# # add a batch item
# df.caiman.add_item(
#     algo='cnmf', # algo is cnmf
#     input_movie_path=df.iloc[good_mcorr_index],  # use mcorr output from a completed batch item
#     params=params_cnmf,
#     item_name=df.iloc[good_mcorr_index]["item_name"], # use the same item name
# )
# #%% md
# # See the cnmf item at the bottom of the dataframe
# #%%
# df
# #%% md
# # # Run CNMF
# #
# # The API is identical to running mcorr
# #%%
# index = -1  # most recently added item
# df.iloc[index].caiman.run()
# #%% md
# # # Reload dataframe
# #%%
# df = df.caiman.reload_from_disk()
# df
# #%% md
# # # CNMF outputs
# #
# # Similar to mcorr, you can use the `mesmerize-core` API to fetch the outputs. The API reference for CNMF is here: https://mesmerize-core.readthedocs.io/en/latest/api/cnmf.html
# #%%
# index = -1  # index of the cnmf item, last item in the dataframe
#
# # temporal components
# temporal = df.iloc[index].cnmf.get_temporal()
# #%%
# temporal.shape
# #%% md
# # Many of the cnmf functions take a rich set of arguments
# #%%
# # get accepted or rejected components
# temporal_good = df.iloc[index].cnmf.get_temporal("good")
#
# # shape is [n_components, n_frames]
# temporal_good.shape
# #%%
# # get specific components
# df.iloc[index].cnmf.get_temporal(np.array([1, 5, 9]))
# #%%
# # get temporal with the residuals, i.e. C + YrA
# temporal_with_residuals = df.iloc[index].cnmf.get_temporal(add_residuals=True)
# #%%
# # get contours
# contours = df.iloc[index].cnmf.get_contours()
# #%% md
# # Returns: `(list of np.ndarray of contour coordinates, list of center of mass)`
# #%%
# print(f"contour 0 coordinates:\n\n{contours[0][0]}\n\n com: {contours[1][0]}")
# #%%
# len(contours)
# #%%
# # get_contours() also takes arguments
# contours_good = df.iloc[index].cnmf.get_contours("good")
# #%%
# len(contours_good[0]) # number of contours
# #%% md
# # swap_dim
# #%%
# # get the first contour using swap_dim=True (default)
# swap_dim_true = df.iloc[index].cnmf.get_contours()[0][0]
# #%%
# # get the first contour  with swap_dim=False
# swap_dim_false = df.iloc[index].cnmf.get_contours(swap_dim=False)[0][0]
# #%%
# plt.plot(
#     swap_dim_true[:, 0],
#     swap_dim_true[:, 1],
#     label="swap_dim=True"
# )
# plt.plot(
#     swap_dim_false[:, 0],
#     swap_dim_false[:, 1],
#     label="swap_dim=False"
# )
# plt.legend()
# #%%
# # swap_dim swaps the x and y dims
# plt.plot(
#     swap_dim_true[:, 0],
#     swap_dim_true[:, 1],
#     linewidth=30
# )
# plt.plot(
#     swap_dim_false[:, 1],
#     swap_dim_false[:, 0],
#     linewidth=10
# )
# #%% md
# # # Reconstructed movie - `A * C`
# # # Reconstructed background - `b * f`
# # # Residuals - `Y - AC - bf`
# #
# # Mesmerize-core provides these outputs as lazy arrays. This allows you to work with arrays that would otherwise be hundreds of gigabytes or terabytes in size.
# #%%
# rcm = df.iloc[index].cnmf.get_rcm()
# rcm
# #%% md
# # LazyArrays behave like numpy arrays
# #%%
# rcm[42]
# #%%
# rcm[10:20].shape
# #%% md
# # # Using LazyArrays
# #%%
# rcm_accepted = df.iloc[index].cnmf.get_rcm("good")
# rcm_rejected = df.iloc[index].cnmf.get_rcm("bad")
# #%%
# iw_max = fpl.ImageWidget(
#     data=[rcm_accepted.max_image, rcm_rejected.max_image],
#     names=["accepted", "rejected"],
#     grid_plot_kwargs={"size": (900, 450)},
#     cmap="gnuplot2"
# )
# iw_max.show()
# #%%
# iw_rcm_separated = fpl.ImageWidget(
#     data=[rcm_accepted, rcm_rejected],
#     names=["accepted", "rejected"],
#     grid_plot_kwargs={"size": (900, 450)},
#     cmap="gnuplot2"
# )
# iw_rcm_separated.show()
# #%% md
# # # All CNMF LazyArrays
# #%%
# rcb = df.iloc[index].cnmf.get_rcb()
# residuals = df.iloc[index].cnmf.get_residuals()
# input_movie = df.iloc[index].caiman.get_input_movie()
# #%% md
# # `ImageWidget` accepts arrays that are sufficiently numpy-like
# #%%
# iw_rcm = fpl.ImageWidget(
#     data=[input_movie, rcm, rcb, residuals],
#     grid_plot_kwargs={"size": (800, 600)},
#     cmap="gnuplot2"
# )
# iw_rcm.show()
# #%%
# iw_rcm.close()
# #%% md
# # # Visualize everything with `mesmerize-viz`
# #%%
# viz_cnmf = df.cnmf.viz()
# viz_cnmf.show()
# #%%
# viz_cnmf.close()
# #%% md
# # # Parameter Gridsearch
# #
# # As shown for motion correction, the purpose of `mesmerize-core` is to perform parameter searches
# #%%
# # itertools.product makes it easy to loop through parameter variants
# # basically, it's easier to read that deeply nested for loops
# from itertools import product
#
# # variants of several parameters
# gSig_variants = [4, 6]
# K_variants = [4, 8]
# merge_thr_variants = [0.8, 0.95]
#
# # always use deepcopy like before
# new_params_cnmf = deepcopy(params_cnmf)
#
# # create a parameter grid
# parameter_grid = product(gSig_variants, K_variants, merge_thr_variants)
#
# # a single for loop to go through all the various parameter combinations
# for gSig, K, merge_thr in parameter_grid:
#     # deep copy params dict just like before
#     new_params_cnmf = deepcopy(new_params_cnmf)
#
#     new_params_cnmf["main"]["gSig"] = [gSig, gSig]
#     new_params_cnmf["main"]["K"] = K
#     new_params_cnmf["main"]["merge_thr"] = merge_thr
#
#     # add param combination variant to batch
#     df.caiman.add_item(
#         algo="cnmf",
#         item_name=df.iloc[good_mcorr_index]["item_name"],  # good mcorr item
#         input_movie_path=df.iloc[good_mcorr_index],
#         params=new_params_cnmf
#     )
# #%% md
# # We now have lot of cnmf items
# #%%
# df
# #%% md
# # View the diffs
# #%%
# df.caiman.get_params_diffs(algo="cnmf", item_name=df.iloc[-1]["item_name"])
# #%% md
# # # Run the `cnmf` batch items
# #%%
# for i, row in df.iterrows():
#     if row["outputs"] is not None: # item has already been run
#         continue # skip
#
#     process = row.caiman.run()
#
#     # on Windows you MUST reload the batch dataframe after every iteration because it uses the `local` backend.
#     # this is unnecessary on Linux & Mac
#     # "DummyProcess" is used for local backend so this is automatic
#     if process.__class__.__name__ == "DummyProcess":
#         df = df.caiman.reload_from_disk()
# #%% md
# # # Load outputs
# #%%
# df = df.caiman.reload_from_disk()
# #%%
# df
# #%% md
# # # Visualize with `mesmerize-viz`
# #%%
# viz_cnmf = df.cnmf.viz()
# viz_cnmf.show(sidecar=True)
# #%% md
# # # Caiman docs on component eval
# #
# # https://caiman.readthedocs.io/en/latest/Getting_Started.html#component-evaluation
# #
# # > The quality of detected components is evaluated with three parameters:
# # >
# # > Spatial footprint consistency (rval): The spatial footprint of the component is compared with the frames where this component is active. Other component’s signals are subtracted from these frames, and the resulting raw data is correlated against the spatial component. This ensures that the raw data at the spatial footprint aligns with the extracted trace.
# # >
# # > Trace signal-noise-ratio (SNR): Peak SNR is calculated from strong calcium transients and the noise estimate.
# # >
# # > CNN-based classifier (cnn): The shape of components is evaluated by a 4-layered convolutional neural network trained on a manually annotated dataset. The CNN assigns a value of 0-1 to each component depending on its resemblance to a neuronal soma.
# # >
# # > Each parameter has a low threshold:
# # > - (rval_lowest (default -1), SNR_lowest (default 0.5), cnn_lowest (default 0.1))
# # >
# # > and high threshold
# # >
# # > - (rval_thr (default 0.8), min_SNR (default 2.5), min_cnn_thr (default 0.9))
# # >
# # > A component has to exceed ALL low thresholds as well as ONE high threshold to be accepted.
# #%% md
# # # This rich visualization is still customizable!
# #
# # Public attributes:
# #
# # - `image_widget`: the `ImageWidget` in the visualization
# # - `plot_temporal`: the `Plot` with the temporal
# # - `plot_heatmap`: the `Plot` with the heatmap
# # - `cnmf_obj`: The cnmf object currently being visualized. This object gets saved to disk when you click the "Save Eval to disk" button.
# # - `component_index`: current component index, `int`
# #
# # A few public methods:
# # - `show()` show the visualization
# # - `set_component_index(index: int)` manually set the component index
# #%%
# viz_cnmf.image_widget.cmap = "gray"
# #%%
# viz_cnmf.plot_heatmap
# #%%
# viz_cnmf.plot_heatmap["heatmap"].cmap = "viridis"
# #%%
# viz_cnmf.plot_heatmap["heatmap"].cmap.vmax
# #%%
# viz_cnmf.plot_heatmap["heatmap"].cmap.vmax = 2_000
# #%% md
# # Customize contours
# #%%
# for subplot in viz_cnmf.image_widget.gridplot:
#     subplot["contours"][:].thickness = 1.0
# #%%
# for subplot in viz_cnmf.image_widget.gridplot:
#     subplot["contours"].visible = False
# #%%
# for subplot in viz_cnmf.image_widget.gridplot:
#     subplot["contours"].visible = True
# #%%
# viz_cnmf.plot_temporal["line"].thickness()
# #%%
# viz_cnmf.plot_temporal["line"].thickness = 1
# #%% md
# # # Visualize fewer things
# #%%
# viz_simple = df.cnmf.viz(
#     image_data_options=["input", "rcm"],
# )
# viz_simple.show(sidecar=True)
# #%% md
# # # More customization of kwargs
# #%%
# viz_more_custom = df.cnmf.viz(
#     image_data_options=["input", "rcm", "rcm-max", "corr"],
#     temporal_kwargs={"add_residuals": True},
# )
# #%%
# viz_more_custom.show(sidecar=True)
