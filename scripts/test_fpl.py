# Imports

from pathlib import Path
import os
import pandas as pd

import mesmerize_core as mc
import numpy as np
import tifffile
from matplotlib import pyplot as plt
import fastplotlib as fpl

from mesmerize_core.caiman_extensions.cnmf import cnmf_cache

import lbm_caiman_python
import lbm_caiman_python as lcp
import lbm_caiman_python.summary

if os.name == "nt":
    # disable the cache on windows, this will be automatic in a future version
    cnmf_cache.set_maxsize(0)

import matplotlib as mpl

mpl.rcParams.update({
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.subplot.wspace': .01,
    'figure.subplot.hspace': .01,
    'figure.figsize': (12, 8),
    'ytick.major.left': True,
})
jet = mpl.colormaps['jet']
jet.set_bad(color='k')

pd.options.display.max_colwidth = 120

# parent_path = Path().home() / "caiman_data"
parent_path = Path(r'E:\2024-12-21_20-22-11')
data_path = parent_path / "batch"
batch_path = data_path / 'batch.pickle'

if not batch_path.exists():
    print(f'creating batch: {batch_path}')
    df = mc.create_batch(batch_path)
else:
    df = lcp.load_batch(batch_path)

# tell mesmerize where the raw data is
mc.set_parent_raw_data_path(r'E:\datasets\high_resolution\zplanes')

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

# def _calculate_centers(A, dims):
#     def calculate_center(i):
#         ixs = np.where(A[:, i].toarray() > 0.07)[0]
#         return np.array(np.unravel_index(ixs, dims)).mean(axis=1)[::-1]
#
#     # Use joblib to parallelize the center calculation for each column in A
#     # centers = Parallel(n_jobs=-1)(delayed(calculate_center)(i) for i in tqdm(range(A.shape[1]), desc="Calculating neuron center coordinates"))
#
#     return np.array(centers)
#
model = df.iloc[1].cnmf.get_output()
A = model.estimates.A
dims = (model.dims[1], model.dims[0])
# centers = _calculate_centers(A[:, :1000], dims)
import wgpu, pygfx

print(f"WGPU version: {wgpu.__version__}")
print(f"Pygfx version: {pygfx.__version__}")
print(f"Fastplotlib version: {fpl.__version__}")

fig = fpl.Figure()
random_centers = np.random.rand(1000, 2) * 512
ig = fig[0, 0].add_image(A[:, 0].reshape(dims).toarray(), cmap="viridis")
isc = fig[0, 0].add_scatter(random_centers, colors="r", sizes=3, alpha=0.7, size_space="world")
fig.show()

metrics_files = lbm_caiman_python.summary.compute_mcorr_metrics_batch(df, overwrite=True)
metrics_df = lbm_caiman_python.summary.metrics_df_from_files(metrics_files)
summary_df = lbm_caiman_python.summary.compute_mcorr_statistics(df)
