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
raw_data_path = Path().home() / "caiman_data_org"
movie_path = raw_data_path / 'animal_01' / "session_01" / 'plane_1.zarr'

parent_dir = Path().home() / 'caiman_data_org' / 'animal_01' / 'session_01'
raw_tiff_files = [x for x in parent_dir.glob("*.tif*")]
save_path = parent_dir / 'tiff'
save_path.mkdir(exist_ok=True)

scan = lcp.read_scan(raw_tiff_files)

scan.trim_x = (5,5)
scan.trim_y = (17, 0)

len(raw_tiff_files)
lcp.save_as_tiff(scan, savedir=save_path)
lcp.lbm_load_batch()
# moviepath
raw_movie = zarr.open(movie_path).info
raw_movie

batch_path = raw_data_path / 'batch.pickle'
mc.set_parent_raw_data_path(str(movie_path))

# create a new batch
try:
    df = mc.load_batch(batch_path)
except (IsADirectoryError, FileNotFoundError):
    df = mc.create_batch(batch_path)

df = df.caiman.reload_from_disk()

# set up logging
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

# df.iloc[0].caiman.run(backend='local', wait=False)
# x = 6

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