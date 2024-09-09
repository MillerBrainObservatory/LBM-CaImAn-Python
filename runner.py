import os
import sys

import logging
import mesmerize_core as mc
from mesmerize_core.caiman_extensions.cnmf import cnmf_cache
from caiman.summary_images import correlation_pnr
import matplotlib.pyplot as plt

import sys
from pathlib import Path
import os
import zarr
import pandas as pd

pd.options.display.max_colwidth = 120

try:
    import cv2

    cv2.setNumThreads(0)
except():
    pass

logging.basicConfig()

os.environ['CONDA_EXE'] = str(Path().home() / 'miniconda3' / 'bin' / 'conda')

if os.name == "nt":
    # disable the cache on windows, this will be automatic in a future version
    cnmf_cache.set_maxsize(0)

raw_data_path = Path().home() / "caiman_data_org"
movie_path = raw_data_path / 'animal_01' / "session_01" / 'plane_1'

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

df.iloc[0].caiman.run()
print(df.iloc[0].outputs['traceback'])
x = 6
