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
from functools import partial
from pathlib import Path

print = partial(print, flush=True)


def run_batch(ops):
    print('Running batch ----')
    params_cnmf = ops['params_cnmf']
    albo = ops['algo']
    df = get_batch_from_path(ops['df_path'])
    # 1. registration. check for registration results
    df.caiman.add_item(
        algo='cnmf',
        input_movie_path=df.iloc[0],
        params=params_cnmf,
        item_name=f'batch_cnmf',
    )
    # df.iloc[-1].caiman.run(backend='local', wait=False)


def run_lcp(ops=None, db=None):
    if db is None:
        db = {}
    if ops is None:
        ops = {}
    print('running lcp')
    print(f'ops {ops}')
    print(f'db {db}')
    return None


def get_batch_from_path(batch_path):
    """
    Load or create a batch at the given batch_path.
    """
    try:
        df = mc.load_batch(batch_path)
        print(f'Batch found at {batch_path}')
    except (IsADirectoryError, FileNotFoundError):
        print(f'Creating batch at {batch_path}')
        df = mc.create_batch(batch_path)
    return df


def run_plane(ops, args_path=None):
    print('Running plane ----')
    print(ops)
    print(args_path)
    return None


def run_lcp():
    print('Running lcp ----')
    return None

#### Load Segmentation Results

# mcorr_movie = df.iloc[0].mcorr.get_output()
# cnmf_model = df.iloc[-1].cnmf.get_output()
# contours = df.iloc[-1].cnmf.get_contours()
#
# good_masks = df.iloc[-1].cnmf.get_masks('good')
# bad_masks = df.iloc[-1].cnmf.get_masks('bad')
#
# combined_masks = np.argmax(good_masks, axis=-1) + 1  # +1 to avoid zero for the background
# all_masks = dask.array.stack([mask[..., i] for i, mask in enumerate(good_masks)])
# correlation_image = df.iloc[-1].caiman.get_corr_image()
