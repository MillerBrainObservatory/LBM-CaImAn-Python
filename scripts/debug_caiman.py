import numpy as np
import argparse
import logging
from pathlib import Path
from pprint import pprint
from functools import partial

import pandas as pd

import lbm_caiman_python as lcp
import lbm_mc as mc


def main():
    parent_path = r"D:\DATA\2025-01-16_GCaMP8s_tdtomato_mk302\mk302_2umpx_17p07hz_224pxby448px_2mroi_200mw_50to550umdeep_GCaMP8s_CavAB_00001"
    raw_path = Path(parent_path) / 'zplanes'
    batch_path = Path(parent_path) / 'registered'
    results_path = batch_path / 'registered.pickle'
    df = mc.load_batch(results_path)

    files = lcp.get_files_ext(raw_path, ".tif", 1)
    files = sorted(files, key=lambda p: int(Path(p).stem.split("_")[-1]))
    mc.set_parent_raw_data_path(raw_path)
    metadata = lcp.get_metadata(files[0])
    params = lcp.params_from_metadata(metadata)

    df.caiman.add_item(
        algo='cnmf',
        input_movie_path=files[0],
        params=params,  # use the same parameters
        item_name=f'cnmf-debug',  # filename of the movie, but can be anything
    )

    df.iloc[-1].caiman.run('local')
    df = df.caiman.reload_from_disk()
    out = df.iloc[-1].outputs["traceback"]
    if not out:
        print('done')
        return False
    else:
        print(out)
        return True


if __name__ == "__main__":
    res = main()
    print(res)
