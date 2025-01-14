import numpy as np
import argparse
import logging
from pathlib import Path
from pprint import pprint
from functools import partial

import pandas as pd

import lbm_caiman_python as lcp
import mesmerize_core as mc
import caiman as cm

path = Path('e:/datasets/relaxed_resolution')
df = mc.load_batch(path / 'batch.pickle')
mc.set_parent_raw_data_path(path / 'zplanes')

if __name__ == "__main__":
    logger = logging.getLogger("caiman")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    log_format = logging.Formatter(
        "%(relativeCreated)12d [%(filename)s:%(funcName)10s():%(lineno)s] [%(process)d] %(message)s")
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    df.iloc[-1].caiman.run('local')
    df = df.caiman.reload_from_disk()
    pprint(df.iloc[-1].outputs["traceback"])
    print('done')
