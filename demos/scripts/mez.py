#%%
from copy import deepcopy
import os
os.environ['CONDA_PREFIX_1'] = '' # needed for mesmerize env
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

import mesmerize_core as mc
from mesmerize_core.caiman_extensions.cnmf import cnmf_cache
if os.name == "nt":
    # disable the cache on windows
    cnmf_cache.set_maxsize(0)
print(os.name)

pd.options.display.max_colwidth = 120

raw_tiff_path = Path('/home/mbo/caiman_data/high_res/analysis')
filename = raw_tiff_path / 'extracted_plane_1.tiff'

mc.set_parent_raw_data_path(raw_tiff_path.expanduser())
batch_path = mc.get_parent_raw_data_path().joinpath("batch_fix/reg")

try:
    df = mc.create_batch(batch_path)
except FileExistsError:
    df = mc.load_batch(batch_path)

mcorr_params = \
    {
        'main': # this key is necessary for specifying that these are the "main" params for the algorithm
            {
                'max_shifts': [20, 20],
                'strides': [64, 64],
                'overlaps': [24, 24],
                'max_deviation_rigid': 3,
                'border_nan': 'copy',
                'pw_rigid': True,
            },
    }

# add other param variant to the batch
df.caiman.add_item(
  algo="mcorr",
  item_name=filename.stem,
  input_movie_path=filename,
  params=mcorr_params
)

df.iloc[0].caiman.run()

df = df.caiman.reload_from_disk()

viz = df.mcorr.viz()
viz.show()