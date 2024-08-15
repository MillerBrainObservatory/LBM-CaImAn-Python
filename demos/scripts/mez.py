#%%
from copy import deepcopy
import os
os.environ['CONDA_PREFIX_1'] = '' # needed for mesmerize env
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import scanreader
from tifffile import imwrite

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mesmerize_core as mc
from mesmerize_core.caiman_extensions.cnmf import cnmf_cache
if os.name == "nt":
    # disable the cache on windows
    cnmf_cache.set_maxsize(0)
print(os.name)

pd.options.display.max_colwidth = 120

parent = Path('/home/rbo/caiman_data')
raw_tiff_name = parent / 'high_res.tif'
reader = scanreader.read_scan(str(raw_tiff_name), join_contiguous=True)
reader.num_channels

#%%

save =  parent / 'v1'
save2 =  parent / 'v1t'
save2.mkdir(exist_ok=True)

for idx, plane in enumerate(range(1, reader.num_channels+1)):
    print((idx, plane))
    if idx == 0:
        savename = save / f"extracted_plane_{plane}.tiff"
        savename2 = save2 / f"extracted_plane_{plane}.tiff"
        # print(savename)
        # if not savename.is_file():
        #     imwrite(savename, reader[:,:,:,idx,:].squeeze().transpose(2,0,1), photometric='minisblack')
        #     imwrite(savename2, reader[:,:,:,idx,:].squeeze(), photometric='minisblack')

#%%
mc.set_parent_raw_data_path("/home/rbo/caiman_data/v1/")
batch_path = mc.get_parent_raw_data_path().joinpath("mesmerize-batch/batch.pickle")
try:
    df = mc.create_batch(batch_path)
except FileExistsError:
    df = mc.load_batch(batch_path)

movie_path = savename

#%%
mcorr_params1 =\
{
  'main': # this key is necessary for specifying that these are the "main" params for the algorithm
    {
        'max_shifts': [6, 6],
        'strides': [48, 48],
        'overlaps': [24, 24],
        'max_deviation_rigid': 3,
        'border_nan': 'copy',
        'pw_rigid': True,
        'gSig_filt': None
    },
}
#%%
# add an item to the DataFrame
df.caiman.add_item(
    algo='mcorr',
    input_movie_path=movie_path,
    params=mcorr_params1,
    item_name=movie_path.stem,  # filename of the movie, but can be anything
)

df.iloc[0].caiman.run()

df = df.caiman.reload_from_disk()