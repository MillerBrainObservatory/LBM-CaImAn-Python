#%%
from pprint import pprint
from dask_image.imread import imread
import napari
import dask_image
from pathlib import Path

try:
    from icecream import ic, install, argumentToString

    install()
except ImportError:  # graceful fallback if icecream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

install()

tiffs = Path("/v-data2/jeff_demas/EXAMPLE DATA/Fig2/20191122/mh89_hemisphere_FOV_50_550um_depth_250mW_dual_stimuli_30min_00001/data")
files = [x for x in tiffs.glob("*.tif*")]

stack = imread(files[0])
stack2 = imread(tiffs)
print('done')
x=4
