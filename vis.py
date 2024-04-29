#%%
from pathlib import Path

import tifffile
from dask_image.imread import imread

try:
    from icecream import ic, install, argumentToString

    install()
except ImportError:  # graceful fallback if icecream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

install()

tiffs = Path("/data2/fpo/data/")
filepath = tiffs.resolve()

files = [x for x in tiffs.glob("*.tif*")]

stack = imread(files[0])

with open(filepath, 'rb') as fh:
    metadata = read_scanimage_metadata(fh)

stack = stack.reshape()

print('done')
