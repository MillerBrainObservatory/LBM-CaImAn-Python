import os
from pathlib import Path
import dask.array as da
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)

LBM_DEBUG_FLAG = os.environ.get('LBM_DEBUG', 1)

if LBM_DEBUG_FLAG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


def get_zarr_files(directory):
    if not isinstance(directory, (str, os.PathLike)):
        logger.error("iter_zarr_dir requires a single string/path object")
    directory = Path(directory)

    # get directory contents
    contents = [x for x in directory.glob("*") if x.is_dir()]
    return [x for x in directory.glob("*") if x.suffix == ".zarr"]

def save_as_zarr(data, savedir: os.PathLike, planes=None, frames=None, metadata={}, prepend_str='extracted'):
    savedir = Path(savedir)
        
    if not isinstance(planes, (list, tuple)):
        planes = [planes]

    filename = savedir / f'{prepend_str}_plane_{idx}.zarr'
    da.to_zarr(data, filename)