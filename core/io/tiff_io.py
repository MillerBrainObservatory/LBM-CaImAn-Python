
import os
from pathlib import Path
import dask.array as da
from logging import getLogger

logger = getLogger(__name__)


def tiffs2zarr(filenames, zarrurl, chunksize, **kwargs):
    """Write images from sequence of TIFF files as zarr."""
    with tifffile.TiffFile(filenames) as tifs:
        with tifs.aszarr() as store:
            da = da.from_zarr(store)
            chunks = (chunksize,) + da.shape[1:]
            da.rechunk(chunks).to_zarr(zarrurl, **kwargs)

def save_as_tiff():
    pass

def get_zarr_files(directory):
    if not isinstance(directory, (str, os.PathLike)):
        logger.error("iter_zarr_dir requires a single string/path object")
    directory = Path(directory)

    # get directory contents
    contents = [x for x in directory.glob("*") if x.is_dir()]
    return [x for x in directory.glob("*") if x.suffix == ".zarr"]
