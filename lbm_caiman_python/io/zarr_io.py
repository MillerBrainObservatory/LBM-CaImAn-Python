import os
import time
from pathlib import Path
import dask.array as da
import logging

import zarr
from scanreader import scans
from zarr.errors import ContainsArrayError

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
    return [x for x in Path(directory).glob("*") if x.suffix == ".zarr"]


def iter_planes(scan, frames, planes, xslice=slice(None), yslice=slice(None)):
    for plane in planes:
        yield da.squeeze(scan[frames, plane, yslice, xslice])


def save_as_zarr(scan: scans.ScanLBM,
                 savedir: os.PathLike,
                 frames: list | slice | int = slice(None),
                 planes: list | slice | int = slice(None),
                 # metadata=None,
                 # prepend_str='extracted',
                 overwrite=False
                 ):
    filestore = zarr.DirectoryStore(str(savedir))
    root = zarr.group(filestore, overwrite=overwrite)

    if isinstance(frames, int):
        frames = [frames]
    if isinstance(planes, int):
        planes = [planes]

    iterator = iter_planes(scan, frames, planes)
    logging.info(f"Selected planes: {planes}")
    outer = time.time()

    for idx, array in enumerate(iterator):
        start = time.time()
        try:
            da.to_zarr(
                arr=array,
                url=savedir,
                component=f"plane_{idx + 1}",
                overwrite=overwrite,
            )
        except ContainsArrayError:
            logging.info(f"Plane {idx + 1} already exists. Skipping...")
            continue
        # root['preprocessed'][f'plane_{idx+1}'].attrs['fps'] = self.metadata['fps']
        logging.info(f"Plane saved in {time.time() - start} seconds...")
    logging.info(f"All z-planes saved in {time.time() - outer} seconds...")