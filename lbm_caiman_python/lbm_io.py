import os
import time
import logging
from pathlib import Path
import dask.array as da

import zarr
from zarr.errors import ContainsArrayError

import dask.array as da
import tifffile

from scanreader import scans

logging.basicConfig()
logger = logging.getLogger(__name__)

LBM_DEBUG_FLAG = os.environ.get('LBM_DEBUG', 1)

if LBM_DEBUG_FLAG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

## General
def iter_planes(scan, frames, planes, xslice=slice(None), yslice=slice(None)):
    for plane in planes:
        yield da.squeeze(scan[frames, plane, yslice, xslice])


def lbm_load_batch(batch_path, overwrite=False):
    batch_path = Path(batch_path)
    try:
        mc.set_parent_raw_data_path(batch_path.parent)
    except:
        import mesmerize_core as mc

    batch_path = raw_data_path / 'batch.pickle'
    mc.set_parent_raw_data_path(str(raw_data_path))

    # you could also load the registration batch and
    # save this patch in a new dataframe (saved to disk automatically)
    try:
        df = mc.load_batch(batch_path)
    except (IsADirectoryError, FileNotFoundError):
        df = mc.create_batch(batch_path)

    df=df.caiman.reload_from_disk()

def get_files(
        pathnames: os.PathLike | str | list[os.PathLike | str],
        ext: str = 'tif',
        exclude_pattern: str = '_plane_',
        debug: bool = False,
) -> list[os.PathLike | str] | os.PathLike:
    """
    Expands a list of pathname patterns to form a sorted list of absolute filenames.

    Parameters
    ----------
    pathnames: os.PathLike
        Pathname(s) or pathname pattern(s) to read.
    ext: str
        Extention, string giving the filetype extention.
    exclude_pattern: str | list
        A string or list of strings that match to files marked as excluded from processing.
    debug: bool
        Flag to print found, excluded and all files.

    Returns
    -------
    List[PathLike[AnyStr]]
        List of absolute filenames.
    """
    if '.' in ext or 'tiff' in ext:
        ext = 'tif'
    if isinstance(pathnames, (list, tuple)):
        out_files = []
        excl_files = []
        for fpath in pathnames:
            if exclude_pattern not in str(fpath):
                if Path(fpath).is_file():
                    out_files.extend([fpath])
                elif Path(fpath).is_dir():
                    fnames = [x for x in Path(fpath).expanduser().glob(f"*{ext}*")]
                    out_files.extend(fnames)
            else:
                excl_files.extend(fpath)
        return sorted(out_files)
    if isinstance(pathnames, (os.PathLike, str)):
        pathnames = Path(pathnames).expanduser()
        if pathnames.is_dir():
            files_with_ext = [x for x in pathnames.glob(f"*{ext}*")]
            if debug:
                excluded_files = [x for x in pathnames.glob(f"*{ext}*") if exclude_pattern in str(x)]
                all_files = [x for x in pathnames.glob("*")]
                logger.debug(excluded_files, all_files)
            return sorted(files_with_ext)
        elif pathnames.is_file():
            if exclude_pattern not in str(pathnames):
                return pathnames
            else:
                raise FileNotFoundError(f"No {ext} files found in directory: {pathnames}")
    else:
        raise ValueError(
            f"Input path should be an iterable list/tuple or PathLike object (string, pathlib.Path), not {pathnames}")


def get_single_file(filepath, ext='tif'):
    return [x for x in Path(filepath).glob(f"*{ext}*")][0]

## Zarr
def get_zarr_files(directory):
    if not isinstance(directory, (str, os.PathLike)):
        logger.error("iter_zarr_dir requires a single string/path object")
    return [x for x in Path(directory).glob("*") if x.suffix == ".zarr"]


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

## Tiff
def save_as_tiff(scan: scans.ScanLBM,
                 savedir: os.PathLike,
                 frames=slice(None),
                 planes=slice(None),
                 metadata=None,
                 prepend_str='extracted',
                 overwrite=False
                 ):

    savedir = Path(savedir)
    if isinstance(frames, int):
        frames = [frames]
    if isinstance(planes, int):
        planes = [planes]
    elif isinstance(planes, slice):
        planes = range(planes.start or 0, planes.stop or scan.num_planes-1, planes.step or 1)
    if not metadata:
        metadata = scan.metadata

    iterator = iter_planes(scan, frames, planes)
    outer = time.time()
    for idx, array in enumerate(iterator):
        start = time.time()
        filename = savedir / f'{prepend_str}_plane_{idx}.tif'
        logging.info(f'Saving {filename}')
        tifffile.imwrite(filename, array, bigtiff=True, metadata=metadata,)
        logging.info(f'{filename} saved ...')
        logging.info(f"Plane saved in {time.time() - start} seconds...")
    logging.info(f"All z-planes saved in {time.time() - outer} seconds...")
