import os
import logging
from pathlib import Path
import pickle
from typing import Any
from .movie import VideoReader
from .zarr_io import save_as_zarr

logging.basicConfig()
logger = logging.getLogger(__name__)

LBM_DEBUG_FLAG = os.environ.get('LBM_DEBUG', 1)

if LBM_DEBUG_FLAG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


def save_object(obj: object, filename: str) -> None:
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename: str) -> Any:
    with open(filename, "rb") as input_obj:
        obj = pickle.load(input_obj)
    return obj


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

    

def lbm_load_batch(batch_path, overwrite=False):
    batch_path = Path(batch_path)
    try:
        mc.set_parent_raw_data_path(batch_path.parent)
    except:
        import mesmerize_core as mc

    batch_path = raw_data_path / 'batch.pickle'
    mc.set_parent_raw_data_path(str(raw_data_path))

    # you could alos load the registration batch and 
    # save this patch in a new dataframe (saved to disk automatically)
    try:
        df = mc.load_batch(batch_path)
    except (IsADirectoryError, FileNotFoundError):
        df = mc.create_batch(batch_path)

    df=df.caiman.reload_from_disk()

__all__ = [
    "save_object",
    "load_object",
    "save_as_zarr",
    "get_files",
    "get_single_file",
    # 'save_zstack',
    # 'load_zstack',
    "VideoReader",
]
