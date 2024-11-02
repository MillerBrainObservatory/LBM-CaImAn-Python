import os
from pathlib import Path
from scanreader import scans
from lbm_io import get_files, save_as_tiff, save_as_zarr, lbm_load_batch


def get_reader(datapath: os.PathLike):
    filepath = Path(datapath)
    if filepath.is_file():
        filepath = datapath
    else:
        filepath = [
            files for files in datapath.glob("*.tif")
        ]  # this accumulates a list of every filepath which contains a .tif file
    return filepath


def read_scan(
        pathnames: os.PathLike | str | list,
        trim_roi_x: list | tuple = (0, 0),
        trim_roi_y: list | tuple = (0, 0),
) -> scans.ScanLBM:
    """
    Reads a ScanImage scan.

    Parameters
    ----------
    pathnames: os.PathLike
        Pathname(s) or pathname pattern(s) to read.
    trim_roi_x: tuple, list, optional
        Indexable (trim_roi_x[0], trim_roi_x[1]) item with 2 integers denoting the amount of pixels to trim on the left [0] and right [1] side of **each roi**.
    trim_roi_y: tuple, list, optional
        Indexable (trim_roi_y[0], trim_roi_y[1]) item with 2 integers denoting the amount of pixels to trim on the top [0] and bottom [1] side of **each roi**.

    Returns
    -------
    ScanLBM
        A Scan object (subclass of ScanMultiROI) with metadata and different offset correction methods.
        See Readme for details.

    """
    # Expand wildcards
    filenames = lbm_io.get_files(pathnames)
    if isinstance(filenames, (list, tuple)):
        if len(filenames) == 0:
            raise FileNotFoundError(f"Pathname(s) {filenames} do not match any files in disk.")

    # Get metadata from first file
    return scans.ScanLBM(
        filenames,
        trim_roi_x=trim_roi_x,
        trim_roi_y=trim_roi_y
    )
