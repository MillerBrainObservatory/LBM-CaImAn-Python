"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import gc
import glob
import json
import math
import os
import time
from typing import Union, Tuple, Optional

import numpy as np
from tifffile import imread, TiffFile, TiffWriter

try:
    from ScanImageTiffReader import ScanImageTiffReader
    HAS_SCANIMAGE = True
except ImportError:
    ScanImageTiffReader = None
    HAS_SCANIMAGE = False


def generate_tiff_filename(functional_chan: int, align_by_chan: int, save_path: str,
                           k: int, ichan: bool) -> str:
    """
    Calculates a tiff filename from different parameters.

    Parameters
    ----------
    functional_chan: int
        The channel number with functional information
    align_by_chan: int
        Which channel to use for alignment
    save_path: str
        The directory to save to
    k: int
        The file number
    wchan: int
        The channel number.

    Returns
    -------
    filename: str
    """
    if ichan:
        if functional_chan == align_by_chan:
            tifroot = os.path.join(save_path, "reg_tif")
            wchan = 0
        else:
            tifroot = os.path.join(save_path, "reg_tif_chan2")
            wchan = 1
    else:
        if functional_chan == align_by_chan:
            tifroot = os.path.join(save_path, "reg_tif_chan2")
            wchan = 1
        else:
            tifroot = os.path.join(save_path, "reg_tif")
            wchan = 0
    if not os.path.isdir(tifroot):
        os.makedirs(tifroot)
    fname = "file00%0.3d_chan%d.tif" % (k, wchan)
    fname = os.path.join(tifroot, fname)
    return fname


def save_tiff(mov: np.ndarray, fname: str) -> None:
    """
    Save image stack array to tiff file.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to save
    fname: str
        The tiff filename to save to

    """
    with TiffWriter(fname) as tif:
        for frame in np.floor(mov).astype(np.int16):
            tif.write(frame, contiguous=True)


def open_tiff(file: str,
              sktiff: bool) -> Tuple[Union[TiffFile, ScanImageTiffReader], int]:
    """ Returns image and its length from tiff file with either ScanImageTiffReader or tifffile, based on "sktiff" """
    if sktiff:
        tif = TiffFile(file)
        Ltif = len(tif.pages)
    else:
        tif = ScanImageTiffReader(file)
        Ltif = 1 if len(tif.shape()) < 3 else tif.shape()[0]  # single page tiffs
    return tif, Ltif


def read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff):
    # tiff reading
    if ix >= Ltif:
        return None
    nfr = min(Ltif - ix, batch_size)
    if use_sktiff:
        im = imread(file, key=range(ix, ix + nfr))
    elif Ltif == 1:
        im = tif.data()
    else:
        im = tif.data(beg=ix, end=ix + nfr)
    # for single-page tiffs, add 1st dim
    if len(im.shape) < 3:
        im = np.expand_dims(im, axis=0)

    # check if uint16
    if im.dtype.type == np.uint16:
        im = (im // 2).astype(np.int16)
    elif im.dtype.type == np.int32:
        im = (im // 2).astype(np.int16)
    elif im.dtype.type != np.int16:
        im = im.astype(np.int16)

    if im.shape[0] > nfr:
        im = im[:nfr, :, :]

    return im