import os
from pathlib import Path
import dask.array as da
import tifffile
from logging import getLogger

from scanreader import scans

logger = getLogger(__name__)


# def save_as_tiff(scan: scans.ScanLBM,
#                  savedir: os.PathLike, frames=slice(None), planes=slice(None), metadata=None,
#                  prepend_str='extracted'):
#     savedir = Path(savedir)
#     if not metadata:
#         metadata = {}
#
#     if channels is None:
#     channels = list(range(self.num_channels))[self.channel_slice]
#     channels = [self.channel_slice]
#     else:
#         raise ValueError(
#             f"ScanLBM.channel_size should be an integer or slice object, not {type(self.channel_slice)}.")
#     for idx, num in enumerate(channels):
#         filename = savedir / f'{prepend_str}_plane_{num}.tif'
#         data = self[:, channels, :, :]
#         tifffile.imwrite(filename, data, bigtiff=True, metadata=combined_metadata)
#

def tiffs2zarr(filenames, zarrurl, chunksize, **kwargs):
    """Write images from sequence of TIFF files as zarr."""
    with tifffile.TiffFile(filenames) as tifs:
        with tifs.aszarr() as store:
            arr = da.from_zarr(store)
            chunks = (chunksize,) + arr.shape[1:]
            da.rechunk(chunks).to_zarr(zarrurl, **kwargs)


def save_as_tiff():
    pass
