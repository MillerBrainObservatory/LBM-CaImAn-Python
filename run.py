import os
from pathlib import Path

import zarr

import scanreader

CHAN_ORDER = [1, 5, 6, 7, 8, 9, 2, 10, 11, 12, 13, 14, 15, 16, 17, 3, 18, 19, 20, 21, 22, 23, 4, 24, 25, 26, 27, 28, 29, 30]
CHAN_ORDER = [x - 1 for x in CHAN_ORDER]  # this is specific to our dataset


def load_data(data_path):
    """ Return a ScanReader object """
    data_path = Path(data_path)  # string pointing to directory containing your data
    files = [x for x in data_path.glob('*.tif')]  # this accumulates a list of every filepath which contains a .tif file
    return scanreader.read_scan(str(files[0]), join_contiguous=True, x_cut=(6, 6), y_cut=(17, 0), lbm=True)


def load_plane(reader: scanreader.core.scans.LBMScanMultiROI, save_path: os.PathLike):
    save_path = Path(save_path).with_suffix('.zarr')
    data = reader[:, :, :, 1, 1]  # single frame, all planes
    store = zarr.DirectoryStore(str(save_path))
    zarr.save_array(store=store, arr=data, path=save_path)
    return zarr.load(store, path=save_path)

def load_zarr(data_path):
    data_path = Path(data_path)
    return zarr.open(str(data_path))


data_path = Path('/data2/fpo/data/extracted/high_res/').glob("*.zarr")
files = [x for x in data_path]
zarr = load_zarr(str(files[0]))
# reader = load_data(data_path)
# data = load_plane(reader, '/data2/fpo/data/extracted/LBM_Test.zarr')
# data = data.squeeze()
#
# # plot the first frame
# import matplotlib.pyplot as plt
# plt.imshow(data[...])
# plt.savefig(data_path / 'first_frame.png')
# myarr = dask.array.from_zarr(data)
x = 2

# create empty zarr store to save on disk and write to
# z1 = zarr.open('/data2/fpo/data/extracted/{dataset_name}/{dataset_name}.zarr', mode='w', shape=(data.shape),
#                chunks=(data.shape[0], data.shape[1], data.shape[2], 1), dtype='int16')
# z1[:] = data
