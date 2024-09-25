# scanreader

Python TIFF Stack Reader for ScanImage 5 scans (including multiROI).

We treat a scan as a collection of recording fields:
rectangular planes at a given x, y, z position in the scan
recorded in a number of channels during a preset amount of time.
All fields have the same number of channels (or z-planes) and number of frames/timesteps.

## Installation

Install the latest version with `pip`

```shell
pip3 install git+https://github.com/MillerBrainObservatory/scanreader.git
```

## Usage

```python
import scanreader

scan = scanreader.read_scan('/data/my_scan_*.tif')

print(scan.version)
print(scan.num_frames)
print(scan.num_channels)
print(scan.num_fields)

x = scan[:]  # 5D array [ROI, X, Y, Z, T]
y = scan[:, :, :, 0:4, -1000:]  # last 1000 frames of first 4 planes
z = scan[1]  # 4-d array: the second ROI (over all z-plane and time)

scan = scanreader.read_scan('/data/my_scan_*.tif', dtype=np.float32, join_contiguous=True)
```

Scan object indexes can be:

- integers
- slice objects (:)
- lists/tuples/arrays of integers

No boolean indexing is yet supported.

## Details on data loading (for future developers)

As of this version, `scanreader` relies on [`tifffile`](https://pypi.org/project/tifffile/) to read the underlying tiff files.

Reading a scan happens in three stages:
1. `scan = scanreader.read_scan(filename)` will create a list of `tifffile.TiffFile`s, one per each tiff file in the scan. This entails opening a file handle and reading the tags of the first page of each; tags for the rest of pages are ignored (they have the same info).
2. `scan.num_frames`, `scan.shape` or another operation that requires the number of frames in the scan---which includes the first stage of any data loading operation---will need the number of pages in each tiff file. `tifffile` was designed for files with pages of varying shapes so it iterates over each page looking for its offset (number of bytes from the start of the file until the very first byte of the page), which it saves to use for reading. After this operation, it knows the number of pages per file.
3. Once the file has been opened and the offset to each page has been calculated we can load the actual data. We load each page sequentially and take care of reformatting them to match the desired output.

This reader and documentation are based off of  is based on a previous [version](https://github.com/atlab/scanreader) developed by [atlab](https://github.com/atlab/).

Some of the older scans have been removed for general cleanliness. These can be reimplemented by cherry-picking the commit. See documentation on `git reflog` to find the commits you want and `git cherry-pick` to apply changes that were introduced by those commits.
