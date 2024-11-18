# scanreader

Python TIFF Stack Reader for ScanImage 5 scans (including multiROI).

We treat a scan as a collection of recording fields:
rectangular planes at a given x, y, z position in the scan
recorded in a number of channels during a preset amount of time.
All fields have the same number of channels (or z-planes) and number of frames/timesteps.

## Installation

Install the latest version with `pip`

```shell
pip install git+https://github.com/MillerBrainObservatory/scanreader.git
```

## Python Usage

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

## Command Line Usage

`scanreader` is a command-line interface (CLI) tool for processing ScanImage TIFF files. It allows users to read, process, and save imaging data with options for selecting specific frames, planes, regions of interest (ROIs), and more.

### Basic Usage

```bash
python scanreader.py [OPTIONS] PATH
```

- `PATH`: Path to the file or directory containing the ScanImage TIFF files to process.

### Optional Arguments

- `--frames FRAME_SLICE`: Frames to read. Use slice notation like NumPy arrays (e.g., `1:50` reads frames 1 to 49, `10:100:2` reads every second frame from 10 to 98). Default is `:` (all frames).

- `--zplanes PLANE_SLICE`: Z-planes to read. Use slice notation (e.g., `1:50`, `5:15:2`). Default is `:` (all planes).

- `--trim_x LEFT RIGHT`: Number of x-pixels to trim from each ROI. Provide two integers for left and right edges (e.g., `--trim_x 4 4`). Default is `0 0` (no trimming).

- `--trim_y TOP BOTTOM`: Number of y-pixels to trim from each ROI. Provide two integers for top and bottom edges (e.g., `--trim_y 4 4`). Default is `0 0` (no trimming).

- `--metadata`: Print a dictionary of ScanImage metadata for the files at the given path.

- `--roi`: Save each ROI in its own folder, organized as `zarr/roi_1/plane_1/`. Without this argument, data is saved as `zarr/plane_1/roi_1`.

- `--save [SAVE_PATH]`: Path to save processed data. If not provided, metadata will be printed instead of saving data.

- `--overwrite`: Overwrite existing files when saving data.

- `--tiff`: Save data in TIFF format. Default is `True`.

- `--zarr`: Save data in Zarr format. Default is `False`.

- `--assemble`: Assemble each ROI into a single image.

### Examples

#### Print Metadata

To print metadata for the TIFF files in a directory:

```bash
python scanreader.py /path/to/data --metadata
```

#### Save All Planes and Frames as TIFF

To save all planes and frames to a specified directory in TIFF format:

```bash
python scanreader.py /path/to/data --save /path/to/output --tiff
```

#### Save Specific Frames and Planes as Zarr

To save frames 10 to 50 and planes 1 to 5 in Zarr format:

```bash
python scanreader.py /path/to/data --frames 10:51 --zplanes 1:6 --save /path/to/output --zarr
```

#### Save with Trimming and Overwrite Existing Files

To trim 4 pixels from each edge, overwrite existing files, and save:

```bash
python scanreader.py /path/to/data --trim_x 4 4 --trim_y 4 4 --save /path/to/output --overwrite
```

#### Save Each ROI Separately

To save each ROI in its own folder:

```bash
python scanreader.py /path/to/data --save /path/to/output --roi
```

### Notes

- **Slice Notation**: When specifying frames or z-planes, use slice notation as you would in NumPy arrays. For example, `--frames 0:100:2` selects every second frame from 0 to 99.

- **Default Behavior**: If `--save` is not provided, the program will print metadata by default.

- **File Formats**: By default, data is saved in TIFF format unless `--zarr` is specified.

- **Trimming**: The `--trim_x` and `--trim_y` options allow you to remove unwanted pixels from the edges of each ROI.

### Help

For more information on the available options, run:

```bash
python scanreader.py --help
```

### Details on data loading (for future developers)

Matlab stores data column by column ("Fortran order"), while NumPy by default stores them row by row ("C order").
This **doesn't affect indexing**, but **may affect performance**.
For example, in Matlab efficient loop will be over columns (e.g. for n = 1:10 a(:, n) end),
while in NumPy it's preferable to iterate over rows (e.g. for n in range(10): a[n, :] -- note n in the first position, not the last). 

As of this version, `scanreader` relies on [`tifffile`](https://pypi.org/project/tifffile/) to read the underlying tiff files.

Reading a scan happens in three stages:
1. `scan = scanreader.read_scan(filename)` will create a list of `tifffile.TiffFile`s, one per each tiff file in the scan. This entails opening a file handle and reading the tags of the first page of each; tags for the rest of pages are ignored (they have the same info).
2. `scan.num_frames`, `scan.shape` or another operation that requires the number of frames in the scan---which includes the first stage of any data loading operation---will need the number of pages in each tiff file. `tifffile` was designed for files with pages of varying shapes so it iterates over each page looking for its offset (number of bytes from the start of the file until the very first byte of the page), which it saves to use for reading. After this operation, it knows the number of pages per file.
3. Once the file has been opened and the offset to each page has been calculated we can load the actual data. We load each page sequentially and take care of reformatting them to match the desired output.

This reader and documentation are based off of  is based on a previous [version](https://github.com/atlab/scanreader) developed by [atlab](https://github.com/atlab/).

Some of the older scans have been removed for general cleanliness. These can be reimplemented by cherry-picking the commit. See documentation on `git reflog` to find the commits you want and `git cherry-pick` to apply changes that were introduced by those commits.
