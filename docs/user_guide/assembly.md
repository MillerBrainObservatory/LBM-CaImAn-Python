(assembly)=
# Image Assembly

## Overview

Before running motion-correction or segmentation, we need to de-interleave raw `.tiff` files. This is done internally with the [scanreader](https://github.com/atlab/scanreader).

## scanreader

The first thing you need to do is initialize a scan. This is done with {ref}`read_scan`.

```{tip}
Using `pathlib.Path().home()` can give quick filepaths to your home directory. 
From there, you can `.glob('*')` to grab all files in a given directory.
```

```{code-block} Python
import lbm_caiman_python as lcp

scan = lcp.read_scan('path/to/data/*.tiff')

```

If you give a string with a wildcard (like an asterisk), this wildcard will expand to match all files around the asterisk. 

In the above example. every file inside `/path/to/data/` ending in `.tiff` will be included in the scan.

```{important}

Make sure your `data_path` contains only `.tiff` files for this imaging session. If there are other `.tiff` files, such as from another session or a processed file for this session, those files will be included in the scan and lead to errors.

```

The default: `lcp.read_scan(data_path, join_contiguous=False)` will give a 5D array, `[rois, y, x, channels, Time]` that can be index just like a numpy array.


```{code-block} Python
scan = lcp.read_scan('path/to/data/*.tiff')
scan[:].shape

>>> (4, 600, 144, 30, 1730)

```

Depending on your scanimage configuration, contiguous ROIs can be joined together via the `join_contiguous` parameter to {ref}`read_scan()`

```{code-block} Python
scan = lcp.read_scan('path/to/data/*.tiff', join_contiguous=True)
scan[:].shape

>>> (600, 576, 30, 1730)

```

If your configuration has combinations of contiguous ROI's, such as left-hemisphere / right-hemisphere pairs of ROI's, they will be only merge the contiguous ROI's based on the metadata x-center-coordinate and y-center-coordinate.

```{code-block} Python
scan = lcp.read_scan('path/to/hemispheric/*.tiff', join_contiguous=True)
scan[:].shape

>>> (2, 212, 212, 30, 1730)

```


## Command Line Usage

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

## Assembly Benchmarks

CPU: 13th Gen Intel(R) Core(TM) i9-13900KS   3.20 GHz
RAM: 128 GB usable
OS: Windows 10 Pro, 22H2

| **Command**                      | **Time (seconds)** | **Details**                            |
|----------------------------------|--------------------|----------------------------------------|
| **--save**                       | **87.02**          | Save each ROI to disk without overwriting data  |
| **--save --roi**                 | **92.26**          | Save each ROI to disk without overwriting data  |
| **--save --assemble**            | **167.44**         | Save .tiffs with ROI's assembled       |