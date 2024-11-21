# User Guide

How to process your data!

## Image Assembly

Before running motion-correction or segmentation, we need to de-interleave raw `.tiff` files. This is done internally with the [scanreader](https://github.com/atlab/scanreader).

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

## Batch Setup with {ref}`mesmerize-core`

`````{tab-set}
````{tab-item} CLI
``` bash
lcp /batch/path --create
```
````

````{tab-item} Python
```python
df = mc.create_batch('/batch/path')
```
````
`````




### Command Line Usage Overview:

| Command                                                          | Description                                    |
|------------------------------------------------------------------|------------------------------------------------|
| `lcp  /path/to/batch`                             | Print DataFrame contents.                       |
| `lcp  /path/to/batch --create`                  | Print batch, create if doesnt exist.                                           |
| `lcp  /path/to/batch --rm [int(s)]`                     | Remove index(s)  from DataFrame. Can provide multiple indices provided as list.            |
| `lcp  /path/to/batch --rm [int(s)] --remove_data`       | Remove index `[int]` and its child items.       |
| `lcp  /path/to/batch --clean`                        | Remove any unsuccessful runs.                   |
| `lcp  /path/to/batch --add [ops/path.npy]`           | Add a batch item with specified parameters.     |
| `lcp  /path/to/batch --run [algo(s)]`                   | Run specified algorithm.                        |
| `lcp  /path/to/batch --run [algo(s)] --data_path [str]` | Run specified algorithm on specified data path. |
| `lcp  /path/to/batch --run [algo(s)] --data_path [int]` | Run specified algorithm on DataFrame index.     |
| `lcp  /path/to/batch --view_params [int]`                  | View parameters for DataFrame index.                |

*int = integer, 0 1 2 3 etc.
*algo = mcorr, cnmf, or cnmfe
*str=a path of some sort. will be fed into pathlib.Path(str).resolve() to expand `~` chars.

Chain mcorr and cnmf together:

```bash
lcp /home/mbo/lbm_data/batch/batch.pickle --run mcorr cnmf --data_path /home/mbo/lbm_data/demo_data.tif
```

Full example:
```bash
$ lcp /home/mbo/lbm_data/batch/batch.pickle --create --run mcorr cnmf --strides 32 32 --overlaps 8 8 --K 100 --data_path /home/mbo/lbm_data/demo_data.tif 
```

## Mesmerize-Core

{external:func}`mesmerize_core.load_batch`
