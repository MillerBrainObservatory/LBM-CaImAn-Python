# User Guide

## Image Assembly

Usage:

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
