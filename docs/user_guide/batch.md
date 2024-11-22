(batch)=
# Batch

Before continuing, users should review the [mesmerize-core user guide](https://mesmerize-core.readthedocs.io/en/latest/user_guide.html).

Functions in this section are just wrappers for mesmerize-core, and all of it's functionality can be directly used in your resulting batch dataframes.

## Overview

The general workflow is as follows:

1. Create an empty batch
2. Add an algorithm to this batch
3. Run the algorithm
4. Preview results

`LBM-CaImAn-Python` provides a command-line interface and example notebooks for each of these steps.

## Create a Batch

```{tip}
See the [batch helpers notebook](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python/blob/master/demos/notebooks/batch_helpers.ipynb). This is handy to have open 
to manage, remove, and manipulate results during analysis.
```

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

See {external:func}`mesmerize_core.load_batch()`, {external:func}`mesmerize_core.create_batch()`,

(batch_add)=
## Add a batch item

Next, we add an item to the batch.

A batch item is a combination of:
- algorithm to run, `algo`
- input movie to run the algorithm on, `input_movie_path`
- parameters for the specified algorithm, `params`
- a name for you to keep track of things, usually the same as the movie filename, `item_name`

As of now, the only way to change parameters via command-line is to run an algorithm (`--run [algo]`) and include a flag `--param` followd by the parameter to be changed.

`````{tab-set}
````{tab-item} CLI
``` bash
lcp /batch/path --run mcorr --max_shifts 20 20 --pw_rigid True --strides 
```
````

````{tab-item} Python
```python
import mesmerize_core as mc

mcorr_params = {
    'main':  # this key is necessary for specifying that these are the "main" params for the algorithm
    {
        'max_shifts': (20, 20),
        'strides': [48, 48],
        'overlaps': [24, 24],
        'max_deviation_rigid': 3,
        'border_nan': 'copy',
        'pw_rigid': True,
        'gSig_filt': (2, 2)
    },
}

mc.set_raw_data_path(path/to/tiff)
df.caiman.add_item(
    algo='mcorr',
    input_movie_path=path/to/tiff,
    params=mcorr_params,
    item_name=movie_path.stem,  # filename of the movie, but can be anything
)

```
````
`````

(batch_run)=
## Run a batch item

After adding an item, running the item is as easy as calling `row.caiman.run()`:

``` python
df.iloc[-1].caiman.run()
```

Here, we are using pandas.iloc[index], where index is the row in the dataframe containing the algorithm you wish to run. 

Choosing -1 indicates the "last item" in the dataframe, which is often the item you just added in the previous step. 

(lcp-cli)=
## Command Line Usage Overview:

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

## Examples

Chain mcorr and cnmf together:

```bash
lcp /home/mbo/lbm_data/batch/batch.pickle --run mcorr cnmf --data_path /home/mbo/lbm_data/demo_data.tif
```

Full example:

```bash
$ lcp /home/mbo/lbm_data/batch/batch.pickle --create --run mcorr cnmf --strides 32 32 --overlaps 8 8 --K 100 --data_path /home/mbo/lbm_data/demo_data.tif 
```

## Mesmerize-Core

