# Code Snippets

Helpful snippets for all things LBM python.

## General

```{code-block} python
:caption: View Computer Info

```


### Input data paths

When in doubt, use a '/' foreward slash. This will work for windows 'C:/Users/' without needing a double backslash.
Using [`pathlib.Path()](https://docs.python.org/3/library/pathlib.html#pathlib.Path).

This will automatically return you a [Windows Path](https://docs.python.org/3/library/pathlib.html#pathlib.PosixPath) or a [PosixPath](https://docs.python.org/3/library/pathlib.html#pathlib.WindowsPath).

Note on windows to not confuse your wsl path "//$wsl/home/<>" with your windows path "C:/Users/".

Single quotes vs double quotes doesn't matter.

```{code-block} python
:caption: Data path inputs

# this works on any operating system, any filepath structure
data_path = Path().home() / 'Documents' / 'data' / 'high_res'

raw_files = [x for x in data_path.glob(f'*.tif*')]

```

# Troubleshooting

caiman heavily relies on opencv, but doens't install it for you. It depends on a few dependencies that are often missing from standard machines. 

```{code-block} python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```

### Get Files

```{code-block} python

def get_files(pathnames: os.PathLike | List[os.PathLike | str]) -> List[PathLike[AnyStr]]:
    """
    Expands a list of pathname patterns to form a sorted list of absolute filenames.

    Parameters
    ----------
    pathnames: os.PathLike
        Pathname(s) or pathname pattern(s) to read.

    Returns
    -------
    List[PathLike[AnyStr]]
        List of absolute filenames.
    """
    pathnames = Path(pathnames).expanduser()  # expand ~ to /home/user
    if not pathnames.exists():
        raise FileNotFoundError(f'Path {pathnames} does not exist as a file or directory.')
    if pathnames.is_file():
        return [pathnames]
    if pathnames.is_dir():
        pathnames = [fpath for fpath in pathnames.glob("*.tif*")]  # matches .tif and .tiff
    return sorted(pathnames, key=path.basename)

```

## Relative Import

`ImportError: attempted relative import with no known parent package`

This almost always occurs when you try to run a specific script directly without running the python package i.e. `python -m path/to/project/` vs `python path/to/project/file.py`

```{admonition} __main__.py
:class: dropdown

The purpose of this file is to tell our python package how to run the code.

You can execute __main__.py as if it were a python module, fixing the above import errors.

Like so:

    `python /home/mbo/repos/scanreader/scanreader/__main__.py`

Equivlent to:

    `python -m /home/mbo/repos/scanreader/scanreader/`

```

## Photometric: MinIsBlack

This is the `PhotometricInterpretation` [TIFF tag](https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml), which defines how to interpret the values of the Tiff strips.
Example: for an RGB Tiff: band 1 = red, band 2 = green, band 3 = blue.
CMYK stands for the Cyan, Magenta, Yellow, blacK color space, etc.

MinIsBlack defines a gradient from 0 to 1 representing shades of gray.

