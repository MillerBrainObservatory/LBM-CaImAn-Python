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

