# Code Snippets

Helpful snippets for all things LBM python.

----

## CaImAn Parameters

[caiman parameters](https://caiman.readthedocs.io/en/latest/Getting_Started.html#parameters)

## Troubleshooting

You may need some additional dependencies on WSL2:

```{code-block} python
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```

### Filepaths and data directories

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

### `ImportError: attempted relative import with no known parent package`

This almost always occurs when you try to run a specific script directly without running the python package i.e. `python -m path/to/project/` vs `python path/to/project/file.py`

```{admonition} __main__ python file
:class: dropdown

The purpose of this file is to tell our python package how to run the code.

You can execute __main__.py as if it were a python module, fixing the above import errors.

Like so:

    `python /home/mbo/repos/scanreader/scanreader/__main__.py`

Equivlent to:

    `python -m /home/mbo/repos/scanreader/scanreader/`

```

----

## System Information

[cloudmesh-cmd5](https://github.com/cloudmesh/cloudmesh-cmd5) is a helpful library to view system information.

Install via [pip](https://pypi.org/project/cloudmesh-sys/):


```{code-block} python

pip install cloudmesh-sys

```

```{table} System Information
:name: user_information
:align: center

| Attribute           | Value                                                                         |
| :------------------ | :---------------------------------------------------------------------------- |
| BUG_REPORT_URL      | "https://bugs.launchpad.net/ubuntu/"                                          |
| DISTRIB_CODENAME    | jammy                                                                         |
| DISTRIB_DESCRIPTION | "Ubuntu 22.04.4 LTS"                                                          |
| DISTRIB_ID          | Ubuntu                                                                        |
| DISTRIB_RELEASE     | 22.04                                                                         |
| HOME_URL            | "https://www.ubuntu.com/"                                                     |
| ID                  | ubuntu                                                                        |
| ID_LIKE             | debian                                                                        |
| NAME                | "Ubuntu"                                                                      |
| PRETTY_NAME         | "Ubuntu 22.04.4 LTS"                                                          |
| PRIVACY_POLICY_URL  | "https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"              |
| SUPPORT_URL         | "https://help.ubuntu.com/"                                                    |
| UBUNTU_CODENAME     | jammy                                                                         |
| VERSION             | "22.04.4 LTS (Jammy Jellyfish)"                                               |
| VERSION_CODENAME    | jammy                                                                         |
| VERSION_ID          | "22.04"                                                                       |
| cpu                 | 13th Gen Intel(R) Core(TM) i9-13900KS                                         |
| cpu_cores           | 16                                                                            |
| cpu_count           | 32                                                                            |
| cpu_threads         | 32                                                                            |
| date                | 2024-08-15 15:59:58.462764                                                    |
| frequency           | scpufreq(current=3187.198999999998, min=0.0, max=0.0)                         |
| mem.active          | 248.8 MiB                                                                     |
| mem.available       | 56.3 GiB                                                                      |
| mem.free            | 55.7 GiB                                                                      |
| mem.inactive        | 5.8 GiB                                                                       |
| mem.percent         | 10.2 %                                                                        |
| mem.total           | 62.7 GiB                                                                      |
| mem.used            | 5.7 GiB                                                                       |
| platform.version    | #1 SMP Thu Mar 7 03:22:57 UTC 2024                                            |
| python              | 3.11.9 | packaged by conda-forge | (main, Apr 19 2024, 18:36:13) [GCC 12.3.0] |
| python.pip          | 24.2                                                                          |
| python.version      | 3.11.9                                                                        |
| sys.platform        | linux                                                                         |
| uname.machine       | x86_64                                                                        |
| uname.node          | RBO-C2                                                                        |
| uname.processor     | x86_64                                                                        |
| uname.release       | 5.15.150.1-microsoft-standard-WSL2                                            |
| uname.system        | Linux                                                                         |
| uname.version       | #1 SMP Thu Mar 7 03:22:57 UTC 2024                                            |
| user                | mbo                                                                           |
+---------------------+-------------------------------------------------------------------------------+

```

## FAQ

### 1. Foreward slash or backwards slash

**When in doubt, use a `/` foreward slash.** 

This will work for windows `C:/Users/` without needing a double backslash using [`pathlib.Path()](https://docs.python.org/3/library/pathlib.html#pathlib.Path) (built into python).

This will automatically return you a [Windows Path](https://docs.python.org/3/library/pathlib.html#pathlib.PosixPath) or a [PosixPath](https://docs.python.org/3/library/pathlib.html#pathlib.WindowsPath).

````{admonition} Filepaths on [WSL](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux)
:class: dropdown

Be sure to not confuse your wsl path `//$wsl/home/MBO` with your windows home path `C:/Users/MBO`.

```{code-block} python
:caption: Data path inputs

# this works on any operating system, any filepath structure
data_path = Path().home() / 'Documents' / 'data' / 'high_res'

raw_files = [x for x in data_path.glob(f'*.tif*')]

```
````
