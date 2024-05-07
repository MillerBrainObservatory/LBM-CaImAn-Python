"""
.. currentmodule:: core

core/__init__.py: This file is the entry point for the core package.
It imports the core module of the LBM pipeline. The __all__ variable is set to ["core"],
which means that when the core package is imported, only the core module will be accessible.

"""
from .io import *
from .util import *

if __name__ == "__main__":
    from pathlib import Path
    filepath = Path("/data2/fpo/data")