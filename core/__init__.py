"""
.. currentmodule:: core

This module contains the core functionality of the package. It provides the following submodules:

- :mod:`core.io`: Contains functions for reading and writing data.
- :mod:`core.util`: Contains utility functions for data processing.

This __init__.py file controls what modules, functions, or packages are imported to the package level.
For example, if you want to import the function :func:`core.io.read_data` to the package level, you would add the following line to this file:

.. code-block:: python

    from .io import read_data
"""
from .io import *
from .util import *

if __name__ == "__main__":
    from pathlib import Path
    filepath = Path("/data2/fpo/data")