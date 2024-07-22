from pathlib import Path
from core import *

def get_reader(datapath):
    filepath = datapath
    if Path(datapath).isfile():
        filepath = datapath
    else:
        filepath = [x for x in datapath.glob('*.tif')]      # this accumulates a list of every filepath which contains a .tif file
    return filepath

__all__ = [
    'core',
    'get_reader'
]
