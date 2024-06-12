import pickle 
from typing import Any
from .h5 import ( save_zstack, load_zstack )
from .movie import VideoReader
from .binary import ( 
    BinaryFile, BinaryFileCombined, binned_mean 
)
from .tiff import (
    save_tiff
)


def save_object(obj, filename:str) -> None:
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename:str) -> Any:
    with open(filename, 'rb') as input_obj:
        obj = pickle.load(input_obj)
    return obj

__all__ = [
    'save_object',
    'load_object',
    'save_zstack',
    'VideoReader',
    'BinaryFile',
    'BinaryFileCombined',
    'binned_mean',
    'save_tiff'
]