import pickle
from typing import Any
from .movie import VideoReader

# from .h5 import ( save_zstack, load_zstack )


def save_object(obj: object, filename: str) -> None:
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename: str) -> Any:
    with open(filename, "rb") as input_obj:
        obj = pickle.load(input_obj)
    return obj


__all__ = [
    "save_object",
    "load_object",
    # 'save_zstack',
    # 'load_zstack',
    "VideoReader",
]
