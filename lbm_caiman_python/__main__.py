# heavily adapted from suite2p
# https://github.com/MouseLand/suite2p/blob/main/suite2p/__main__.py
import argparse
import numpy as np
from lbm_caiman_python import default_ops


def add_args(parser: argparse.ArgumentParser):
    """
    Adds ops arguments to parser.
    """
    parser.add_argument("--single_plane", action="store_true", help="run single plane ops")
    parser.add_argument("--ops", default=[], type=str, help="options")
    parser.add_argument("--db", default=[], type=str, help="options")
    ops0 = default_ops()
    for k in ops0.keys():
        v = dict(default=ops0[k], help="{0} : {1}".format(k, ops0[k]))
        if k in ["fast_disk", "save_folder", "save_path0"]:
            v["default"] = None
            v["type"] = str
        if (type(v["default"]) in [np.ndarray, list]) and len(v["default"]):
            v["nargs"] = "+"
            v["type"] = type(v["default"][0])
        parser.add_argument("--" + k, **v)
    return parser
