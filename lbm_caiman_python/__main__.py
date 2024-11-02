# heavily adapted from suite2p
# https://github.com/MouseLand/suite2p/blob/main/suite2p/__main__.py
import argparse
import numpy as np
from lbm_caiman_python import default_ops

with open("VERSION", "r") as VERSION:
    version = VERSION.read().strip()


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


def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns ops with parameters filled in.
    From: https://github.com/MouseLand/suite2p/blob/main/suite2p/__main__.py
    """
    args = parser.parse_args()
    dargs = vars(args)
    ops0 = default_ops()
    ops = np.load(args.ops, allow_pickle=True).item() if args.ops else {}
    set_param_msg = "->> Setting {0} to {1}"
    # options defined in the cli take precedence over the ones in the ops file
    for k in ops0:
        default_key = ops0[k]
        args_key = dargs[k]
        if k in ["fast_disk", "save_folder", "save_path0"]:
            if args_key:
                ops[k] = args_key
                print(set_param_msg.format(k, ops[k]))
        elif type(default_key) in [np.ndarray, list]:
            n = np.array(args_key)
            if np.any(n != np.array(default_key)):
                ops[k] = n.astype(type(default_key))
                print(set_param_msg.format(k, ops[k]))
        elif isinstance(default_key, bool):
            args_key = bool(int(args_key))  # bool("0") is true, must convert to int
            if default_key != args_key:
                ops[k] = args_key
                print(set_param_msg.format(k, ops[k]))
        # checks default param to args param by converting args to same type
        elif not (default_key == type(default_key)(args_key)):
            ops[k] = type(default_key)(args_key)
            print(set_param_msg.format(k, ops[k]))
    return args, ops


def main():
    args, ops = parse_args(
        add_args(argparse.ArgumentParser(description="LBM-Caiman pipeline parameters")))
    if args.version:
        print("lbm_caiman_python v{}".format(version))
    elif args.single_plane and args.ops:
        from lbm_caiman_python.run_lcp import run_plane
        # run single plane (does registration)
        run_plane(ops, ops_path=args.ops)
    elif len(args.db) > 0:
        db = np.load(args.db, allow_pickle=True).item()
        from lbm_caiman_python import run_lcp
        run_lcp(ops)
    else:
        print('else')


if __name__ == "__main__":
    main()
