# Heavily adapted from suite2p
import argparse
from pathlib import Path
import numpy as np
from functools import partial

import lbm_caiman_python as lcp

current_file = Path(__file__).parent
with open(f"{current_file}/VERSION", "r") as VERSION:
    version = VERSION.read().strip()

print = partial(print, flush=True)

DEFAULT_BATCH_PATH = Path().home() / 'caiman_data' / 'batch'
DEFAULT_DATA_PATH = Path().home() / 'caiman_data' / 'data'
if not DEFAULT_BATCH_PATH.is_dir():
    print(f'Creating default batch path in {DEFAULT_BATCH_PATH}.')
    DEFAULT_BATCH_PATH.mkdir(exist_ok=True, parents=True)
if not DEFAULT_DATA_PATH.is_dir():
    print(f'Creating default data path in {DEFAULT_DATA_PATH}.')
    DEFAULT_DATA_PATH.mkdir(exist_ok=True, parents=True)


def add_args(parser: argparse.ArgumentParser):
    """
    Adds ops arguments to parser.
    """
    parser.add_argument("--run", type=str, help="algorithm to run, options mcorr, cnmf or cnmfe")
    parser.add_argument("--rm", type=int, help="0 based int of the row to delete")
    parser.add_argument("--version", action="store_true", help="current pipeline version")
    parser.add_argument("--ops", default=[], type=str, help="options")
    parser.add_argument("--db", default=[], type=str, help="database options")

    # Load default operations
    ops0 = lcp.default_ops()
    main_params = ops0.pop('main', {})
    ops0.update(main_params)
    # existing_args = {action.dest for action in parser._actions}

    # Add arguments for each key in the flattened dictionary
    for k, default_val in ops0.items():
        v = dict(default=default_val, help=f"{k} : {default_val}")
        if isinstance(v["default"], (np.ndarray, list)) and v["default"]:
            v["nargs"] = "+"
            v["type"] = type(v["default"][0])
        if k in ["batch_path", "batch-path"]:
            v['default'] = None  # required
            v['type'] = str
            v["dest"] = "batch_path"
        if k in ["data_path", "data-path"]:
            v['default'] = None  # required
            v['type'] = str
            v["dest"] = "data_path"
        if isinstance(v["default"], (np.ndarray, list)) and v["default"]:
            if (type(v["default"]) in [np.ndarray, list]) and len(v["default"]):
                v["nargs"] = "+"
                v["type"] = type(v["default"][0])
        parser.add_argument(f"--{k}", **v)
    return parser


def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns ops with parameters filled in.
    """
    args = parser.parse_args()
    dargs = vars(args)
    ops0 = lcp.default_ops()

    main_params = ops0.pop('main', {})
    ops0.update(main_params)

    ops = np.load(args.ops, allow_pickle=True).item() if args.ops else {}
    set_param_msg = "->> Setting {0} to {1}"

    for k in ops0:
        default_key = ops0[k]
        args_key = dargs[k]
        if isinstance(default_key, bool):
            args_key = bool(int(args_key))  # bool("0") is true, must convert to int
            if default_key != args_key:
                ops[k] = args_key
                print(set_param_msg.format(k, ops[k]))
        elif not (default_key == type(default_key)(args_key)):
            ops[k] = type(default_key)(args_key)
            print(set_param_msg.format(k, ops[k]))
    return args, ops


def get_matching_main_params(args):
    """
    Match arguments supplied through the cli with parameters found in the defaults.
    """
    matching_params = {k: getattr(args, k) for k in lcp.default_ops()['main'].keys() if hasattr(args, k)}
    return matching_params


def main():
    args, ops = parse_args(
        add_args(argparse.ArgumentParser(description="LBM-Caiman pipeline parameters")))
    if args.version:
        print("lbm_caiman_python v{}".format(version))
    if args.batch_path == "":
        print('No batch path provided. Provide a path to save results in a dataframe.')
    elif args.run:
        print('Batch path provided, retrieving batch:')
        from lbm_caiman_python.io.batch import load_batch
        print(args.batch_path)
        df = load_batch(args.batch_path)
        print(df)
        if args.rm:
            try:
                print(f'deleting row {args.rm}')
                lcp.batch.delete_batch_rows(df, [args.rm], safe=False)
                print(f'Row {args.rm} deleted.')
            except:
                raise NotImplementedError
        print('rm provided')
    elif args.run:
        # only import mesmerize + its caiman dependencies if we're running an algorithm
        import mesmerize_core as mc
        print(args.batch_path)
        df = mc.load_batch(args.batch_path)
        run_path = Path(args.run_path).resolve()
        mc.set_parent_raw_data_path(run_path.parent)
        # if run_path.is_dir():
        #     mc.set_parent_raw_data_path(run_path)
        # elif run_path.is_file():
        #     mc.set_parent_raw_data_path(run_path.parent)
        # clear df
        df.caiman.add_item(
            algo='mcorr',
            input_movie_path=run_path,
            params={'main': get_matching_main_params(args)},
            item_name=f'item_name',
        )
        algo = 'mcorr'
        print(f'Running {algo} -----------')
        df.iloc[-1].caiman.run()
        df = df.caiman.reload_from_disk()
        algo = 'cnmf'
        df.caiman.add_item(
            algo='cnmf',
            input_movie_path=df.iloc[-1],
            params={'main': get_matching_main_params(args)},
            item_name=f'item_name',
        )

        print(f'Running {algo} -----------')
        df.iloc[-1].caiman.run()
        df = df.caiman.reload_from_disk()
        print('Processing complete -----------')
    else:
        print('else')
        raise NotImplementedError


if __name__ == "__main__":
    main()
