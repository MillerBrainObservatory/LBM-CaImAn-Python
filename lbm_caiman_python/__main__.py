# Heavily adapted from suite2p
import argparse
from pathlib import Path
import numpy as np
from functools import partial
import lbm_caiman_python as lcp
import mesmerize_core as mc

current_file = Path(__file__).parent
with open(f"{current_file}/VERSION", "r") as VERSION:
    version = VERSION.read().strip()

# logging.getLogger("tensorflow").setLevel(logging.ERROR)

print = partial(print, flush=True)

DEFAULT_BATCH_PATH = Path().home() / "caiman_data" / "batch"
DEFAULT_DATA_PATH = Path().home() / "caiman_data" / "data"
if not DEFAULT_BATCH_PATH.is_dir():
    print(f"Creating default batch path in {DEFAULT_BATCH_PATH}.")
    DEFAULT_BATCH_PATH.mkdir(exist_ok=True, parents=True)
if not DEFAULT_DATA_PATH.is_dir():
    print(f"Creating default data path in {DEFAULT_DATA_PATH}.")
    DEFAULT_DATA_PATH.mkdir(exist_ok=True, parents=True)


def parse_data_path(value):
    # try to convert to integer if possible, otherwise treat as a file path
    try:
        return int(value)
    except ValueError:
        return Path(value).resolve()


def add_args(parser: argparse.ArgumentParser):
    """
    Adds ops arguments to parser.
    """
    parser.add_argument("--run", type=str, nargs='+', help="algorithm to run, options mcorr, cnmf or cnmfe")
    parser.add_argument("--rm", type=int, nargs='+', help="0 based int of the row to delete")
    parser.add_argument(
        "--remove-data",
        "--remove_data",
        dest="remove_data",
        help="If removing a batch item, also delete child results.",
        action="store_true",  # set to True if present
    )

    parser.add_argument(
        "--f",
        "--force",
        "-f",
        dest="force",
        help="Force deletion of the batch item without prompt.",
        action="store_true",  # set to True if present
    )

    parser.add_argument("--version", action="store_true", help="current pipeline version")
    parser.add_argument("--ops", default=[], type=str, help="options")

    # uncollapse dict['main'], used by mescore for parameters
    ops0 = lcp.default_ops()
    main_params = ops0.pop("main", {})
    ops0.update(main_params)

    # Add arguments for each key in the flattened dictionary
    for k, default_val in ops0.items():
        v = dict(default=default_val, help=f"{k} : {default_val}")
        if isinstance(v["default"], (np.ndarray, list)) and v["default"]:
            v["nargs"] = "+"
            v["type"] = type(v["default"][0])
        if k in ["batch_path", "batch-path"]:
            v["default"] = None  # required
            v["type"] = str
            v["dest"] = "batch_path"
        if k in ["data_path", "data-path"]:
            v["default"] = None  # required
            v["type"] = parse_data_path
            v["nargs"] = "+"
            v["dest"] = "data_path"
        parser.add_argument(f"--{k}", **v)
    return parser


def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns ops with parameters filled in.
    """
    args = parser.parse_args()
    dargs = vars(args)
    ops0 = lcp.default_ops()

    main_params = ops0.pop("main", {})
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
        elif not (default_key == type(default_key)(args_key)):  # type conversion, ensure type match
            ops[k] = type(default_key)(args_key)
            print(set_param_msg.format(k, ops[k]))
    return args, ops


def get_matching_main_params(args):
    """
    Match arguments supplied through the cli with parameters found in the defaults.
    """
    matching_params = {
        k: getattr(args, k)
        for k in lcp.default_ops()["main"].keys()
        if hasattr(args, k)
    }
    return matching_params


def main():
    args, ops = parse_args(add_args(argparse.ArgumentParser(description="LBM-Caiman pipeline parameters")))
    if args.version:
        print("lbm_caiman_python v{}".format(version))
        return
    if not args.batch_path:
        print("No batch path provided. Provide a path to save results in a dataframe.")
        return
    print("Batch path provided, retrieving batch:")
    print(args.batch_path)
    try:
        df = mc.load_batch(args.batch_path)
    except:
        print(f"No dataframe exists at {args.batch_path}")

        # Ask user if they want to create a new DataFrame
        create_new = input("Would you like to create a new batch DataFrame? (yes/no): ").strip().lower()

        if create_new == 'yes':
            # Prompt for save location, defaulting to the current batch path
            save_path = input(f"Enter save path (default: {args.batch_path}): ").strip()
            if not save_path:
                save_path = args.batch_path

            # Create a new DataFrame and save it to the specified location
            df = mc.create_batch(save_path)
            print(f"New batch DataFrame created and saved at {save_path}.")
        else:
            print("No new batch created. Exiting.")
            df = None  # Ensure `df` is unset or handled appropriately for downstream code
    # start batch manipulation
    if isinstance(args.rm, (int, list)):
        if args.rm > len(df.index):
            raise ValueError(f'Attempting to delete row {args.rm}. Dataframe size: {df.index}')

        print(f"Deleting row {args.rm} from the following dataframe:")
        print(f"--------------------------")
        print(df)
        if args.force:
            safe = False
        else:
            safe = True
        try:
            df = lcp.batch.delete_batch_rows(
                df, [args.rm], remove_data=args.remove_data, safe_removal=safe
            )
            df = df.caiman.reload_from_disk()
            print(df)
        except Exception as e:
            print(f"Cannot remove row, this likely occured because there was a downstream item ran on this batch "
                  f"item. Try with --force.")
    if args.run:
        if isinstance(args.data_path, int):
            input_movie_path = df.iloc[args.data_path]
            if isinstance(args.data_path, list):
                input_movie_path = df.iloc[args.data_path]
        elif isinstance(args.data_path, Path):
            input_movie_path = args.data_path
            mc.set_parent_raw_data_path(str(args.data_path))
        else:
            raise ValueError(f'{args.data_path} is not a valid data_path.')

        # Add and run batch item
        df.caiman.add_item(
            algo=args.run,
            input_movie_path=input_movie_path,
            params={"main": get_matching_main_params(args)},
            item_name="lbm-batch-item",
        )
        print(f"Running {args.run} -----------")
        df.iloc[-1].caiman.run()
        df = df.caiman.reload_from_disk()

        # Additional algorithm run (example)
        algo = "cnmf"
        df.caiman.add_item(
            algo=algo,
            input_movie_path=df.iloc[-1],
            params={"main": get_matching_main_params(args)},
            item_name="item_name",
        )
        print(f"Running {algo} -----------")
        df.iloc[-1].caiman.run()
        df = df.caiman.reload_from_disk()
        print("Processing complete -----------")
    else:
        print(df)


if __name__ == "__main__":
    main()
