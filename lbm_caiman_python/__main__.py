# Heavily adapted from suite2p
import numpy as np
import argparse
import logging
from pathlib import Path
from functools import partial
import lbm_caiman_python as lcp
import mesmerize_core as mc

current_file = Path(__file__).parent
with open(f"{current_file}/VERSION", "r") as VERSION:
    version = VERSION.read().strip()

# logging.getLogger("tensorflow").setLevel(logging.ERROR)

print = partial(print, flush=True)

DEFAULT_BATCH_PATH = Path().home() / "lbm_data" / "batch"
DEFAULT_DATA_PATH = Path().home() / "lbm_data" / "data"
if not DEFAULT_BATCH_PATH.is_dir():
    print(f"Creating default batch path in {DEFAULT_BATCH_PATH}.")
    DEFAULT_BATCH_PATH.mkdir(exist_ok=True, parents=True)
if not DEFAULT_DATA_PATH.is_dir():
    print(f"Creating default data path in {DEFAULT_DATA_PATH}.")
    DEFAULT_DATA_PATH.mkdir(exist_ok=True, parents=True)


def print_params(params, indent=5):
    for k, v in params.items():
        # if value is a dictionary, recursively call the function
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_params(v, indent + 4)
        else:
            print(" " * indent + f"{k}: {v}")


def parse_data_path(value):
    # try to convert to integer if possible, otherwise treat as a file path
    try:
        return int(value)
    except ValueError:
        return str(Path(value).expanduser().resolve())  # expand ~


def add_args(parser: argparse.ArgumentParser):
    """
    Adds ops arguments to parser.
    """

    parser.add_argument("batch_path", type=str, help="Path to batch file")  # Define as positional argument
    parser.add_argument(
        "--run",
        type=str,
        nargs="+",
        help="algorithm to run, options mcorr, cnmf or cnmfe",
    )
    parser.add_argument(
        "--rm",
        type=int,
        nargs="+",
        help="algorithm to run, options mcorr, cnmf or cnmfe",
    )
    parser.add_argument(
        "-c",
        "--clean",
        help="Clean unsuccessful batch items and associated data.",
        action="store_true",  # if present, sets args.clean to True
    )
    parser.add_argument(
        "--create",
        help="Create a new batch if one is not found at the given batch path.",
        action="store_true",  # if present, sets args.clean to True
    )
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

    parser.add_argument("-d", "--debug", action="store_false", help="Run with verbose debug logging.")
    parser.add_argument(
        "--name", type=str, help="Name of the batch, qualified as path/to/name.pickle."
    )
    parser.add_argument("--show_params", help="View parameters for the given index")
    parser.add_argument("--save_params", help="Store this parameter set to file")
    parser.add_argument(
        "--version", action="store_true", help="current pipeline version"
    )
    parser.add_argument("--ops", default=[], type=str, help="options")
    parser.add_argument("--data_path", "--data-path", dest="data_path", type=parse_data_path, default=None,
                        help="Path to data file or index of data in batch")

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
        elif not (
                default_key == type(default_key)(args_key)
        ):  # type conversion, ensure type match
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
    df = None
    parent = None
    print("Beginning processing run ...")
    args, ops = parse_args(
        add_args(argparse.ArgumentParser(description="LBM-Caiman pipeline parameters"))
    )
    if args.version:
        print("lbm_caiman_python v{}".format(version))
        return
    if args.debug:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
        backend = "local"
    else:
        backend = None
    if not args.batch_path:
        print("No batch path provided. Provide a path to save results in a dataframe.")
        return
    print("Batch path provided, retrieving batch:")
    args.batch_path = Path(args.batch_path).expanduser()
    print(args.batch_path)
    if Path(args.batch_path).is_file():
        print("Found existing batch.")
        df = mc.load_batch(args.batch_path)
    elif Path(args.batch_path).is_dir():
        print(
            f"Given batch path {args.batch_path} is a directory. Please use a fully qualified path, including "
            f"the filename and file extension, i.e. /path/to/batch.pickle."
        )
        # see if any existing pickle files
    elif args.create:
        df = mc.create_batch(args.batch_path)
        print(f'Batch created at {args.batch_path}')
    else:
        print('No batch found. Use --create to create a new batch.')
        return
    # start parsing main arguments (run, rm)
    if args.rm:
        print(
            "--rm provided as an argument. Checking the index(s) to delete are valid for this dataframe."
        )
        if args.force:  # stored false, access directly
            print(
                "--force provided as an argument."
                "Performing unsafe deletion."
                "(This action may delete an mcorr item with an associated cnmf processing run)"
            )
            safe = False
        else:
            print("--force not provided as an argument. Performing safe deletion.")
            safe = True
        for arg in args.rm:
            if arg > len(df.index):
                raise ValueError(
                    f"Attempting to delete row {args.rm}. Dataframe size: {df.index}"
                )
        try:
            df = lcp.batch.delete_batch_rows(
                df, args.rm, remove_data=args.remove_data, safe_removal=safe
            )
            df.caiman.reload_from_disk()
        except Exception as e:
            print(
                f"Cannot remove row, this likely occured because there was a downstream item ran on this batch "
                f"item. Try with --force."
            )
    elif args.clean:
        print("Cleaning unsuccessful batch items and associated data.")
        print(f"Previous DF size: {len(df.index)}")
        df = lcp.batch.clean_batch(df)
        print(f"Cleaned DF size: {len(df.index)}")
    elif args.show_params:
        from caiman.source_extraction.cnmf.params import CNMFParams
        # params = CNMFParams()
        params = df.iloc[int(args.show_params)]["params"]
        print_params(params)
    elif args.run:
        input_movie_path = None  # for setting raw_data_path

        # args.data_path can be an int or str/path
        # if int, use it as an index to the dataframe
        if args.data_path is None:
            print(
                "No argument given for --data_path. Using the last row of the dataframe."
            )
            if len(df.index) > 0:
                args.data_path = -1
            else:
                raise ValueError('Attemtping to run a batch item without giving a datapath and with an empty '
                                 'dataframe. Supply a data path with --data_path followed by the path to your input '
                                 'data.')
        if isinstance(args.data_path, int):
            row = df.iloc[args.data_path]
            in_algo = row["algo"]
            assert (
                    in_algo == "mcorr"
            ), f"Input algoritm must be mcorr, algo at idx {args.data_path}: {in_algo}"
            if (
                    isinstance(row["outputs"], dict)
                    and row["outputs"].get("success") is False
            ):
                raise ValueError(
                    f"Given data_path index {args.data_path} references an unsuccessful batch item."
                )
            input_movie_path = row
            filename = Path(df.iloc[0].caiman.get_input_movie())
            metadata = lcp.get_metadata(filename)
            parent = filename.parent
            mc.set_parent_raw_data_path(parent)
        elif isinstance(args.data_path, (Path, str)):
            if Path(args.data_path).is_file():
                input_movie_path = Path(args.data_path)
                parent = input_movie_path.parent
                metadata = lcp.get_metadata(input_movie_path)
                mc.set_parent_raw_data_path(parent)
            elif Path(args.data_path).is_dir():
                # regex all .p files to get pickled files
                files = [x for x in Path(args.data_path).glob("*.tif*")]
                if len(files) == 0:
                    raise ValueError(f"No datafiles found data_path: {args.data_path}")
                if len(files) >= 1:
                    # found a pickle file in the data_path
                    input_movie_path = files[0]
                    metadata = lcp.get_metadata(input_movie_path)
                    parent = Path(input_movie_path).parent
                    mc.set_parent_raw_data_path(parent)
                else:
                    raise NotADirectoryError(
                        f"{args.data_path} is not a valid directory."
                    )
            else:
                raise NotADirectoryError(
                    f"{args.data_path} is not a valid file or directory."
                )
        else:
            raise ValueError(f"{args.data_path} is not a valid data_path.")
        for algo in args.run:
            # RUN MCORR
            if algo == "mcorr":
                params = {"main": get_matching_main_params(args)}
                if metadata:
                    fr = metadata["frame_rate"]
                    dxy = metadata["pixel_resolution"]
                    params["main"]["fr"] = fr
                    params["main"]["dxy"] = dxy
                else:
                    metadata = lcp.get_metadata(input_movie_path)
                    if metadata is None:
                        print(
                            ## TODO: update this to be more descriptive
                            "No metadata found for the input data. Please provide metadata."
                        )
                    else:
                        fr = metadata["frame_rate"]
                        dxy = metadata["pixel_resolution"]
                        params["main"]["fr"] = fr
                        params["main"]["dxy"] = dxy

                df.caiman.add_item(
                    algo=algo,
                    input_movie_path=input_movie_path,
                    params=params,
                    item_name="lbm-batch-item",
                )
                print(f"Running {algo} -----------")
                df.iloc[-1].caiman.run(backend=backend)
                df = df.caiman.reload_from_disk()
                print(f"Processing time: {df.iloc[-1].algo_duration}")
            if algo in ["cnmf", "cnmfe"]:
                df.caiman.add_item(
                    algo=algo,
                    input_movie_path=input_movie_path,
                    params={"main": get_matching_main_params(args)},
                    item_name="item_name",
                )
                print(f"Running {algo} -----------")
                df.iloc[-1].caiman.run()
    else:  # if only batch_path was provided
        print(df)

    print("Processing complete -----------")


if __name__ == "__main__":
    main()
