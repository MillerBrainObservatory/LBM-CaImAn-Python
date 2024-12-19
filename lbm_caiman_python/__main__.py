# Heavily adapted from suite2p
import numpy as np
import argparse
import logging
from pathlib import Path
from functools import partial
import lbm_caiman_python as lcp
import mesmerize_core as mc

current_file = Path(__file__).parent

print = partial(print, flush=True)


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
    Add command-line arguments to the parser, dynamically adding arguments
    for each key in the `ops` dictionary.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which arguments are added.

    Returns
    -------
    argparse.ArgumentParser
        The parser with added arguments.
    """
    default_ops = lcp.default_ops()["main"]
    for param, default_value in default_ops.items():
        param_type = type(default_value)
        if param_type == bool:
            parser.add_argument(f'--{param}', type=int, choices=[0, 1], help=f'Set {param} (default: {default_value})')
        elif param_type in [int, float, str]:
            parser.add_argument(f'--{param}', type=param_type, help=f'Set {param} (default: {default_value})')
        else:
            parser.add_argument(f'--{param}', help=f'Set {param} (default: {default_value})')

    parser.add_argument('--ops', type=str, help='Path to the ops .npy file.')
    parser.add_argument('--save_params', type=str, help='Path to save the ops parameters.')
    parser.add_argument('--version', action='store_true', help='Show version information.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--batch_path', type=str, help='Path to the batch file.')
    parser.add_argument('--create', action='store_true', help='Create a new batch.')
    parser.add_argument('--rm', type=int, nargs='+', help='Indices of batch rows to remove.')
    parser.add_argument('--force', action='store_true', help='Force removal without safety checks.')
    parser.add_argument('--remove_data', action='store_true', help='Remove associated data.')
    parser.add_argument('--clean', action='store_true', help='Clean unsuccessful batch items.')
    parser.add_argument('--show_params', type=int, help='Index of batch row to show parameters.')
    parser.add_argument('--run', type=str, nargs='+', help='Algorithms to run (e.g., mcorr, cnmf).')
    parser.add_argument('--data_path', type=str, help='Path to the input data.')

    return parser


def load_ops(args):
    """
    Load or create the 'ops' dictionary from a file or default parameters.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing the 'ops' path and 'save_params' option.

    Returns
    -------
    dict
        The loaded or default 'ops' dictionary.
    """
    if args.ops:
        ops_path = Path(args.ops)
        if ops_path.is_dir():
            raise ValueError(
                f"Given ops path {ops_path} is a directory. Please use a fully qualified path, "
                f"including the filename and file extension, i.e. /path/to/ops.npy."
            )
        elif not ops_path.is_file():
            raise FileNotFoundError(f"Given ops path {ops_path} is not a file.")
        ops = np.load(ops_path, allow_pickle=True).item()
    else:
        ops = lcp.default_ops()

    if args.save_params:
        save_path = Path(args.save_params)
        if save_path.is_dir():
            savename = save_path / "ops.npy"
        else:
            savename = save_path.with_suffix(".npy")
        print(f"Saving parameters to {savename}")
        np.save(str(savename.resolve()), ops)

    return ops


def get_matching_main_params(args):
    """
    Match arguments supplied through the CLI with parameters found in the defaults.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing potential parameter overrides.

    Returns
    -------
    dict
        A dictionary of parameters matching the default 'main' keys.
    """
    return {
        k: getattr(args, k)
        for k in lcp.default_ops()["main"].keys()
        if hasattr(args, k) and getattr(args, k) is not None
    }


def update_ops_with_matching_params(ops, matching_params):
    """
    Update the current 'ops' dictionary with the matching parameters.

    Parameters
    ----------
    ops : dict
        The original 'ops' dictionary.
    matching_params : dict
        The parameters to overwrite in the 'ops' dictionary.

    Returns
    -------
    dict
        The updated 'ops' dictionary.
    """
    ops["main"].update(matching_params)
    return ops


def save_ops(ops: dict, savename):
    """
    Save the 'ops' dictionary to a .npy file.

    Parameters
    ----------
    ops : dict
        The 'ops' dictionary to save.
    savename : str or Path
        The file path where to save the 'ops' dictionary.
    """
    np.save(str(Path(savename).resolve()), ops)
    print(f"Parameters saved to {savename}")


def parse_args(parser: argparse.ArgumentParser):
    """
    Parse arguments and apply overrides to the ops dictionary.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments.
    ops : dict
        The updated 'ops' dictionary with CLI overrides applied.
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
        args_key = dargs.get(k)
        if args_key is not None:  # CLI argument exists
            if isinstance(default_key, bool):
                args_key = bool(int(args_key))  # bool("0") is True, so we convert it properly
            else:
                args_key = type(default_key)(args_key)  # Ensure type match
            if default_key != args_key:
                ops[k] = args_key
                print(set_param_msg.format(k, ops[k]))

    return args, ops


def main():
    """
    The main function that orchestrates the CLI operations.
    """
    print("Beginning processing run ...")
    parser = argparse.ArgumentParser(description="LBM-Caiman pipeline parameters")
    parser = add_args(parser)
    args = parse_args(parser)

    # Handle version
    if args.version:
        print("lbm_caiman_python v{}".format(lcp.__version__))
        return

    # Setup logging
    if args.debug:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
        backend = "local"
    else:
        backend = None

    # Ensure batch_path is provided
    if not args.batch_path:
        parser.print_help()
        return

    # Load or create batch
    df = None
    batch_path = Path(args.batch_path).expanduser()
    print(f"Batch path provided: {batch_path}")

    if batch_path.is_file():
        print("Found existing batch.")
        df = mc.load_batch(batch_path)
    elif batch_path.is_dir():
        raise ValueError(
            f"Given batch path {batch_path} is a directory. Please use a fully qualified path, including "
            f"the filename and file extension, i.e. /path/to/batch.pickle."
        )
    elif args.create:
        df = mc.create_batch(batch_path)
        print(f'Batch created at {batch_path}')
    else:
        print('No batch found. Use --create to create a new batch.')
        return

    # Handle removal of batch rows
    if args.rm:
        print(
            "--rm provided as an argument. Checking the index(s) to delete are valid for this dataframe."
        )
        safe = not args.force
        if args.force:
            print(
                "--force provided as an argument. Performing unsafe deletion."
                " (This action may delete an mcorr item with an associated cnmf processing run)"
            )
        else:
            print("--force not provided as an argument. Performing safe deletion.")

        for arg in args.rm:
            if arg >= len(df.index) or arg < -len(df.index):
                raise ValueError(
                    f"Attempting to delete row {arg}. DataFrame size: {len(df.index)}"
                )

        try:
            df = lcp.batch.delete_batch_rows(
                df, args.rm, remove_data=args.remove_data, safe_removal=safe
            )
            df = df.caiman.reload_from_disk()
        except Exception as e:
            print(
                f"Cannot remove row, this likely occurred because there was a downstream item run on this batch "
                f"item. Try with --force. Error: {e}"
            )
        return  # Exit after removal

    # Handle cleaning of batch
    if args.clean:
        print("Cleaning unsuccessful batch items and associated data.")
        print(f"Previous DF size: {len(df.index)}")
        df = lcp.batch.clean_batch(df)
        print(f"Cleaned DF size: {len(df.index)}")
        return  # Exit after cleaning

    # Handle showing parameters
    if args.show_params is not None:
        try:
            params = df.iloc[args.show_params]["params"]
            print_params(params)
        except IndexError:
            print(f"Index {args.show_params} is out of bounds for the DataFrame.")
        return

    # Handle running algorithms
    if args.run:
        # Load or initialize ops
        ops = load_ops(args)

        # Get matching parameters from CLI args and update ops
        matching_params = get_matching_main_params(args)
        ops = update_ops_with_matching_params(ops, matching_params)

        # Save ops if requested
        if args.save_params:
            save_ops(ops, args.save_params)

        # Determine input movie path and metadata
        input_movie_path = None
        metadata = None

        if args.data_path is None:
            print(
                "No argument given for --data_path. Using the last row of the dataframe."
            )
            if len(df.index) > 0:
                args.data_path = -1
            else:
                raise ValueError(
                    'Attempting to run a batch item without providing a data path and with an empty '
                    'dataframe. Supply a data path with --data_path followed by the path to your input '
                    'data.'
                )

        if isinstance(args.data_path, int):
            if not (-len(df.index) <= args.data_path < len(df.index)):
                raise ValueError(
                    f"data_path index {args.data_path} is out of bounds for the DataFrame with size {len(df.index)}."
                )
            row = df.iloc[args.data_path]
            in_algo = row["algo"]
            assert (
                    in_algo == "mcorr"
            ), f"Input algorithm must be mcorr, algo at idx {args.data_path}: {in_algo}"
            if (
                    isinstance(row["outputs"], dict)
                    and row["outputs"].get("success") is False
            ):
                raise ValueError(
                    f"Given data_path index {args.data_path} references an unsuccessful batch item."
                )
            input_movie_path = row["input_movie_path"]  # Adjust based on actual DataFrame structure
            metadata = row.get("metadata")  # Adjust based on actual DataFrame structure
            if metadata is None:
                filename = Path(row["input_movie_path"])
                metadata = lcp.get_metadata(filename)
            parent = Path(input_movie_path).parent
            mc.set_parent_raw_data_path(parent)
        elif isinstance(args.data_path, (Path, str)):
            data_path = Path(args.data_path)
            if data_path.is_file():
                input_movie_path = data_path
                metadata = lcp.get_metadata(input_movie_path)
                parent = data_path.parent
                mc.set_parent_raw_data_path(parent)
            elif data_path.is_dir():
                files = list(data_path.glob("*.tif*"))
                if not files:
                    raise ValueError(f"No .tif files found in data_path: {data_path}")
                input_movie_path = files[0]
                metadata = lcp.get_metadata(input_movie_path)
                parent = input_movie_path.parent
                mc.set_parent_raw_data_path(parent)
            else:
                raise NotADirectoryError(
                    f"{args.data_path} is not a valid file or directory."
                )
        else:
            raise ValueError(f"{args.data_path} is not a valid data_path.")

        if metadata:
            ops["main"]["fr"] = metadata.get("frame_rate", ops["main"].get("fr"))
            ops["main"]["dxy"] = metadata.get("pixel_resolution", ops["main"].get("dxy"))
        else:
            print(
                "No metadata found for the input data. Please provide metadata."
            )

        for algo in args.run:
            if algo == "mcorr":
                df.caiman.add_item(
                    algo=algo,
                    input_movie_path=input_movie_path,
                    params=ops,
                    item_name="lbm-batch-item",
                )
                print(f"Running {algo} -----------")
                df.iloc[-1].caiman.run(backend=backend)
                df = df.caiman.reload_from_disk()
                print(f"Processing time: {df.iloc[-1].algo_duration}")
            elif algo in ["cnmf", "cnmfe"]:
                cnmf_params = {"main": get_matching_main_params(args)}
                df.caiman.add_item(
                    algo=algo,
                    input_movie_path=input_movie_path,
                    params=cnmf_params,
                    item_name="lbm-batch-item",
                )
                print(f"Running {algo} -----------")
                df.iloc[-1].caiman.run(backend=backend)
                df = df.caiman.reload_from_disk()
                print(f"Processing time: {df.iloc[-1].algo_duration}")
            else:
                print(f"Algorithm '{algo}' is not recognized and will be skipped.")

        return  # Exit after running algorithms

    # If only batch_path was provided without any operations
    print(df)
    print("Processing complete -----------")


if __name__ == "__main__":
    main()
