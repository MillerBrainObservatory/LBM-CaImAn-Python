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
    """
    Cast the value to an integer if possible, otherwise treat as a file path.
    """
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
        elif param_type in [tuple, list] and len(default_value) == 2:
            inner_type = type(default_value[0])
            # Handle list/tuple arguments with 2 items
            parser.add_argument(f'--{param}', nargs='+', type=inner_type,
                                help=f'Set {param} (default: {default_value}). Provide one or two values.')
        else:
            parser.add_argument(f'--{param}', help=f'Set {param} (default: {default_value})')

    # Set default values so that args contains the defaults if no CLI input is given
    parser.set_defaults(**default_ops)
    parser.add_argument('--ops', type=str, help='Path to the ops .npy file.')
    parser.add_argument('--save', type=str, help='Path to save the ops parameters.')
    parser.add_argument('--version', action='store_true', help='Show version information.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--batch_path', type=str, help='Path to the batch file.')
    parser.add_argument('--data_path', type=parse_data_path, help='Path to the input data.')
    parser.add_argument('--create', action='store_false', help='Create a new batch.')
    parser.add_argument('--rm', type=int, nargs='+', help='Indices of batch rows to remove.')
    parser.add_argument('--force', action='store_true', help='Force removal without safety checks.')
    parser.add_argument('--remove_data', action='store_true', help='Remove associated data.')
    parser.add_argument('--clean', action='store_true', help='Clean unsuccessful batch items.')
    parser.add_argument('--run', type=str, nargs='+', help='Algorithms to run (e.g., mcorr, cnmf).')

    return parser


def load_ops(args):
    """
    Load or create the 'ops' dictionary from a file or default parameters.
    Handles matching CLI arguments to the 'ops' dictionary.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing the 'ops' path and 'save' option.

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

    if args.save:
        save_path = Path(args.save)
        if save_path.is_dir():
            savename = save_path / "ops.npy"
        else:
            savename = save_path.with_suffix(".npy")
        print(f"Saving parameters to {savename}")
        np.save(str(savename.resolve()), ops)

    # Get matching parameters from CLI args and update ops
    defaults = lcp.default_ops()["main"]

    matching_params = {
        k: getattr(args, k)
        for k in defaults.keys()
        if hasattr(args, k) and getattr(args, k) is not None
    }
    ops["main"].update(matching_params)

    for param in ops["main"]:
        # If defaults contain a list of length 2, handle cli with single entries
        if hasattr(defaults[param], "__len__") and len(defaults[param]) == 2:
            arg_value = getattr(args, param, None)  # value from cli
            if arg_value is not None:
                # if scalar
                if not isinstance(arg_value, (list, tuple)):
                    arg_value = [arg_value]
                if len(arg_value) == 1:
                    ops["main"][param] = [arg_value[0], arg_value[0]]
                elif len(arg_value) == 2:
                    ops["main"][param] = list(arg_value)
                else:
                    raise ValueError(
                        f"Invalid number of values for --{param}. Expected 1 or 2 values, got {len(arg_value)}.")
    return ops


def main():
    """
    The main function that orchestrates the CLI operations.
    """
    print("Beginning processing run ...")
    parser = argparse.ArgumentParser(description="LBM-Caiman pipeline parameters")
    parser = add_args(parser)
    args = parser.parse_args()

    # Handle version
    if args.version:
        print("lbm_caiman_python v{}".format(lcp.__version__))
        return

    # Setup logging/backend
    if args.debug:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
        backend = "local"
    else:
        backend = None

    if args.data_path is None:
        parser.print_help()
    if not args.batch_path:
        parser.print_help()
        return

    batch_path = Path(args.batch_path).expanduser()
    print(f"Batch path provided: {batch_path}")
    if batch_path.is_file():
        # make sure its a pickle file
        if batch_path.suffix != ".pickle":
            print(f"Wrong suffix: {batch_path.suffix}. Casting to .pickle: {batch_path.with_suffix('.pickle')}")
            batch_path = batch_path.with_suffix(".pickle")
        print(f"Found existing batch {batch_path}")
        df = mc.load_batch(batch_path)
    elif batch_path.is_dir():
        batch_path = batch_path / "batch.pickle"
        print(f"Found existing batch {batch_path}")
        df = mc.load_batch(batch_path)
    elif batch_path.parent.is_dir():
        print(f"Batch path {batch_path} is not a directory, but its parent is."
              f"Creating batch at {batch_path / 'batch.pickle'}")
        batch_path = batch_path / "batch.pickle"
        df = mc.create_batch(batch_path)
        print(f"Batch created at {batch_path}")
    else:
        print(f'{batch_path} is not a file, directory and does not have a valid parent directory. Exiting.')
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
            df = lcp.batch.delete_batch_rows(df, args.rm, remove_data=args.remove_data, safe_removal=safe)
        except Exception as e:
            print(
                f"Cannot remove row, this likely occurred because there was a downstream item run on this batch "
                f"item. Try with --force. Error: {e}"
            )
        return

    # Handle cleaning of batch
    if args.clean:
        print("Cleaning unsuccessful batch items and associated data.")
        print(f"Previous batch size: {len(df.index)}")
        df = lcp.batch.clean_batch(df)
        print(f"Cleaned batch size: {len(df.index)}")
        return  # Exit after cleaning

    # Handle running algorithms
    if args.run:
        ops = load_ops(args)
        if not isinstance(args.data_path, (Path, str)):
            raise ValueError("Data path must be a string or Path object.")

        data_path = Path(args.data_path).expanduser().resolve()
        if data_path.is_file():
            files = [data_path]
        elif data_path.is_dir():
            files = list(data_path.glob("*.tif*"))
            if not files:
                raise ValueError(f"No .tif files found in data_path: {data_path}")
        else:
            raise NotADirectoryError(f"{args.data_path} is not a valid file or directory.")

        for algo in args.run:
            if algo not in ["mcorr", "cnmf", "cnmfe"]:
                print(f"Algorithm '{algo}' is not recognized and will be skipped."
                      f"Avaliable algorithms are: 'mcorr', 'cnmf', 'cnmfe'.")
            if algo == "mcorr":
                for input_movie_path in files:
                    input_movie_path = Path(input_movie_path)
                    # TODO: update_ops() to handle metadata
                    metadata = lcp.get_metadata(input_movie_path)
                    ops["main"]["fr"] = metadata.get("frame_rate", ops["main"].get("fr"))
                    ops["main"]["dxy"] = metadata.get("pixel_resolution", ops["main"].get("dxy"))
                    mc.set_parent_raw_data_path(input_movie_path.parent)
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
            if algo in ['cnmf', 'cnmfe']:
                for input_movie_path in files:
                    input_movie_path = Path(input_movie_path)
                    mcorr_item = df[df.input_movie_path == input_movie_path.name]

                    # Check if exactly one row matches
                    if len(mcorr_item) == 0:
                        raise ValueError(f"No row found with input_movie_path == {input_movie_path.name}")
                    elif len(mcorr_item) > 1:
                        raise ValueError(
                            f"Multiple rows found with input_movie_path == {input_movie_path.name}. Expected only one "
                            f"match.")
                    mc.set_parent_raw_data_path(input_movie_path.parent)
                    if mcorr_item.empty:
                        print(f"No matching mcorr item found for {input_movie_path}."
                              f"Current batch items: {df.input_movie_path}."
                              f"Proceeding to run on the input movie.")
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
                    df.caiman.add_item(
                        algo=algo,
                        input_movie_path=mcorr_item.iloc[0],
                        params=ops,
                        item_name="lbm-batch-item",
                    )
                    print(f"Running {algo} -----------")
                    df.iloc[-1].caiman.run(backend=backend)
                    df = df.caiman.reload_from_disk()
                    print(f"Processing time: {df.iloc[-1].algo_duration}")
        return

    print(df)
    print("Processing complete -----------")


if __name__ == "__main__":
    main()
