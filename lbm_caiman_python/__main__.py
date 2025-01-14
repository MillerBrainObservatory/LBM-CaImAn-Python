import numpy as np
import argparse
import logging
from pathlib import Path
from functools import partial

import pandas as pd

import lbm_caiman_python as lcp
import lbm_mc as mc

import lbm_caiman_python.visualize

current_file = Path(__file__).parent

print = partial(print, flush=True)


def _print_params(params, indent=5):
    for k, v in params.items():
        # if value is a dictionary, recursively call the function
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            _print_params(v, indent + 4)
        else:
            print(" " * indent + f"{k}: {v}")


def _parse_data_path(value):
    """
    Cast the value to an integer if possible, otherwise treat as a file path.
    """
    try:
        return int(value)
    except ValueError:
        return str(Path(value).expanduser().resolve())  # expand ~


def _parse_int_float(value):
    """ Cast the value to an integer if possible, otherwise treat as a float. """
    try:
        return int(value)
    except ValueError:
        return float(value)


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
    default_ops = lcp.default_params()["main"]

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

    parser.set_defaults(**default_ops)
    # non-run flags
    parser.add_argument('--ops', type=str, help='Path to the ops .npy file.')
    parser.add_argument('--save', type=str, help='Path to save the ops parameters.')
    parser.add_argument('--version', action='store_true', help='Show version information.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--batch_path', type=str, help='Path to the batch file.')
    parser.add_argument('--data_path', type=_parse_data_path, help='Path to the input data or index of the batch item.')
    # run flags
    parser.add_argument('--create', action='store_false', help='Create a new batch.')
    parser.add_argument('--rm', type=int, nargs='+', help='Indices of batch df to remove.')
    parser.add_argument('--force', action='store_true', help='Force removal without safety checks.')
    parser.add_argument('--remove_data', action='store_true', help='Remove associated data.')
    parser.add_argument('--clean', action='store_true', help='Clean unsuccessful batch items.')
    parser.add_argument('--run', type=str, nargs='+', help='Algorithms to run (e.g., mcorr, cnmf).')
    # --summary opts
    parser.add_argument('--summary', type=str, help='Get a summary of pickle files.')
    parser.add_argument('--cnmf', action="store_true", help='Get a summary of cnmf items.')
    parser.add_argument('--mcorr', action="store_true", help='Get a summary of mcorr files.')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth for searching pickle files. Default: 3.')
    parser.add_argument('--summary_plots', action='store_true', help='Get plots for the summary. Only works with --summary.')
    parser.add_argument('--marker_size', type=_parse_int_float, help='Scatterplot marker size for summary plots. Default: 3.')

    return parser


def load_ops(args, batch_path=None):
    """
    Load or create the 'ops' dictionary from a file or default parameters.
    Handles matching CLI arguments to the 'ops' dictionary.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing the 'ops' path and 'save' option.
    batch_path : str or Path
        Path to the batch file.

    Returns
    -------
    dict
        The loaded or default 'ops' dictionary.
    """
    # If a filepath was provided, use that
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
        if batch_path is not None:
            opsfile = Path(batch_path).parent / "ops.npy"
            if opsfile.is_file():
                ops = np.load(opsfile, allow_pickle=True).item()
                print(f"Loading parameters from {opsfile}")
            else:
                print(f"Using default parameters.")
                ops = lcp.default_params()
        else:
            print(f"Using default parameters.")
            ops = lcp.default_params()

    # Get matching parameters from CLI args and update ops
    main_ops = ops["main"]

    matching_params = {
        k: getattr(args, k)
        for k in main_ops.keys()
        if hasattr(args, k) and getattr(args, k) is not None
    }
    ops["main"].update(matching_params)

    for param in ops["main"]:
        # If defaults contain a list of length 2, handle cli with single entries
        if hasattr(main_ops[param], "__len__") and len(main_ops[param]) == 2:
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


def create_load_batch(batch_path):
    """
    Handles the creation or loading of a batch file based on the provided path.

    Parameters
    ----------
    batch_path : str or Path
        Path to the batch file or directory where the batch should be created/loaded.

    Returns
    -------
    df : object
        The loaded or newly created batch as returned by `mc.load_batch` or `mc.create_batch`.
    batch_path : Path
        The full path to the batch file as a Path object.

    Raises
    ------
    ValueError
        If the provided path has an invalid suffix or does not meet requirements for file creation.
    """
    batch_path = Path(batch_path).expanduser()
    print(f"Batch path provided: {batch_path}")

    if batch_path.exists():
        if batch_path.is_dir():
            # If given path is an existing directory, create/load batch.pickle inside it
            batch_path = batch_path / "batch.pickle"
            if batch_path.exists():
                print(f"Found existing batch {batch_path}")
                df = mc.load_batch(batch_path)
            else:
                print(f"Creating batch at {batch_path}")
                df = mc.create_batch(batch_path)
                print(f"Batch created at {batch_path}")
        else:
            # If given path is an existing file
            if batch_path.suffix != ".pickle":
                print(f"Wrong suffix: {batch_path.suffix}. Changing to .pickle: {batch_path.with_suffix('.pickle')}")
                batch_path = batch_path.with_suffix(".pickle")
            print(f"Found existing batch {batch_path}")
            df = mc.load_batch(batch_path)

    elif batch_path.suffix == '.pickle':
        # non-existent fully qualified filename
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        df = mc.create_batch(batch_path)
        print(f"Created batch at {batch_path}")
    elif batch_path.parent.exists() and batch_path.parent.is_dir():
        # If parent directory exists, create batch.pickle
        batch_path = batch_path / "batch.pickle"
        print(f"Creating batch at {batch_path}")
        df = mc.create_batch(batch_path)
        print(f"Batch created at {batch_path}")
    else:
        raise ValueError(f"Batch path {batch_path} cannot be created.")

    return df, batch_path


def resolve_data_path(data_path, df):
    """Resolves the data_path input to a list of file paths or a dataframe item.

    Parameters
    ----------
    data_path : str, Path, or int
        Path to a directory, file, or an integer index for a dataframe row.
    df : pandas.DataFrame
        Dataframe containing paths or other relevant data when data_path is an int.

    Returns
    -------
    files : list
        List of resolved file paths or dataframe items.

    Raises
    ------
    ValueError
        If data_path is invalid or no files are found.
    """
    if isinstance(data_path, (Path, str)):
        data_path = Path(data_path).expanduser().resolve()
        if data_path.is_file():
            return [data_path]
        elif data_path.is_dir():
            return list(data_path.rglob("*.tif*"))
        else:
            raise NotADirectoryError(f"{data_path} is not a valid file or directory.")
    elif isinstance(data_path, int):
        return [df.iloc[data_path]]
    else:
        raise ValueError(f"Invalid data_path: {data_path}")


def handle_input_data_path(input_path, ops):
    """Handles the metadata and raw data path for the input movie path.

    Parameters
    ----------
    input_path : Path or pd.Series
        Input movie path to process.
    ops : dict
        Parameters for the algorithm.

    Returns
    -------
    ops : dict
        Updated parameters with metadata if applicable.
    """
    input_movie_path = None  # what is fed into df.caiman.add_item()
    if isinstance(input_path, Path):
        if input_path.is_file():
            input_movie_path = input_path
            raw_data_path = input_path.parent
        elif input_path.is_dir():
            input_movie_path = input_path
            raw_data_path = input_path
        else:
            raise ValueError(f"Invalid input_path: {input_path}")

        metadata = lcp.get_metadata(input_path)
        mc.set_parent_raw_data_path(raw_data_path)
        ops['metadata'] = metadata

    elif isinstance(input_path, pd.Series):
        input_movie_path = input_path
        # make sure its a mcorr algo
        if input_path.algo != "mcorr":
            raise ValueError(f"Data-path is an index.\n"
                             f"Must provide the index of a mcorr item to run cnmf/cnmfe on.")
        output_path = input_path.mcorr.get_output_path()
        mc.set_parent_raw_data_path(output_path.parent)
    else:
        raise ValueError(f"Invalid input_path: {input_path}")
    return ops, input_movie_path


def run_item(algo, input_path, df, ops, backend):
    """Runs the specified algorithm on a single input item.

    Parameters
    ----------
    algo : str
        Algorithm to run (e.g., 'mcorr', 'cnmf', 'cnmfe').
    input_path : Path or pd.Series
        Input movie path or dataframe item to process.
    df : pandas.DataFrame
        Dataframe for managing processing results.
    ops : dict
        Parameters for the algorithm.
    backend : str
        Backend to use for processing.

    Returns
    -------
    df : pandas.DataFrame
        Updated dataframe after processing.
    """
    ops, input_movie_path = handle_input_data_path(input_path, ops)
    df.caiman.add_item(
        algo=algo,
        input_movie_path=input_movie_path,
        params=ops,
        item_name=f"{algo}-lbm",
    )
    print(f"Running {algo} -----------")
    df.iloc[-1].caiman.run(backend=backend)
    df = df.caiman.reload_from_disk()
    print(f"Processing time: {df.iloc[-1].algo_duration}")
    return df


def run_algorithm(algo, files, df, ops, backend):
    """Runs the specified algorithm on the input files.

    Parameters
    ----------
    algo : str
        Algorithm to run (e.g., 'mcorr', 'cnmf', 'cnmfe').
    files : list
        List of input file paths or dataframe items.
    df : pandas.DataFrame
        Dataframe for managing processing results.
    ops : dict
        Parameters for the algorithm.
    backend : str
        Backend to use for processing.

    Returns
    -------
    df : pandas.DataFrame
        Updated dataframe after processing.
    """
    if algo not in ["mcorr", "cnmf", "cnmfe"]:
        print(f"Algorithm '{algo}' is not recognized and will be skipped.\n"
              f"Available algorithms are: 'mcorr', 'cnmf', 'cnmfe'.")
        return df

    for input_movie_path in files:
        df = run_item(algo, input_movie_path, df, ops, backend)
    return df


def main():
    """
    The main function that orchestrates the CLI operations.
    """
    print("\n")
    print("-----------LBM-Caiman pipeline -----------")
    print("\n")
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

    if args.summary:
        if args.summary_plots:
            args.cnmf = True
            args.mcorr = True

        # find all .pickle files in the given directory
        files = lcp.get_files_ext(args.summary, '.pickle', args.max_depth)

        if not files:
            raise ValueError(f"No .pickle files found in {args.summary} or its subdirectories.")

        print(f"Found {len(files)} pickle files in {args.summary}.")
        batch_df = lcp.get_item_by_algo(files, algo="all")

        if batch_df.empty:
            print("No batch items found in the given pickle files.")

        print(f"----Summary of batch files in {args.summary}:")
        batch_summary_df = lcp.create_batch_summary(batch_df)
        print(batch_summary_df)
        print("\n")
        if args.cnmf:
            print("---Summary of CNMF items:")

            cnmf_df = batch_df[batch_df.algo == "cnmf"]
            cnmf_summary_df = lcp.summarize_cnmf(cnmf_df)
            print_cols = ["algo", "algo_duration", "Accepted", "Rejected", "K", "gSig"]

            # no max columns
            pd.set_option('display.max_columns', None)
            print_df = cnmf_summary_df[print_cols]
            formatted_output = "\n".join(print_df.to_string(index=False).splitlines())

            print(formatted_output)

            # save df to disk
            cnmf_summary_df.to_csv(args.summary + '/summary.csv')
            print(f"Summary saved to {args.summary}/summary.csv")
            print('See this summary for batch_paths.')

        if args.mcorr:
            mcorr_metrics_files = lcp.compute_mcorr_metrics_batch(batch_df)
            mcorr_metrics_df = lcp.metrics_df_from_files(mcorr_metrics_files)

            formatted_output = "\n".join(mcorr_metrics_df.to_string(index=False).splitlines())
            print("\n---Summary of MCORR items:")
            print(formatted_output)

        if args.summary_plots:
            print("Generating summary plots.")
            try:
                save_path = args.summary + "residual_flows.png"
                lcp.plot_residual_flows(mcorr_metrics_df, save_path=save_path)
                save_path = args.summary + "correlations.png"
                lcp.plot_correlations(mcorr_metrics_df, save_path=save_path)
                save_path = args.summary + "optical_flows.png"
                lcp.plot_optical_flows(mcorr_metrics_df, save_path=save_path)
            except Exception as e:
                print(f"Error generating summary plots: {e}")
            lbm_caiman_python.visualize.plot_spatial_components(cnmf_summary_df, savepath=args.summary, marker_size=args.marker_size)

        if args.run or args.rm or args.clean:
            print("Cannot run algorithms or modify batch when --summary is provided.")
        return

    if args.data_path is None:
        parser.print_help()

    if not args.batch_path:
        parser.print_help()
        return

    df, batch_path = create_load_batch(args.batch_path)
    ops = load_ops(args, batch_path)
    ops['package'] = {'version': lcp.__version__}

    # Handle removal of batch df
    if args.rm:
        print("--rm provided as an argument. Checking the index(s) to delete are valid for this dataframe.")
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
            lcp.batch.delete_batch_rows(df, args.rm, remove_data=args.remove_data, safe_removal=safe)
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
        files = resolve_data_path(args.data_path, df)
        if not files:
            print(f"No files found in {args.data_path}.")
            print(f"Current directory contents:\n")
            print("\n".join([str(f) for f in Path(args.data_path).rglob("*")]))
            return
        for algo in args.run:
            run_algorithm(algo, files, df, ops, backend)
            df = df.caiman.reload_from_disk()
            row = df.iloc[-1]
            if isinstance(row["outputs"], dict) and row["outputs"].get("success") is False or row["outputs"] is None:
                print(f"{algo} failed.")
                traceback = row["outputs"].get("traceback")
                if traceback:
                    print(f"Traceback: {traceback}")
            print(f'{df.iloc[-1].algo} duration: {df.iloc[-1].algo_duration}')

    print(df)
    print("Processing complete -----------")
    return


if __name__ == "__main__":
    main()
