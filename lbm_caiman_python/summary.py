import sys
import time
from pathlib import Path
from typing import List

import numpy as np

import pandas as pd
import tifffile

from matplotlib import pyplot as plt
from tqdm import tqdm

from .batch import load_batch
from .helpers import _compute_metrics_with_temp_file, _compute_metrics
from .lcp_io import get_metrics_path
from .util.transform import calculate_centers

SUMMARY_PARAMS = (
    "K",
    "gSig",
    "gSig_filt",
    "min_SNR",
    "rval_thr"
)


def get_item_by_algo(files: list, algo="cnmf") -> pd.DataFrame:
    """
    Load all cnmf items from a list of .pickle files.

    Parameters
    ----------
    files : list
        List of .pickle files to load.
    algo : str
        Algorithm to filter by. Default is "cnmf". Options are "cnmf", "cnmfe", "mcorr".
    """
    temp_row = []
    for file in files:
        try:
            df = load_batch(file)
            df.paths.set_batch_path(file)
            df['batch_path'] = file
        except Exception as e:
            print(f"Error loading {file}: {e}", file=sys.stderr)
            continue

        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}."
        for _, row in df.iterrows():
            if (isinstance(row["outputs"], dict)
                    and not row["outputs"].get("success")
                    or row["outputs"] is None
            ):
                continue
            if row["algo"] == algo:
                temp_row.append(row)
    return pd.DataFrame(temp_row)


def plot_cnmf_components(data: pd.DataFrame | pd.Series, savepath: str | Path | None = None, marker_size=3):
    """
    Plot CNMF components for a DataFrame or a Series.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        A DataFrame containing CNMF data or a single Series (row) from the DataFrame.
    savepath : str, Path, or None, optional
        Directory to save the plots. If None, plots are not saved. Default is None.
    marker_size : int, optional
        Size of the markers for the center points. Set to 0 to skip drawing centers. Default is 3.

    Returns
    -------
    None
        Displays the plots and optionally saves them to the specified directory.

    Notes
    -----
    - The function handles both `pandas.DataFrame` and `pandas.Series` as input.
    - If `marker_size` is set to 0, no center points are drawn on the plot.
    - The `savepath` must be a valid directory path if saving is enabled.

    Examples
    --------
    For a DataFrame:
    >>> plot_cnmf_components(df, savepath="./plots", marker_size=5)

    For a single row (Series):
    >>> plot_cnmf_components(df.iloc[0], savepath="./plots", marker_size=5)
    """
    if isinstance(data, pd.DataFrame):
        for idx, row in data.iterrows():
            if isinstance(row["outputs"], dict) and not row["outputs"].get("success") or row["outputs"] is None:
                print(f"Skipping {row.uuid} as it is not successful.")
                continue

            if row["algo"] == "cnmf":
                model = row.cnmf.get_output()
                red_idx = model.estimates.idx_components_bad

                spatial_footprints = model.estimates.A
                dims = (model.dims[1], model.dims[0])

                max_proj = spatial_footprints.max(axis=1).toarray().reshape(dims)
                plt.imshow(max_proj, cmap="gray")

                # Check marker size
                if marker_size == 0:
                    print('Skipping drawing centers')
                else:
                    print(f'Marker size is set to {marker_size}')
                    centers = calculate_centers(spatial_footprints, dims)
                    colors = ['b'] * len(centers)

                    for i in red_idx:
                        colors[i] = 'r'
                    plt.scatter(centers[:, 0], centers[:, 1], c=colors, s=marker_size, marker='.')

                plt.tight_layout()
                plt.show()
                if savepath:
                    save_name = Path(savepath) / f"{row.uuid}_segmentation_plot.png"
                    print(f"Saving to {save_name}.")
                    plt.savefig(save_name.expanduser(), dpi=600, bbox_inches="tight")
    else:
        row = data
        if isinstance(row["outputs"], dict) and not row["outputs"].get("success") or row["outputs"] is None:
            print(f"Skipping {row.uuid} as it is not successful.")
            return

        if row["algo"] == "cnmf":
            model = row.cnmf.get_output()
            red_idx = model.estimates.idx_components_bad

            spatial_footprints = model.estimates.A
            dims = (model.dims[1], model.dims[0])

            max_proj = spatial_footprints.max(axis=1).toarray().reshape(dims)
            plt.imshow(max_proj, cmap="gray")

            # Check marker size
            if marker_size == 0:
                print('Skipping drawing centers')
            else:
                print(f'Marker size is set to {marker_size}')
                centers = calculate_centers(spatial_footprints, dims)
                colors = ['b'] * len(centers)

                for i in red_idx:
                    colors[i] = 'r'
                plt.scatter(centers[:, 0], centers[:, 1], c=colors, s=marker_size, marker='.')

            plt.tight_layout()
            plt.show()
            if savepath:
                save_name = Path(savepath) / f"{row.uuid}_segmentation_plot.png"
                print(f"Saving to {save_name}.")
                plt.savefig(save_name.expanduser(), dpi=600, bbox_inches="tight")


def summarize_cnmf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize CNMF results from a list of df.
    Returns a DataFrame with the following columns:
    - batch_path (str): Path of the batch.
    - algo_duration (float): Duration of the algorithm in seconds.
    - Total Traces (int): Number of traces detected.
    - Accepted (int): Number of accepted traces.
    - Rejected (int): Number of rejected traces.
    - K, gSig, gSiz, gSig_filt: Parameters used in the CNMF algorithm.
    """
    # Safely add new columns with traces / params
    return _params_from_df(_num_traces_from_df(df))


def concat_param_diffs(input_df, param_diffs):
    """
    Add parameter differences to the input DataFrame.

    Parameters
    ----------
    input_df : DataFrame
        The input DataFrame containing a 'batch_index' column.
    param_diffs : DataFrame
        The DataFrame containing the parameter differences for each batch.

    Returns
    -------
    input_df : DataFrame
        The input DataFrame with the parameter differences added.

    Examples
    --------
    >>> import pandas as pd
    >>> import lbm_caiman_python as lcp
    >>> import mesmerize_core as mc
    >>> batch_df = mc.load_batch('path/to/batch.pickle')
    >>> metrics_files = lcp.summary.compute_batch_metrics(batch_df)
    >>> metrics_df = lcp.summary.create_metrics_df(metrics_files)
    >>> param_diffs = batch_df.caiman.get_params_diffs("mcorr", item_name=batch_df.iloc[0]["item_name"])
    >>> final_df = lcp.concat_param_diffs(metrics_df, param_diffs)
    >>> print(final_df.head())
    """
    # add an empty column for each param diff
    for col in param_diffs.columns:
        if col not in input_df.columns:
            input_df[col] = None

    for i, row in input_df.iterrows():
        # raw data will not have an index in the dataframe
        if row['batch_index'] == -1:
            continue
        batch_index = int(row['batch_index'])

        if batch_index < len(param_diffs):
            param_diff = param_diffs.iloc[batch_index]

            for col in param_diffs.columns:
                input_df.at[i, col] = param_diff[col]

    input_df = input_df[
        ['mean_corr', 'mean_norm', 'crispness']
        + list(param_diffs.columns)
        + ['batch_index', 'uuid', 'metric_path']
        ]

    return input_df


def create_metrics_df(metrics_filepaths: list[str | Path]) -> pd.DataFrame:
    """
    Create a DataFrame from a list of metrics files.

    Parameters
    ----------
    metrics_filepaths : list of str or Path
        List of paths to the metrics files (.npz) containing 'correlations', 'norms',
        'smoothness', 'flows', and the batch item UUID.
        Typically, use the output of `compute_batch_metrics` to get the list of metrics files.

    Returns
    -------
    metrics_df : DataFrame
        A DataFrame containing the mean correlation, mean norm, crispness, UUID, batch index, and metric path.

    Examples
    --------
    >>> import pandas as pd
    >>> import lbm_caiman_python as lcp
    >>> import mesmerize_core as mc
    >>> batch_df = mc.load_batch('path/to/batch.pickle')
    >>> # overwrite=False will not recompute metrics if they already exist
    >>> metrics_files = lcp.summary.compute_batch_metrics(batch_df, overwrite=False)
    >>> metrics_df = lcp.create_metrics_df(metrics_files)
    >>> print(metrics_df.head())
    """
    metrics_list = []
    for i, file in enumerate(metrics_filepaths):
        with np.load(file) as f:
            corr = f['correlations']
            norms = f['norms']
            crispness = f['smoothness_corr']
            uuid = f['uuid']
            batch_index = f['batch_id']
        metrics_list.append({
            'mean_corr': np.mean(corr),
            'mean_norm': np.mean(norms),
            'crispness': float(crispness),
            'uuid': str(uuid),
            'batch_index': int(batch_index),
            'metric_path': file
        })
    return pd.DataFrame(metrics_list)


def create_summary_df(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for each batch of image data and output results in a DataFrame.

    Parameters
    ----------
    batch_df : DataFrame
        A DataFrame containing information about each batch of image data.
        Must be compatible with the mesmerize-core DataFrame API to call
        `get_output` on each row.

    Returns
    -------
    summary_df : DataFrame
        A DataFrame containing summary statistics for each batch of image data.
        This includes the minimum, maximum, mean, standard deviation, and percentiles.

    Examples
    --------
    >>> import pandas as pd
    >>> import lbm_caiman_python as lcp
    >>> import mesmerize_core as mc
    >>> batch_df = mc.load_batch('path/to/batch.pickle')
    >>> summary_df = lcp.create_summary_df(batch_df)
    >>> print(summary_df)
    """
    # Filter DataFrame to only process 'mcorr' df
    batch_df = batch_df[batch_df.item_name == 'mcorr']
    total_tqdm = len(batch_df) + 1  # +1 for the raw file processing

    with tqdm(total=total_tqdm, position=0, leave=True, desc="Computing Data Summary") as pbar:

        # Check for unique input files
        if batch_df.input_movie_path.nunique() != 1:
            raise ValueError(
                "\n\n"
                "The batch df have different input files. All input files must be the same.\n"
                "Please check the **input_movie_path** column in the DataFrame.\n\n"
                "To select a subset of your DataFrame with the same input file, you can use the following code:\n\n"
                "batch_df = batch_df[batch_df.input_movie_path == batch_df.input_movie_path.iloc[0]]\n"
            )

        raw_filepath = batch_df.iloc[0].caiman.get_input_movie_path()
        raw_data = tifffile.memmap(raw_filepath)
        met = {
            'item_name': 'Raw Data',
            'batch_index': 'None',
            'min': np.min(raw_data),
            'max': np.max(raw_data),
            'mean': np.mean(raw_data),
            'std': np.std(raw_data),
            'p1': np.percentile(raw_data, 1),
            'p50': np.percentile(raw_data, 50),
            'p99': np.percentile(raw_data, 99),
            'uuid': None
        }
        metrics_list = [met]
        pbar.update(1)

        for i, row in batch_df.iterrows():
            mmap_file = row.mcorr.get_output()
            metrics_list.append({
                'item_name': row.item_name,
                'batch_index': i,
                'min': np.min(mmap_file),
                'max': np.max(mmap_file),
                'mean': np.mean(mmap_file),
                'std': np.std(mmap_file),
                'p1': np.percentile(mmap_file, 1),
                'p50': np.percentile(mmap_file, 50),
                'p99': np.percentile(mmap_file, 99),
                'uuid': row.uuid,
            })
            pbar.update(1)
    return pd.DataFrame(metrics_list)


def compute_batch_metrics(batch_df: pd.DataFrame, overwrite: bool = False) -> List[Path]:
    """
    Compute and store various statistical metrics for each batch of image data.

    Parameters
    ----------
    batch_df : DataFrame, optional
        A DataFrame containing information about each batch of image data.
        Must be compatible with the mesmerize-core DataFrame API to call
        `get_params_diffs` and `get_output` on each row.
    overwrite : bool, optional
        If True, recompute and overwrite existing metric files. Default is False.

    Returns
    -------
    metrics_paths : list of Path
        List of file paths where metrics are stored for each batch.

    Examples
    --------
    >>> import pandas as pd
    >>> import lbm_caiman_python as lcp
    >>> import mesmerize_core as mc
    >>> batch_df = mc.load_batch('path/to/batch.pickle')
    >>> metrics_paths = lcp.compute_batch_metrics(batch_df)
    >>> print(metrics_paths)
    [Path('path/to/metrics1.npz'), Path('path/to/metrics2.npz'), ...]

    TODO: This can be made to run in parallel.
    """
    metrics_paths = []

    try:
        raw_filename = batch_df.iloc[0].caiman.get_input_movie_path()
    except Exception as e:
        print('Skipping raw data metrics computation.'
              'Could not find raw data file.'
              'Make sure to call mc.set_parent_raw_data_path(data_path) before calling this function.')
        raw_filename = None

    if raw_filename is not None:
        if not raw_filename.exists():
            raise FileNotFoundError(f"Raw data file {raw_filename} not found.")

        raw_metrics_path = get_metrics_path(raw_filename)
        if raw_metrics_path.exists() and not overwrite:
            print(f"Raw metrics file {raw_metrics_path} already exists. Skipping. To overwrite, set `overwrite=True`.")
        else:
            if raw_metrics_path.exists():
                print(f"Overwriting raw metrics file {raw_metrics_path}.")
                raw_metrics_path.unlink(missing_ok=True)

            start = time.time()
            raw_metrics_path = _compute_metrics_with_temp_file(raw_filename, overwrite=overwrite)
            print(f'Computed metrics for raw data in {time.time() - start:.2f} seconds.')

        metrics_paths.append(raw_metrics_path)

    for i, row in batch_df.iterrows():
        print(f'Processing batch index {i}...')

        if row.algo != 'mcorr':
            print(f"Skipping batch index {i} as algo is not 'mcorr'.")
            continue

        data = row.mcorr.get_output()
        final_size = data.shape[1:]

        # Pre-fetch metrics path
        metrics_path = get_metrics_path(row.mcorr.get_output_path())

        # Check if metrics already exist and skip if not overwriting
        if metrics_path.exists() and not overwrite:
            print(f"Metrics file {metrics_path} already exists. Skipping. To overwrite, set `overwrite=True`.")
            metrics_paths.append(metrics_path)
            continue

        if metrics_path.exists() and overwrite:
            print(f"Overwriting metrics file {metrics_path}.")
            metrics_path.unlink(missing_ok=True)

        try:
            start = time.time()
            _ = _compute_metrics(row.mcorr.get_output_path(), row.uuid, i, final_size[0], final_size[1])

            print(f'Computed metrics for batch index {i} in {time.time() - start:.2f} seconds.')
            metrics_paths.append(metrics_path)
        except Exception as e:
            print(f"Failed to compute metrics for batch index {i}. Error: {e}")

    return metrics_paths


def create_batch_summary(cnmf_df, mcorr_df) -> pd.DataFrame:
    succ_mcorr = _num_successful_from_df(mcorr_df)
    succ_cnmf = _num_successful_from_df(cnmf_df)
    unsucc_mcorr = len(mcorr_df) - succ_mcorr
    unsucc_cnmf = len(cnmf_df) - succ_cnmf

    return pd.DataFrame([
        {'algo': 'mcorr', 'Runs': len(mcorr_df), 'Successful': succ_mcorr,
         'Unsuccessful': unsucc_mcorr},
        {'algo': 'cnmf', 'Runs': len(cnmf_df), 'Successful': succ_cnmf,
         'Unsuccessful': unsucc_cnmf}
    ])


def _num_traces_from_df(df: pd.DataFrame) -> pd.DataFrame:
    # Safely add new columns with default values of None
    add_cols = ["Total Traces", "Accepted", "Rejected"]
    for col in add_cols:
        if col not in df.columns:
            df[col] = None

    for idx, row in df.iterrows():
        batch_df = load_batch(row["batch_path"])  # Ensure access using correct key
        item = batch_df[batch_df.uuid == row["uuid"]].iloc[0]
        if row["algo"] in ("cnmf", "cnmfe"):
            df.at[idx, "Total Traces"] = item.cnmf.get_temporal().shape[0]
            df.at[idx, "Accepted"] = len(item.cnmf.get_output().estimates.idx_components)
            df.at[idx, "Rejected"] = len(item.cnmf.get_output().estimates.idx_components_bad)
        else:
            df.at[idx, "Total Traces"] = None
            df.at[idx, "Accepted"] = None
            df.at[idx, "Rejected"] = None

    return df


# def _params_from_df(df: pd.DataFrame, params: tuple | list | None = None):
#     if params is None:
#         params = SUMMARY_PARAMS
#     for col in params:
#         if col not in df.columns:
#             df[col] = None
#     for idx, row in df.iterrows():
#         batch_df = load_batch(row.batch_path)
#         item = batch_df[batch_df.uuid == row.uuid].iloc[0]
#         for param in params:
#             df.at[idx, param] = item.params['main'].get(param)
#     return df

def _params_from_df(df: pd.DataFrame, params: tuple | list | None = None):
    if params is None:
        params = SUMMARY_PARAMS
    for col in params:
        if col not in df.columns:
            df[col] = None
    for idx, row in df.iterrows():
        batch_df = load_batch(row.batch_path)
        item = batch_df[batch_df.uuid == row.uuid].iloc[0]
        for param in params:
            value = item.params['main'].get(param)
            # Handle iterable values
            if isinstance(value, (list, tuple, np.ndarray)):
                df.at[idx, param] = str(value)  # Store as a string
            else:
                df.at[idx, param] = value
    return df



def _num_successful_from_df(df: pd.DataFrame) -> int:
    return len(df[df.outputs.apply(lambda x: x.get("success"))])
