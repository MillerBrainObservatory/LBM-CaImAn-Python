import sys
import time
from pathlib import Path
from typing import List

import numpy as np

import pandas as pd

from .batch import load_batch
from .helpers import _compute_metrics_with_temp_file, _compute_metrics
from .lcp_io import get_metrics_path

SUMMARY_PARAMS = (
    "K",
    "gSig",
    "gSig_filt",
    "min_SNR",
    "rval_thr"
)


def get_item_by_algo(files: list, algo="all") -> pd.DataFrame:
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

        for _, row in df.iterrows():
            if (isinstance(row["outputs"], dict)
                    and not row["outputs"].get("success")
                    or row["outputs"] is None
            ):
                continue
            if algo == "all":
                temp_row.append(row)
            elif row["algo"] == algo:
                temp_row.append(row)
    return pd.DataFrame(temp_row)


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
    >>> import lbm_mc as mc
    >>> batch_df = mc.load_batch('path/to/batch.pickle')
    >>> metrics_files = lcp.summary.compute_mcorr_metrics_batch(batch_df)
    >>> metrics_df = lcp.summary.metrics_df_from_files(metrics_files)
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


def metrics_df_from_files(metrics_filepaths: list[str | Path]) -> pd.DataFrame:
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
    >>> import lbm_mc as mc
    >>> batch_df = mc.load_batch('path/to/batch.pickle')
    >>> # overwrite=False will not recompute metrics if they already exist
    >>> metrics_files = lcp.summary.compute_mcorr_metrics_batch(batch_df, overwrite=False)
    >>> metrics_df = lcp.metrics_df_from_files(metrics_files)
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


def compute_mcorr_metrics_batch(batch_df: pd.DataFrame, overwrite: bool = False) -> List[Path]:
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
    >>> import lbm_mc as mc
    >>> batch_df = mc.load_batch('path/to/batch.pickle')
    >>> metrics_paths = lcp.compute_mcorr_metrics_batch(batch_df)
    >>> print(metrics_paths)
    [Path('path/to/metrics1.npz'), Path('path/to/metrics2.npz'), ...]

    TODO: This can be made to run in parallel.
    """
    metrics_paths = []

    try:
        raw_filename = batch_df.iloc[0].caiman.get_input_movie_path()
    except AttributeError:
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


def create_batch_summary(df) -> pd.DataFrame:
    """
    Create a summary of successful and unsuccessful runs for each algorithm.

    Parameters
    ----------
    df

    Returns
    -------

    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    elif not hasattr(df, 'item_name'):
        raise ValueError("Input DataFrame does not have an 'item_name' column.")

    mcorr_df = df[df.algo == 'mcorr']
    cnmf_df = df[df.algo.isin(['cnmf', 'cnmfe'])]
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
    df = df[df["algo"].isin(["cnmf", "cnmfe"])]

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