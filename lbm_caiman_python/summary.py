import sys
from pathlib import Path
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm

from .util.io import get_files_ext
from .batch import load_batch


def get_pickle_files(data_path):
    files = get_files_ext(data_path, '.pickle', 3)
    if not files:
        raise ValueError(f"No .pickle files found in {data_path} or its subdirectories.")
    return files


def get_item_by_algo(files, algo="cnmf"):
    """
    Load all cnmf items from a list of .pickle files.

    Parameters
    ----------
    files : list
        List of .pickle files to load.
    """
    temp_row = []
    for file in files:
        try:
            df = load_batch(file)
            df['batch_path'] = file
        except Exception as e:
            print(f"Error loading {file}: {e}", file=sys.stderr)
            continue

        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}."
        for _, row in df.iterrows():
            if \
                    (isinstance(row["outputs"], dict)
                            and not row["outputs"].get("success")
                            or row["outputs"] is None
                    ):
                continue
            if row["algo"] == algo:
                temp_row.append(row)
    return temp_row


def plot_cnmf_components(df, savepath=None):
    for _, row in df.iterrows():
        if isinstance(row["outputs"], dict) and not row["outputs"].get("success") or row["outputs"] is None:
            continue

        if row["algo"] == "cnmf":
            model = row.cnmf.get_output()
            red_idx = model.estimates.idx_components_bad

            spatial_footprints = model.estimates.A

            dims = (model.dims[1], model.dims[0])
            centers = _calculate_centers(spatial_footprints, dims)

            colors = ['b'] * len(centers)

            for i in red_idx:
                colors[i] = 'r'

            max_proj = spatial_footprints.max(axis=1).toarray().reshape(dims)
            plt.imshow(max_proj, cmap="gray")
            plt.scatter(centers[:, 0], centers[:, 1], c="r", s=3)

            plt.tight_layout()
            plt.show()
            if savepath:
                save_name = Path(savepath) / f"{row.uuid}_segmentation_plot.png"
                print(f"Saving to {save_name}.")
                plt.savefig(save_name.expanduser(), dpi=600, bbox_inches="tight")


def plot_summary(df, savepath=None):
    """
    Plot a summary of the CNMF results.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    df.plot.bar(x="batch_path", y=["Accepted", "Rejected"], ax=ax[0])
    ax[0].set_title("Accepted vs Rejected Components")
    ax[0].set_ylabel("Number of Components")
    ax[0].set_xlabel("Batch Path")

    df.plot.bar(x="batch_path", y="Total Traces", ax=ax[1])
    ax[1].set_title("Total Traces Detected")
    ax[1].set_ylabel("Number of Traces")


def _calculate_centers(A, dims):
    def calculate_center_component(i):
        ixs = np.where(A[:, i].toarray() > 0.07)[0]
        return np.array(np.unravel_index(ixs, dims)).mean(axis=1)[::-1]

    # Use joblib to parallelize the center calculation for each column in A
    centers = Parallel(n_jobs=-1)(delayed(calculate_center_component)(i) for i in tqdm(range(A.shape[1]), desc="Calculating neuron center coordinates"))

    return np.array(centers)



def summarize_cnmf(rows):
    """
    Summarize CNMF results from a list of rows.
    Returns a DataFrame with the following columns:
    - batch_path (str): Path of the batch.
    - algo_duration (float): Duration of the algorithm in seconds.
    - Total Traces (int): Number of traces detected.
    - Accepted (int): Number of accepted traces.
    - Rejected (int): Number of rejected traces.
    - K, gSig, gSiz, gSig_filt: Parameters used in the CNMF algorithm.
    """
    df_temporal = _num_traces_from_rows(rows)
    df_comp = _accepted_rejected_from_rows(rows)
    df_params = _params_from_rows(rows)

    # Merge DataFrames step by step
    merged_df = pd.merge(
        df_temporal[["batch_path", "algo_duration", "Total Traces"]],
        df_comp[["batch_path", "Accepted", "Rejected"]],
        on="batch_path"
    )

    batch_path_idx = df_params.columns.get_loc('batch_path')
    df_batch_and_after = df_params.iloc[:, batch_path_idx:]

    merged_df = pd.merge(
        merged_df,
        df_batch_and_after,
        on="batch_path"
    )

    return merged_df


def _num_traces_from_rows(rows):
    df = pd.DataFrame(rows)
    df["Total Traces"] = [row.cnmf.get_temporal().shape[0] for row in rows]
    return df


def _accepted_rejected_from_rows(rows):
    df = pd.DataFrame(rows)
    df["Accepted"] = [len(row.cnmf.get_output().estimates.idx_components) for row in rows]
    df["Rejected"] = [len(row.cnmf.get_output().estimates.idx_components_bad) for row in rows]
    return df


def _params_from_rows(rows):
    df = pd.DataFrame(rows)
    params_to_query = ["K", "gSig", "min_SNR", "rval_thr"]
    for param in params_to_query:
        df[param] = [row.params["main"][param] for row in rows]
    return df
