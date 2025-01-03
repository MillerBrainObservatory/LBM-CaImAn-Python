import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm

from .util.io import get_files_ext
from .util.quality import get_cnmf_plots
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


def _contours_from_df(df):
    plots = {}
    for _, row in df.iterrows():
        if isinstance(row["outputs"], dict) and not row["outputs"].get("success") or row["outputs"] is None:
            continue

        if row["algo"] == "cnmf":
            plots[f"{row.uuid}"] = get_cnmf_plots(row.cnmf.get_output())
    return plots


def plot_summary(df, savepath=None):
    plots = _contours_from_df(df)
    for uuid, (contours, bg) in plots.items():
        _, centers = contours
        if not centers:
            continue
        print(f"Plotting {uuid}...")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(bg, cmap='magma')
        for center in centers:
            ax.scatter(center[0], center[1], color="blue", s=2, alpha=0.5)
        ax.set_title(f"Centers for {uuid}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        if savepath:
            save_name = Path(savepath) / f"{uuid}_segmentation_plot.png"
            print(f"Saving to {save_name}!")
            plt.savefig(save_name.expanduser(), dpi=600, bbox_inches="tight")


def plot_cnmf_components(df, savepath=None):
    """
    Generate and optionally save segmentation plots for CNMF components.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing CNMF component information.
    savepath : str or Path, optional
        Directory to save the generated plots. If None, plots are displayed but not saved.

    Returns
    -------
    None
    """
    plots = _contours_from_df(df)
    for uuid, (good, bad) in plots.items():
        print(f"Plotting {uuid}...")
        fig, ax = plt.subplots(1, 2, figsize=(8, 8))
        ax[0].imshow(good, cmap='magma')
        ax[0].set_title("Accepted Components")
        ax[1].imshow(bad, cmap='magma')
        ax[1].set_title("Rejected Components")
        ax[0].axis("off")
        ax[1].axis("off")
        plt.show()
        if savepath:
            save_name = Path(savepath) / f"{uuid}_segmentation_plot.png"
            print(f"Saving to {save_name}.")
            plt.savefig(save_name.expanduser(), dpi=600, bbox_inches="tight")

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

