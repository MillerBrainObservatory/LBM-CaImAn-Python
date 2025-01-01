import sys
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm

from .util.io import get_files_ext
from .util.quality import reshape_spatial
from .batch import load_batch


def get_pickle_files(data_path):
    files = get_files_ext(data_path, '.pickle', 3)
    if not files:
        raise ValueError(f"No .pickle files found in {data_path} or its subdirectories.")
    return files


def get_cnmf_items(files):
    """
    Load all cnmf items from a list of .pickle files.

    Parameters
    ----------
    files : list
        List of .pickle files to load.
    """
    cnmf_rows = []
    for file in files:
        try:
            df = load_batch(file)
            df['batch_path'] = file
        except Exception as e:
            print(f"Error loading {file}: {e}", file=sys.stderr)
            continue

        for _, row in df.iterrows():
            if isinstance(row["outputs"], dict) and not row["outputs"].get("success") or row["outputs"] is None:
                continue
            if row["algo"] == "cnmf":
                cnmf_rows.append(row)
    return cnmf_rows


def _num_traces_from_rows(rows):
    df = pd.DataFrame(rows)
    df["num_traces"] = [row.cnmf.get_temporal().shape[0] for row in rows]
    return df


def _accepted_rejected_from_rows(rows):
    df = pd.DataFrame(rows)
    df["num_good"] = [len(row.cnmf.get_output().estimates.idx_components) for row in rows]
    df["num_bad"] = [len(row.cnmf.get_output().estimates.idx_components_bad) for row in rows]
    return df


def get_background_image(row, background_image):
    if background_image == "corr":
        return row.caiman.get_corr_image()
    if background_image == "pnr":
        return row.caiman.get_pnr_image()
    if background_image == "max_proj":
        return row.caiman.get_projection("max")
    if background_image == "mean_proj":
        return row.caiman.get_projection("mean")
    if background_image == "std_proj":
        return row.caiman.get_projection("std")
    if background_image == "reshaped":
        return reshape_spatial(row.cnmf.get_output())
    raise ValueError(
        f"Background image type: {background_image} not recognized. Must be one of: max_proj, min_proj, mean_proj, "
        f"std_proj, pnr, or corr."
    )


def _contours_from_df(df, background_image="max_proj"):
    plots = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing loading contours. This processes ~200 neurons / "
                                                          "second."):
        if isinstance(row["outputs"], dict) and not row["outputs"].get("success") or row["outputs"] is None:
            continue
        if row["algo"] == "cnmf":
            bg = get_background_image(row, background_image)
            plots[f"{row.uuid}"] = (row.cnmf.get_contours("good"), bg)
    return plots


def plot_summary(df, savepath=None):
    plots = _contours_from_df(df, background_image="reshaped")
    for uuid, (contours, corr) in plots.items():
        _, centers = contours
        if not centers:
            continue
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(corr.T, cmap="gray")
        for center in centers:
            ax.scatter(center[0], center[1], color="blue", s=5, alpha=0.5)
        ax.set_title(f"Centers for {uuid}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        if savepath:
            print(f"Saving to {savepath / uuid}.png")
            save_name = savepath / f"{uuid}.png"
            plt.savefig(save_name , dpi=300, bbox_inches="tight")


def summarize_cnmf(rows):
    """
    Summarize CNMF results from a list of rows.
    Returns a DataFrame with the following columns:
    - uuid (str): UUID of the row.
    - algo_duration (float): Duration of the algorithm in seconds.
    - num_traces (int): Number of traces detected.
    - num_good (int): Number of accepted traces.
    - num_bad (int): Number of rejected traces.

    """
    df_temporal = _num_traces_from_rows(rows)
    df_comp = _accepted_rejected_from_rows(rows)
    merged_df = pd.merge(
        df_temporal[["batch_path", "algo_duration", "num_traces"]],
        df_comp[["batch_path", "num_good", "num_bad"]],
        on="batch_path"
    )
    return merged_df
