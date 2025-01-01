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


def load_cnmf_rows(files):
    cnmf_rows = []
    for file in files:
        try:
            df = load_batch(file)
        except Exception as e:
            print(f"Error loading {file}: {e}", file=sys.stderr)
            continue

        for _, row in df.iterrows():
            if isinstance(row["outputs"], dict) and not row["outputs"].get("success") or row["outputs"] is None:
                continue
            if row["algo"] == "cnmf":
                cnmf_rows.append(row)
    return cnmf_rows


def get_num_temporal_from_rows(rows):
    df = pd.DataFrame(rows)
    df["num_traces"] = [row.cnmf.get_temporal().shape[0] for row in rows]
    return df


def get_good_bad_components_from_rows(rows):
    df = pd.DataFrame(rows)
    df["num_good"] = [len(row.cnmf.get_output().estimates.idx_components) for row in rows]
    df["num_bad"] = [len(row.cnmf.get_output().estimates.idx_components_bad) for row in rows]
    return df


def merge_summaries(df_temporal, df_comp):
    return pd.merge(
        df_temporal[["uuid", "algo_duration", "num_traces"]],
        df_comp[["uuid", "num_good", "num_bad"]],
        on="uuid"
    )


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


def plot_summary(plots):
    for filename, (contours, corr) in plots.items():
        _, centers = contours
        if not centers:
            continue
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(corr.T, cmap="gray")
        for center in centers:
            ax.scatter(center[0], center[1], color="blue", s=5, alpha=0.5)
        ax.set_title(f"Centers for {filename}")
        ax.axis("off")
        plt.show()


def summarize_cnmf(data_path, plot=False):
    files = get_pickle_files(data_path)
    cnmf_rows = load_cnmf_rows(files)
    df_temporal = get_num_temporal_from_rows(cnmf_rows)
    df_comp = get_good_bad_components_from_rows(cnmf_rows)
    merged_df = merge_summaries(df_temporal, df_comp)
    if plot:
        plots = _contours_from_df(merged_df, background_image="reshaped")
        plot_summary(plots)
        return merged_df, plots
    else:
        return merged_df
