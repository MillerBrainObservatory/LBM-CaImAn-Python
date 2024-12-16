import os
import shutil
import sys
import tempfile
import time

import cv2
import scipy
import tifffile
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from typing import Any as ArrayLike, List
import mesmerize_core as mc

import caiman as cm
from tqdm import tqdm


def plot_with_scalebars(image: ArrayLike, pixel_resolution: float):
    """
    Plot a 2D image with scale bars of 5, 10, and 20 microns.

    Parameters
    ----------
    image : ndarray
        A 2D NumPy array representing the image to be plotted.
    pixel_resolution : float
        The resolution of the image in microns per pixel.

    Returns
    -------
    None
    """
    scale_bar_sizes = [5, 10, 20]  # Sizes of scale bars in microns

    # Calculate the size of scale bars in pixels for each bar size
    scale_bar_lengths = [int(size / pixel_resolution) for size in scale_bar_sizes]

    # Create subplots to display each version of the image
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, scale_length, size in zip(axes, scale_bar_lengths, scale_bar_sizes):
        ax.imshow(image, cmap='gray')

        # Determine image dimensions for dynamic placement of scale bar
        image_height, image_width = image.shape

        # Scale bar thickness is 1% of the image height, but at least 2px thick
        bar_thickness = max(2, int(0.01 * image_height))  # Thinner bar than before

        # Center the scale bar horizontally and vertically
        bar_x = (image_width // 2) - (scale_length // 2)  # Centered horizontally
        bar_y = (image_height // 2) - (bar_thickness // 2)  # Centered vertically

        # Draw the scale bar
        ax.add_patch(patches.Rectangle((bar_x, bar_y), scale_length, bar_thickness,
                                       color='white', edgecolor='black', linewidth=1))

        # Add annotation for the scale bar (below the bar)
        font_size = max(10, int(0.03 * image_height))  # Font size relative to image size
        text = ax.text(bar_x + scale_length / 2, bar_y + bar_thickness + font_size + 5,
                       f'{size} μm', color='white', ha='center', va='top',
                       fontsize=font_size, fontweight='bold')

        # Apply a stroke effect to the text for better contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()
        ])

        # Remove axis for a clean image
        ax.axis('off')

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()


def generate_patch_view(image: ArrayLike, pixel_resolution: float, target_patch_size: int = 40,
                        overlap_fraction: float = 0.5):
    """
    Generate a patch visualization for a 2D image with approximately square patches of a specified size in microns.
    Patches are evenly distributed across the image, using calculated strides and overlaps.

    Parameters
    ----------
    image : ndarray
        A 2D NumPy array representing the input image to be divided into patches.
    pixel_resolution : float
        The pixel resolution of the image in microns per pixel.
    target_patch_size : float, optional
        The desired size of the patches in microns. Default is 40 microns.
    overlap_fraction : float, optional
        The fraction of the patch size to use as overlap between patches. Default is 0.5 (50%).

    Returns
    -------
    fig : matplotlib.figure.Figure
        A matplotlib figure containing the patch visualization.
    ax : matplotlib.axes.Axes
        A matplotlib axes object showing the patch layout on the image.

    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> data = np.random.random((144, 600))  # Example 2D image
    >>> pixel_resolution = 0.5  # Microns per pixel
    >>> fig, ax = generate_patch_view(data, pixel_resolution)
    >>> plt.show()
    """

    from caiman.utils.visualization import get_rectangle_coords, rect_draw

    # Calculate stride and overlap in pixels
    stride = int(target_patch_size / pixel_resolution)
    overlap = int(overlap_fraction * stride)

    # pad the image like caiman does
    def pad_image_for_even_patches(image, stride, overlap):
        patch_width = stride + overlap
        padded_x = int(np.ceil(image.shape[0] / patch_width) * patch_width) - image.shape[0]
        padded_y = int(np.ceil(image.shape[1] / patch_width) * patch_width) - image.shape[1]
        return np.pad(image, ((0, padded_x), (0, padded_y)), mode='constant'), padded_x, padded_y

    padded_image, pad_x, pad_y = pad_image_for_even_patches(image, stride, overlap)

    # Get patch coordinates
    patch_rows, patch_cols = get_rectangle_coords(padded_image.shape, stride, overlap)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(padded_image, cmap='gray')

    # Draw patches using rect_draw
    for patch_row in patch_rows:
        for patch_col in patch_cols:
            rect_draw(patch_row, patch_col, color='white', alpha=0.2, ax=ax)

    ax.set_title(f"Stride: {stride} pixels (~{stride * pixel_resolution:.1f} μm)\n"
                 f"Overlap: {overlap} pixels (~{overlap * pixel_resolution:.1f} μm)\n")
    plt.tight_layout()
    return fig, ax, stride, overlap


def compute_flow_single_frame(frame, templ, pyr_scale=.5, levels=3, winsize=100, iterations=15, poly_n=5,
                              poly_sigma=1.2 / 5, flags=0):
    flow = cv2.calcOpticalFlowFarneback(
        templ, frame, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    return flow


def _compute_metrics(fname, uuid, batch_id, final_size_x, final_size_y, swap_dim=False, pyr_scale=.5, levels=3,
                     winsize=100, iterations=15, poly_n=5, poly_sigma=1.2 / 5, flags=0,
                     resize_fact_flow=.2, template=None, gSig_filt=None):
    """
    Compute metrics for a given movie file.
    """
    print('Computing metrics for', fname)

    if not uuid:
        raise ValueError("UUID must be provided.")

    m = cm.load(fname)
    if gSig_filt is not None:
        m = cm.motion_correction.high_pass_filter_space(m, gSig_filt)

    max_shft_x = int(np.ceil((np.shape(m)[1] - final_size_x) / 2))
    max_shft_y = int(np.ceil((np.shape(m)[2] - final_size_y) / 2))
    max_shft_x_1 = - ((np.shape(m)[1] - max_shft_x) - (final_size_x))
    max_shft_y_1 = - ((np.shape(m)[2] - max_shft_y) - (final_size_y))
    if max_shft_x_1 == 0:
        max_shft_x_1 = None

    if max_shft_y_1 == 0:
        max_shft_y_1 = None
    m = m[:, max_shft_x:max_shft_x_1, max_shft_y:max_shft_y_1]
    if np.sum(np.isnan(m)) > 0:
        raise Exception('Movie contains NaN')

    img_corr = m.local_correlations(eight_neighbours=True, swap_dim=swap_dim)
    if template is None:
        tmpl = cm.motion_correction.bin_median(m)
    else:
        tmpl = template

    smoothness = np.sqrt(
        np.sum(np.sum(np.array(np.gradient(np.mean(m, 0))) ** 2, 0)))
    smoothness_corr = np.sqrt(
        np.sum(np.sum(np.array(np.gradient(img_corr)) ** 2, 0)))

    correlations = []
    count = 0
    sys.stdout.flush()
    for fr in tqdm(m, desc="Correlations"):
        count += 1
        correlations.append(scipy.stats.pearsonr(
            fr.flatten(), tmpl.flatten())[0])

    m = m.resize(1, 1, resize_fact_flow)
    norms = []
    flows = []
    count = 0
    sys.stdout.flush()
    for fr in tqdm(m, desc="Optical flow"):
        count += 1
        flow = cv2.calcOpticalFlowFarneback(
            tmpl, fr, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

        n = np.linalg.norm(flow)
        flows.append(flow)
        norms.append(n)

    # cast to numpy-loadable primatives, handle variable cases of None
    uuid = str(uuid) if uuid not in [None, 'None', 'nan'] else 'None'
    batch_id = int(batch_id) if batch_id not in [None, 'None', 'nan'] else -1
    np.savez(
        os.path.splitext(fname)[0] + '_metrics',
        uuid=uuid,
        batch_id=batch_id,
        flows=flows,
        norms=norms,
        correlations=correlations,
        smoothness=smoothness,
        tmpl=tmpl,
        smoothness_corr=smoothness_corr,
        img_corr=img_corr
    )


def _compute_metrics_with_temp_file(raw_fname: Path, overwrite=False) -> Path:
    """
    Wrapper for caiman.motion_correction.compute_metrics_motion_correction. Writes raw_file to a temporary memmapped file to
    run compute_metrics_motion_correction, and move the metrics file back to the fname directory.

    Needed due to compute_metrics_motion_correction not accepting memmapped files, just filenames.

    Parameters
    ----------
    raw_fname : Path
        The path to the raw data file. Must be a TIFF file.
    overwrite : bool, optional
        If True, recompute the metrics even if the file already exists. Default is False.

    Returns
    -------
    final_metrics_path : Path
        The path to the computed metrics file.

    Notes
    -----
    The final metrics files contains the following keys:
    - 'correlations': The correlation coefficients between frames.
    - 'flows': The flow vectors between frames.
    - 'norms': A list of magnitudes of optical flow for each frame. Represents the amount of motion in each frame.
    - 'smoothness': A measure of the sharpness of the image.
    """
    # make a new uuid with raw_{uuid}
    import uuid
    raw_uuid = f'raw_{uuid.uuid4()}'

    final_metrics_path = get_metrics_path(raw_fname)

    if final_metrics_path.exists() and not overwrite:
        return final_metrics_path

    data = tifffile.memmap(raw_fname)

    if final_metrics_path.exists() and overwrite:
        final_metrics_path.unlink()

    with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        tifffile.imwrite(temp_path, data)
        _ = _compute_metrics(temp_path, raw_uuid, None, data.shape[1], data.shape[2], swap_dim=False)

        temp_metrics_path = get_metrics_path(temp_path)

        if temp_metrics_path.exists():
            shutil.move(temp_metrics_path, final_metrics_path)
        else:
            raise FileNotFoundError(f"Expected metrics file {temp_metrics_path} not found.")
    finally:
        temp_path.unlink(missing_ok=True)

    return final_metrics_path


def get_metrics_path(fname: Path) -> Path:
    """
    Get the path to the computed metrics file for a given data file.
    Assumes the metrics file is to be stored in the same directory as the data file,
    with the same name stem and a '_metrics.npz' suffix.

    Parameters
    ----------
    fname : Path
        The path to the input data file.

    Returns
    -------
    metrics_path : Path
        The path to the computed metrics file.
    """
    return fname.with_stem(fname.stem + '_metrics').with_suffix('.npz')


def compute_batch_metrics(df: pd.DataFrame, overwrite: bool = False) -> List[Path]:
    """
    Compute and store various statistical metrics for each batch of image data.

    Parameters
    ----------
    df : DataFrame, optional
        A DataFrame containing information about each batch of image data.
        Must be compatible with the mesmerize-core DataFrame API to call
        `get_params_diffs` and `get_output` on each row.
    raw_filename : Path, optional
        The path to the raw data file. Must be a TIFF file. Default is None.
    overwrite : bool, optional
        If True, recompute and overwrite existing metric files. Default is False.

    Returns
    -------
    metrics_paths : list of Path
        List of file paths where metrics are stored for each batch.

    TODO: This can be made to run in parallel.
    """
    metrics_paths = []

    try:
        raw_filename = df.iloc[0].caiman.get_input_movie_path()
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

    for i, row in df.iterrows():
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


def create_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    # Filter DataFrame to only process 'mcorr' rows
    df = df[df.item_name == 'mcorr']
    total_tqdm = len(df) + 1  # +1 for the raw file processing

    with tqdm(total=total_tqdm, position=0, leave=True, desc="Computing Data Summary") as pbar:

        # Check for unique input files
        if df.input_movie_path.nunique() != 1:
            raise ValueError(
                "\n\n"
                "The batch rows have different input files. All input files must be the same.\n"
                "Please check the **input_movie_path** column in the DataFrame.\n\n"
                "To select a subset of your DataFrame with the same input file, you can use the following code:\n\n"
                "df = df[df.input_movie_path == df.input_movie_path.iloc[0]]\n"
            )

        raw_filepath = df.iloc[0].caiman.get_input_movie_path()
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

        for i, row in df.iterrows():
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


def create_metrics_df(metrics_p: list[str | Path]) -> pd.DataFrame:
    """
    Create a DataFrame from a list of metrics files.

    Parameters
    ----------
    metrics_p : list of str or Path
        List of paths to the metrics files (.npz) containing 'correlations', 'norms',
        'smoothness', 'flows', and the batch item UUID.
    """
    metrics_list = []
    for i, file in enumerate(metrics_p):
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


def add_param_diffs(input_df, param_diffs):
    """
    Add parameter differences to the input DataFrame.

    Input can be any dataframe as long as there exists a 'batch_index' column.

    Parameters
    ----------
    input_df : DataFrame
        The input DataFrame containing a 'batch_index' column.
    param_diffs : DataFrame
        The DataFrame containing the parameter differences for each batch.
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

    return input_df


def plot_optical_flows(input_df: pd.DataFrame, max_columns=4):
    """
    Plots the optical flow images from a DataFrame containing metrics information.

    Parameters
    ----------
    input_df : DataFrame
        DataFrame containing 'flows', 'batch_index', 'mean_corr', 'mean_norm', 'crispness', and other related columns.
    max_columns : int, optional
        Maximum number of columns to display in the plot. Default is 4.
    """
    num_graphs = len(input_df)
    num_rows = int(np.ceil(num_graphs / max_columns))

    fig, axes = plt.subplots(num_rows, max_columns, figsize=(20, 5 * num_rows))
    axes = axes.flatten()

    flow_images = []

    highest_corr_batch = input_df.loc[input_df['mean_corr'].idxmax()]['batch_index']
    highest_crisp_batch = input_df.loc[input_df['crispness'].idxmax()]['batch_index']
    lowest_norm_batch = input_df.loc[input_df['mean_norm'].idxmin()]['batch_index']

    for i, (index, row) in enumerate(input_df.iterrows()):
        # Avoid indexing beyond available axes if there are more rows than plots
        if i >= len(axes):
            break
        ax = axes[i]

        batch_idx = row['batch_index']
        metric_path = row['metric_path']
        with np.load(metric_path) as f:
            flows = f['flows']
            flow_img = np.mean(np.sqrt(flows[:, :, :, 0] ** 2 + flows[:, :, :, 1] ** 2), axis=0)
            del flows  # free up expensive array
            flow_images.append(flow_img)

        ax.imshow(flow_img, vmin=0, vmax=0.3, cmap='viridis')

        title_parts = []

        # Title Part 1: Item and Batch Index
        if batch_idx == -1:
            item_title = "Raw Data"
        else:
            item_title = f'Batch Index: {batch_idx}'

        if batch_idx == highest_corr_batch:
            item_title = f'Batch Index: {batch_idx} **(Highest Correlation)**'
        title_parts.append(item_title)

        mean_norm = row['mean_norm']
        norm_title = f'ROF: {mean_norm:.2f}'
        if batch_idx == lowest_norm_batch:
            norm_title = f'ROF: **{mean_norm:.2f}** (Lowest Norm)'
        title_parts.append(norm_title)

        smoothness = row['crispness']
        crisp_title = f'Crispness: {smoothness:.2f}'
        if batch_idx == highest_crisp_batch:
            crisp_title = f'Crispness: **{smoothness:.2f}** (Highest Crispness)'
        title_parts.append(crisp_title)

        title = '\n'.join(title_parts)

        ax.set_title(
            title,
            fontsize=14,
            fontweight='bold',
            color='black',
            loc='center'
        )

        ax.axis('off')

    # Turn off unused axes
    for i in range(len(input_df), len(axes)):
        axes[i].axis('off')

    cbar_ax = fig.add_axes((0.92, 0.2, 0.02, 0.6))
    norm = mpl.colors.Normalize(vmin=0, vmax=0.3)
    sm = mpl.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)

    cbar.set_label('Flow Magnitude', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 0.9, 1))
    plt.show()


def plot_residual_flows(results, num_batches=3):
    """
    Plot the top num_batches residual optical flows across batches.

    Parameters
    ----------
    results : DataFrame
        DataFrame containing 'uuid' and 'batch_index' columns.
    num_batches : int, optional
        Number of "best" batches to plot. Default is 3.
    """
    # Sort and filter for top batches by mean_norm
    results_sorted = results.sort_values(by='mean_norm')
    top_uuids = results_sorted['uuid'].values[:num_batches]
    results_filtered = results[results['uuid'].isin(top_uuids)]

    # Identify raw data UUID
    raw_uuid = results.loc[results['uuid'].str.contains('raw', case=False, na=False), 'uuid'].values[0]
    best_uuid = top_uuids[0]  # Best (lowest) value

    fig, ax = plt.subplots(figsize=(20, 10))

    # Color logic
    colors = plt.cm.Set1(np.linspace(0, 1, num_batches))  # Standout colors for other batches
    plotted_uuids = set()  # Track plotted UUIDs to avoid duplicates

    if raw_uuid in results['uuid'].values:
        row = results.loc[results['uuid'] == raw_uuid].iloc[0]
        metric_path = row['metric_path']
        batch_idx = row['batch_index']

        with np.load(metric_path) as metric:
            flows = metric['flows']

        residual_flows = [np.linalg.norm(flows[i] - flows[i - 1], axis=2).mean() for i in range(1, len(flows))]

        if raw_uuid == best_uuid:
            # Raw is also the best, so plot as blue dotted
            ax.plot(residual_flows, color='blue', linestyle='dotted', linewidth=2.5,
                    label=f'Best (Raw) | Batch Row Index: {batch_idx}')
        else:
            # Raw is not the best, so plot as red dashed
            ax.plot(residual_flows, color='red', linestyle='dotted', linewidth=2.5,
                    label=f'Raw Data')

        plotted_uuids.add(raw_uuid)  # Add raw UUID to avoid double plotting

    for i, row in results_filtered.iterrows():
        file_uuid = row['uuid']
        batch_idx = row['batch_index']

        # Avoid double-plotting the raw file if it was already plotted
        if file_uuid in plotted_uuids:
            continue

        metric_path = row['metric_path']

        with np.load(metric_path) as metric:
            flows = metric['flows']

        residual_flows = [np.linalg.norm(flows[i] - flows[i - 1], axis=2).mean() for i in range(1, len(flows))]

        if file_uuid == best_uuid:
            ax.plot(residual_flows, color='blue', linestyle='solid', linewidth=2.5,
                    label=f'Best Value | Batch Row Index: {batch_idx}')
        else:
            color_idx = list(top_uuids).index(file_uuid) if file_uuid in top_uuids else len(plotted_uuids) - 1
            ax.plot(residual_flows, color=colors[color_idx], linestyle='solid', linewidth=1.5,
                    label=f'Batch Row Index: {batch_idx}')

        plotted_uuids.add(file_uuid)

    ax.set_xlabel('Frames (downsampled)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual Optical Flow (ROF)', fontsize=12, fontweight='bold')
    ax.set_title(f'Batches with Lowest Residual Optical Flow', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12, title='Figure Key', title_fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_correlations(metrics_files, results, num_batches=3):
    """
    Plot the correlations across batches.

    Parameters
    ----------
    metrics_files : list of str
        List of paths to the metrics files (.npz) containing 'correlations'.
    results : DataFrame
        DataFrame containing 'uuid' and 'batch_index' columns.
    num_batches : int, optional
        Number of batches to plot. Default is 3.
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    if len(metrics_files) != len(results):
        raise ValueError("Number of metrics files does not match number of rows in results DataFrame")

    results_sorted = results.sort_values(by='mean_corr')
    top_uuids = results_sorted['uuid'].values[:num_batches]

    raw_uuid = results.loc[results['item_name'].str.contains('Raw Data', case=False, na=False), 'uuid'].values[0]

    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    plotted_uuids = set()

    for metrics_path in metrics_files:
        with np.load(metrics_path) as metric:
            correlations = metric['correlations']
            batch_idx = results.loc[results['uuid'] == file_uuid, 'batch_index'].values[0]

            # Extract and flatten the file's UUID properly
            if isinstance(metric['uuid'], np.ndarray):
                if metric['uuid'].ndim == 0:
                    file_uuid = metric['uuid'].item()
                else:
                    file_uuid = ''.join(map(str, metric['uuid']))
            else:
                file_uuid = str(metric['uuid'])

            if file_uuid not in top_uuids and file_uuid != raw_uuid:
                continue

            if file_uuid == raw_uuid and file_uuid not in plotted_uuids:
                ax.plot(correlations, linestyle='dotted', label='Raw Data', color='red', linewidth=3.5)
            elif file_uuid == top_uuids[0] and file_uuid not in plotted_uuids:
                ax.plot(correlations,
                        color='blue',
                        linewidth=2.5,
                        label='Lowest Correlations | Batch Row Index {batch_idx}'
                        )
            elif file_uuid in top_uuids and file_uuid not in plotted_uuids:
                color_idx = list(top_uuids).index(file_uuid) if file_uuid in top_uuids else len(plotted_uuids) - 1
                ax.plot(correlations, label=f'Batch Row Index {batch_idx}', color=colors[color_idx], linewidth=1.5)

            plotted_uuids.add(file_uuid)

    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Correlations Across Batches', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pathlib import Path
    import mesmerize_core as mc

    parent_path = Path().home() / "caiman_data"
    data_path = parent_path / 'out'  # where the output files from the assembly step are located
    batch_path = data_path / 'batch_v2.pickle'
    df = mc.load_batch(batch_path)
    # grab first 3 rows
    sub_df = df.iloc[:3]
    metrics_files = compute_batch_metrics(df, overwrite=False)
    metrics_df = create_metrics_df(metrics_files)
    merged = add_param_diffs(metrics_df, df.caiman.get_params_diffs("mcorr", item_name=df.iloc[0]["item_name"]))
    # summary_df = create_summary_df(df)
    # plot_optical_flows(input_df=merged)
    plot_residual_flows(metrics_df)
    # plot_correlations(metrics_files, final_df)
