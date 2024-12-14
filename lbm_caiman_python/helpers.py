import shutil
import tempfile
import time
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from typing import Any as ArrayLike, List
import mesmerize_core as mc

from caiman.motion_correction import compute_metrics_motion_correction


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


def _compute_metrics_with_temp_file(raw_fname: Path, overwrite=False) -> Path:
    """
    Wrapper for caiman.motion_correction.compute_metrics_motion_correction. Writes raw_file to a temporary memmapped file to
    run compute_metrics_motion_correction, and move the metrics file back to the raw_fname directory.

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
        _ = compute_metrics_motion_correction(temp_path, data.shape[1], data.shape[2], swap_dim=False)

        temp_metrics_path = get_metrics_path(temp_path)

        if temp_metrics_path.exists():
            shutil.move(temp_metrics_path, final_metrics_path)
        else:
            raise FileNotFoundError(f"Expected metrics file {temp_metrics_path} not found.")
    finally:
        temp_path.unlink(missing_ok=True)

    return final_metrics_path


def get_metrics_path(raw_fname: Path) -> Path:
    """
    Get the path to the computed metrics file for a given raw data file. Assumes the metrics file is stored in the same
    directory as the raw data file, with the same name stem and a '_metrics.npz' suffix.

    Parameters
    ----------
    raw_fname : Path
        The path to the raw data file. Must be a TIFF file.

    Returns
    -------
    metrics_path : Path
        The path to the computed metrics file.
    """
    return raw_fname.with_stem(raw_fname.stem + '_metrics').with_suffix('.npz')


def get_metrics_paths_from_df(df: pd.DataFrame) -> list[Path]:
    """
    Get the paths to the computed metrics files for each row in a DataFrame. Assumes the metrics files are stored in the
    same directory as the raw data files, with the same name stem and a '_metrics.npz' suffix. Only returns paths for
    rows where the algorithm is 'mcorr'.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing information about each batch of image data.
        Must be compatible with the mesmerize-core DataFrame API to call `get_params_diffs` and `get_output` on each row.

    Returns
    -------
    metrics_paths : list of Path
        List of file paths where metrics are stored for each batch.
    """
    return [get_metrics_path(Path(row.caiman.get_input_movie_path())) for i, row in df.iterrows() if
            row.algo == 'mcorr']


def compute_batch_metrics(df: pd.DataFrame = None, raw_filename=None, overwrite: bool = False) -> List:
    """
    Compute and store various statistical metrics for each batch of image data.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing information about each batch of image data.
        Must be compatible with the mesmerize-core DataFrame API to call
        `get_params_diffs` and `get_output` on each row.
    raw_filename : Path, optional
        The path to the raw data file. Must be a TIFF file. Default is None.

    Returns
    -------
    metrics_list : list of dict
        List of dictionaries, where each dictionary contains statistical metrics for a batch.
    subplot_paths : list of Path
        List of file paths where metrics are stored for each batch.
    subplot_names : list of str
        List of descriptive names for each subplot.
    """
    if raw_filename is not None:
        raw_filename = Path(raw_filename)
    else:
        try:
            import mesmerize_core as mc
            raw_filename = mc.get_parent_raw_data_path() / df.iloc[0].input_movie_path
        except Exception as e:
            print('Skipping raw data metrics computation. Could not find raw data file.')
            raw_filename = None

    if raw_filename is not None:
        if not raw_filename.exists():
            raise FileNotFoundError(f"Raw data file {raw_filename} not found.")
        raw_metrics_path = get_metrics_path(raw_filename)
        if raw_metrics_path.exists() and not overwrite:
            raw_metrics_path.unlink()
        start = time.time()
        raw_metrics_path = _compute_metrics_with_temp_file(raw_filename)
        print(f'Computed metrics for raw data in {time.time() - start:.2f} seconds.')
        metrics_paths = [raw_metrics_path]
    else:
        metrics_paths = []
    if df is not None:
        for i, row in df.iterrows():
            print(f'Computing metrics for batch index {i}...')
            start = time.time()

            if row.algo != 'mcorr':
                continue

            data = df.iloc[i].mcorr.get_output()
            final_size = data.shape[1:]

            # Pre-fetch metrics path and check if it exists
            metrics_path = get_metrics_path(df.iloc[i].mcorr.get_output_path())
            if metrics_path.exists():
                if overwrite:
                    print(f"Overwriting metrics file {metrics_path}.")
                    metrics_path.unlink(missing_ok=True)
                else:
                    print(f"Metrics file {metrics_path} already exists. Skipping. To overwrite, set `overwrite=True`.")
                    continue

            _ = compute_metrics_motion_correction(df.iloc[i].mcorr.get_output_path(), final_size[0], final_size[1],
                                                  swap_dim=False, gSig_filt=None)
            print(f'Computed metrics for batch index {i} in {time.time() - start:.2f} seconds')
            metrics_paths.append(metrics_path)
            print(f'Metrics computed in {time.time() - start:.2f} seconds')
    return metrics_paths


def create_summary_df(batch_df):
    # filter any non 'mcorr' items and outputs that are
    batch_df = batch_df[batch_df.item_name == 'mcorr']
    # check all df raw_data_paths are the same input file
    assert batch_df.input_movie_path.nunique() == 1, "All input files must be the same"

    raw_filename = batch_df.iloc[0].input_movie_path
    raw_filepath = mc.get_parent_raw_data_path() / raw_filename
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
        'uuid': 'None'
    }
    metrics_list = [met]
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
    return pd.DataFrame(metrics_list)


def create_metrics_df(metrics_p: list[str]) -> pd.DataFrame:
    metrics_list = []
    for i, file in enumerate(metrics_p):
        with np.load(file) as f:
            corr = f['correlations']
            norms = f['norms']
            smoothness = f['smoothness']
        metrics_list.append({
            'correlations': np.mean(corr),
            'norms': np.mean(norms),
            'smoothness': float(smoothness),
        })
    return pd.DataFrame(metrics_list)


def add_param_diffs(summary_df, metrics_df, param_diffs):
    final_df = pd.concat([summary_df, metrics_df], axis=1)

    for col in param_diffs.columns:
        final_df[col] = "None"
    for i, row in final_df.iterrows():
        if row.batch_index == "None":
            continue
        batch_index = int(row.batch_index)
        param_diff = param_diffs.iloc[batch_index]
        for col in param_diffs.columns:
            final_df.at[i, col] = param_diff[col]
    return final_df
