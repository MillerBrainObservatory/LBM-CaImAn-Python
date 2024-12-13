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
from typing import Any as ArrayLike

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


def _compute_metrics_with_temp_file(raw_fname: Path) -> Path:
    """
    Wrapper for caiman.motion_correction.compute_metrics_motion_correction. Writes raw_file to a temporary memmapped file to
    run compute_metrics_motion_correction, and move the metrics file back to the raw_fname directory.

    Needed due to compute_metrics_motion_correction not accepting memmapped files, just filenames.

    Parameters
    ----------
    raw_fname : Path
        The path to the raw data file. Must be a TIFF file.

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
    # Load the data
    data = tifffile.memmap(raw_fname)

    with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        tifffile.imwrite(temp_path, data)
        _ = compute_metrics_motion_correction(temp_path, data.shape[1], data.shape[2], swap_dim=False)

        temp_metrics_path = get_metrics_path(temp_path)
        # temp_metrics_path = temp_path.with_stem(temp_path.stem + '_metrics').with_suffix('.npz')
        final_metrics_path = get_metrics_path(raw_fname)
        # final_metrics_path = raw_fname.with_stem(raw_fname.stem + '_metrics').with_suffix('.npz')

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


def compute_batch_metrics(df: pd.DataFrame, raw_filename=None):
    """
    Compute and store various statistical metrics for each batch of image data.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing information about each batch of image data.
        Must be compatible with the mesmerize-core DataFrame API to call `get_params_diffs` and `get_output` on each row.
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
    raw_filename = Path(raw_filename)
    if raw_filename is not None:
        if not raw_filename.exists():
            raise FileNotFoundError(f"Raw data file {raw_filename} not found.")
        data = tifffile.memmap(raw_filename)
        flat_data = data.ravel()
        raw_metrics_path = _compute_metrics_with_temp_file(raw_filename)

        with np.load(raw_metrics_path) as ld:
            mean_norm = np.mean(ld['norms'])
            mean_corr = np.mean(ld['correlations'])

        metrics = {
            'batch_index': "None (Raw Data)",
            'min': np.min(flat_data),
            'max': np.max(flat_data),
            'mean': np.mean(flat_data),
            'median': np.median(flat_data),
            'std': np.std(flat_data),
            'p1': np.percentile(flat_data, 1),
            'p99': np.percentile(flat_data, 99),
            'mean_corr': mean_corr,
            'mean_norm': mean_norm,
            'uuid': "None",
        }

        metrics_list = [metrics]
        subplot_paths = [raw_metrics_path]
    else:
        metrics_list = []
        subplot_paths = []

    for i, row in df.iterrows():
        start = time.time()

        if row.algo != 'mcorr':
            continue

        data = df.iloc[i].mcorr.get_output()
        final_size = data.shape[1:]
        flat_data = data.ravel()

        mmap_path = Path(df.iloc[i].mcorr.get_output_path())
        metrics_path = mmap_path.with_name(f"{mmap_path.stem}_metrics.npz")
        metrics_path.unlink(missing_ok=True)
        gSig_filt = df.iloc[0].params['main']['gSig_filt']

        _, correlations, flows, norms, _ = compute_metrics_motion_correction(mmap_path, final_size[0], final_size[1],
                                                                             swap_dim=False, gSig_filt=gSig_filt)
        metrics_list.append({
            'batch_index': i,
            'min': np.min(flat_data),
            'max': np.max(flat_data),
            'mean': np.mean(flat_data),
            'median': np.median(flat_data),
            'std': np.std(flat_data),
            'p1': np.percentile(flat_data, 1),
            'p99': np.percentile(flat_data, 99),
            'mean_corr': np.mean(correlations),
            'mean_norm': np.mean(norms),
            'uuid': df.iloc[i].uuid,
        })
        subplot_paths.append(metrics_path)
        print(f'Metrics computed in {time.time() - start:.2f} seconds')
    return metrics_list, subplot_paths
