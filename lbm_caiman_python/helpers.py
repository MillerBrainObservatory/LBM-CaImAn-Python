import os
import shutil
import sys
import tempfile

import cv2
import scipy
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Any as ArrayLike

import caiman as cm
from tqdm import tqdm

from .lcp_io import get_metrics_path


def get_single_patch_coords(dims, stride, overlap, patch_index):
    """
    Get coordinates of a single patch based on stride, overlap parameters of motion-correction.

    Parameters
    ----------
    dims : tuple
        Dimensions of the image as (rows, cols).
    stride : int
        Number of pixels to include in each patch.
    overlap : int
        Number of pixels to overlap between patches.
    patch_index : tuple
        Index of the patch to return.
    """
    patch_height = stride + overlap
    patch_width = stride + overlap
    rows = np.arange(0, dims[0] - patch_height + 1, stride)
    cols = np.arange(0, dims[1] - patch_width + 1, stride)

    row_idx, col_idx = patch_index
    y_start = rows[row_idx]
    x_start = cols[col_idx]

    return y_start, y_start + patch_height, x_start, x_start + patch_width


def _pad_image_for_even_patches(image, stride, overlap):
    patch_width = stride + overlap
    padded_x = int(np.ceil(image.shape[0] / patch_width) * patch_width) - image.shape[0]
    padded_y = int(np.ceil(image.shape[1] / patch_width) * patch_width) - image.shape[1]
    return np.pad(image, ((0, padded_x), (0, padded_y)), mode='constant'), padded_x, padded_y


def calculate_num_patches(image, stride, overlap):
    """
    Calculate the total number of patches in an image given stride and rf.

    Parameters
    ----------
    image_size : tuple
        Size of the image as (df, cols).
    stride : int
        Half-size of the patches in pixels (patch width is rf*2 + 1).
    overlap : int
        Amount of overlap between patches in pixels.

    Returns
    -------
    int
        Total number of patches.
    """
    from caiman.utils.visualization import get_rectangle_coords

    # pad the image like caiman does
    padded_image, pad_x, pad_y = _pad_image_for_even_patches(image, stride, overlap)

    # Get patch coordinates
    patch_rows, patch_cols = get_rectangle_coords(padded_image.shape, stride, overlap)
    return len(patch_rows) * len(patch_cols)


def calculate_neurons_per_patch(rf, pixel_resolution, neuron_density):
    """
    Calculate the expected number of neurons in a 2D patch.

    Parameters
    ----------
    rf : int
        The receptive field size in pixels.
    pixel_resolution : tuple
        The resolution of the image in microns per pixel.
    neuron_density : float
        The density of neurons in the image in neurons per square micron.
    """
    row_size, col_size = pixel_resolution
    surface_density = neuron_density  # Already in neurons per square micron for a 2D slice

    # Patch size in pixels
    patch_rows = 2 * rf + 1
    patch_cols = 2 * rf + 1

    # Patch area in microns^2
    patch_area = (patch_rows * row_size) * (patch_cols * col_size)

    # Expected neurons per patch
    expected_neurons = patch_area * surface_density

    return expected_neurons


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
