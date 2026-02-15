import matplotlib.pyplot as plt
import numpy as np
from typing import Any as ArrayLike


def _get_30p_order():
    return (np.array([
        1, 5, 6, 7, 8, 9, 2, 10, 11, 12, 13, 14, 15, 16, 17, 3, 18, 19, 20, 21, 22, 23, 4, 24, 25, 26, 27, 28, 29, 30
    ]) - 1)


def extract_center_square(images, size):
    """
    Extract a square crop from the center of the input images.

    Parameters
    ----------
    images : numpy.ndarray
        Input array. Can be 2D (H x W) or 3D (T x H x W), where:
        - H is the height of the image(s).
        - W is the width of the image(s).
        - T is the number of frames (if 3D).
    size : int
        The size of the square crop. The output will have dimensions
        (size x size) for 2D inputs or (T x size x size) for 3D inputs.

    Returns
    -------
    numpy.ndarray
        A square crop from the center of the input images. The returned array
        will have dimensions:
        - (size x size) if the input is 2D.
        - (T x size x size) if the input is 3D.

    Raises
    ------
    ValueError
        If `images` is not a NumPy array.
        If `images` is not 2D or 3D.
        If the specified `size` is larger than the height or width of the input images.

    Notes
    -----
    - For 2D arrays, the function extracts a square crop directly from the center.
    - For 3D arrays, the crop is applied uniformly across all frames (T).
    - If the input dimensions are smaller than the requested `size`, an error will be raised.

    Examples
    --------
    Extract a center square from a 2D image:

    >>> import numpy as np
    >>> image = np.random.rand(600, 576)
    >>> cropped = extract_center_square(image, size=200)
    >>> cropped.shape
    (200, 200)

    Extract a center square from a 3D stack of images:

    >>> stack = np.random.rand(100, 600, 576)
    >>> cropped_stack = extract_center_square(stack, size=200)
    >>> cropped_stack.shape
    (100, 200, 200)
    """
    if not isinstance(images, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    if images.ndim == 2:  # 2D array (H x W)
        height, width = images.shape
        center_h, center_w = height // 2, width // 2
        half_size = size // 2
        return images[center_h - half_size:center_h + half_size,
               center_w - half_size:center_w + half_size]

    elif images.ndim == 3:  # 3D array (T x H x W)
        T, height, width = images.shape
        center_h, center_w = height // 2, width // 2
        half_size = size // 2
        return images[:,
               center_h - half_size:center_h + half_size,
               center_w - half_size:center_w + half_size]
    else:
        raise ValueError("Input array must be 2D or 3D.")

def _get_30p_order():
    return (np.array([
        1, 5, 6, 7, 8, 9, 2, 10, 11, 12, 13, 14, 15, 16, 17, 3, 18, 19, 20, 21, 22, 23, 4, 24, 25, 26, 27, 28, 29, 30
    ]) - 1)


def extract_center_square(images, size):
    """
    Extract a square crop from the center of the input images.

    Parameters
    ----------
    images : numpy.ndarray
        Input array. Can be 2D (H x W) or 3D (T x H x W), where:
        - H is the height of the image(s).
        - W is the width of the image(s).
        - T is the number of frames (if 3D).
    size : int
        The size of the square crop. The output will have dimensions
        (size x size) for 2D inputs or (T x size x size) for 3D inputs.

    Returns
    -------
    numpy.ndarray
        A square crop from the center of the input images. The returned array
        will have dimensions:
        - (size x size) if the input is 2D.
        - (T x size x size) if the input is 3D.

    Raises
    ------
    ValueError
        If `images` is not a NumPy array.
        If `images` is not 2D or 3D.
        If the specified `size` is larger than the height or width of the input images.

    Notes
    -----
    - For 2D arrays, the function extracts a square crop directly from the center.
    - For 3D arrays, the crop is applied uniformly across all frames (T).
    - If the input dimensions are smaller than the requested `size`, an error will be raised.

    Examples
    --------
    Extract a center square from a 2D image:

    >>> import numpy as np
    >>> image = np.random.rand(600, 576)
    >>> cropped = extract_center_square(image, size=200)
    >>> cropped.shape
    (200, 200)

    Extract a center square from a 3D stack of images:

    >>> stack = np.random.rand(100, 600, 576)
    >>> cropped_stack = extract_center_square(stack, size=200)
    >>> cropped_stack.shape
    (100, 200, 200)
    """
    if not isinstance(images, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    if images.ndim == 2:  # 2D array (H x W)
        height, width = images.shape
        center_h, center_w = height // 2, width // 2
        half_size = size // 2
        return images[center_h - half_size:center_h + half_size,
               center_w - half_size:center_w + half_size]

    elif images.ndim == 3:  # 3D array (T x H x W)
        T, height, width = images.shape
        center_h, center_w = height // 2, width // 2
        half_size = size // 2
        return images[:,
               center_h - half_size:center_h + half_size,
               center_w - half_size:center_w + half_size]
    else:
        raise ValueError("Input array must be 2D or 3D.")


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
