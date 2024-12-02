import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

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
    import matplotlib.patches as patches

    scale_bar_sizes = [5, 10, 20]
    
    # Calculate the size of scale bars in pixels
    scale_bar_lengths = [int(size / pixel_resolution) for size in scale_bar_sizes]

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, scale_length, size in zip(axes, scale_bar_lengths, scale_bar_sizes):
        ax.imshow(image, cmap='gray')

        # Add a scale bar (bottom-left corner of the image)
        bar_x = 10  # Horizontal position of the bar (pixels)
        bar_y = image.shape[0] - 20  # Vertical position of the bar (pixels)
        ax.add_patch(patches.Rectangle((bar_x, bar_y), scale_length, 5, color='white'))

        # Annotate
        ax.text(bar_x + scale_length / 2, bar_y - 10, f'{size} μm', color='white',
                ha='center', va='bottom', fontsize=10)
        ax.axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def generate_patch_view(image: ArrayLike, pixel_resolution: float, target_patch_size: int=40, overlap_fraction: float=0.5):
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
                 f"Overlap: {overlap} pixels (~{overlap * pixel_resolution:.1f} μm)\n"
                 f"Padding: X={pad_x}, Y={pad_y} pixels")
    plt.tight_layout()
    return fig, ax