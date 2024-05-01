import numpy as np
import torch
from torch.nn import functional as F


def create_grid(um_sizes, desired_res=1):
    """
    Create a spatial grid that represents the sample positions for each pixel or voxel in a field of view (FOV),
    given specified physical sizes and desired resolution, centered at (0, 0, 0).

    Parameters
    ----------
    um_sizes : tuple of float
        The sizes in microns for each dimension of the FOV, e.g., (d1, d2, d3) for a 3D stack.
    desired_res : float or tuple of float, optional
        The desired resolution in microns per pixel for each dimension. If a single float is provided,
        it is applied uniformly across all dimensions.

    Returns
    -------
    np.array
        A multi-dimensional array where each element contains the coordinate (x, y, z, ...) for that pixel or voxel.
        The array shape follows the dimensions specified in `um_sizes` with an additional last dimension for the coordinates.
        Note: the coordinate system is centered such that the center of the FOV is at (0, 0, 0).

    Example
    -------
    >>> grid = create_grid((100, 200, 50), (1, 2, 1))
    >>> print(grid.shape)  # Output should be (50, 100, 200, 3) for a 3D grid with 3 coordinates per point

    Notes
    -----
    - This function assumes sampling occurs at the center of each pixel/voxel.
    - The edges of the grid do not align at maximum extents due to the central alignment of samples.

    """
    # Make sure desired_res is a tuple with the same size as um_sizes
    if np.isscalar(desired_res):
        desired_res = (desired_res,) * len(um_sizes)

    # Create grid
    out_sizes = [int(round(um_s / res)) for um_s, res in zip(um_sizes, desired_res)]
    um_grids = [
        np.linspace(-(s - 1) * res / 2, (s - 1) * res / 2, s, dtype=np.float32)
        for s, res in zip(out_sizes, desired_res)
    ]  # *
    full_grid = np.stack(np.meshgrid(*um_grids, indexing="ij")[::-1], axis=-1)
    # * this preserves the desired resolution by slightly changing the size of the FOV to
    # out_sizes rather than um_sizes / desired_res.

    return full_grid


def resize(original, um_sizes, desired_res):
    """
    Resize an array to a specified resolution, maintaining the center of the array and ensuring exact resolution,
    potentially altering the field of view (FOV) slightly to meet these constraints.

    Parameters
    ----------
    original : np.array
        The original array to be resized. Can be of any dimensionality (2D image, 3D volume, etc.).
    um_sizes : tuple of float
        The size in microns of the original array in each dimension.
    desired_res : int or tuple of float
        The desired resolution in microns per pixel for the resized array. If a tuple, it specifies
        the resolution for each dimension.

    Returns
    -------
    np.array
        The resized array with the specified resolution, typically used in imaging applications where
        maintaining exact spatial dimensions and alignment is critical.

    Example
    -------
    >>> original = np.random.rand(256, 256)
    >>> resized = resize(original, (256, 256), 2)
    >>> print(resized.shape)  # Expect approximately (128, 128) depending on exact calculation and rounding

    Notes
    -----
    - The function ensures the center of the original and resized arrays are aligned.
    - Resolution is strictly enforced, which may result in a slight adjustment to the FOV.
    - This implementation uses PyTorch's grid_sample for resampling, ensuring high performance and flexibility.

    """
    import torch.nn.functional as F

    # Create grid to sample in microns
    grid = create_grid(um_sizes, desired_res)  # d x h x w x 3

    # Re-express as a torch grid [-1, 1]
    um_per_px = np.array([um / px for um, px in zip(um_sizes, original.shape)])
    torch_ones = (
        np.array(um_sizes) / 2 - um_per_px / 2
    )  # sample position of last pixel in original
    grid = grid / torch_ones[::-1].astype(np.float32)

    # Resample
    input_tensor = torch.from_numpy(
        original.reshape(1, 1, *original.shape).astype(np.float32)
    )
    grid_tensor = torch.from_numpy(grid.reshape(1, *grid.shape))
    resized_tensor = F.grid_sample(
        input_tensor, grid_tensor, padding_mode="border", align_corners=True
    )
    resized = resized_tensor.numpy().squeeze()

    return resized


def affine_product(X, A, b):
    """
    Perform a special case of an affine transformation that applies a 2D to 3D mapping.

    This function transforms 2D coordinates into 3D coordinates using a given affine matrix
    and translation vector, handling the case where the affine matrix is only partially provided
    and needs to be augmented for 3D transformation.

    Parameters
    ----------
    X : torch.Tensor
        A tensor containing 2D coordinates with shape (d1, d2, 2), where d1 and d2 are spatial
        dimensions, and the last dimension stores (x, y) coordinates.
    A : torch.Tensor
        The first two columns of a 3x3 affine transformation matrix, provided as a 3x2 tensor.
    b : torch.Tensor
        A 3D translation vector provided as a 1D tensor with 3 elements (x, y, z).

    Returns
    -------
    torch.Tensor
        A tensor with shape (d1, d2, 3) representing the 3D coordinates obtained by applying
        the affine transformation to each 2D coordinate in X.

    Example
    -------
    >>> X = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> A = torch.tensor([[1, 0], [0, 1], [1, 1]])
    >>> b = torch.tensor([1, 2, 3])
    >>> transformed = affine_product(X, A, b)
    >>> print(transformed.shape)  # Output: (2, 2, 3)

    Notes
    -----
    - The function extends a typical 2D affine transformation into 3D space by assuming
      zero displacement in the third dimension before applying the translation.

    """

    return torch.einsum("ij,klj->kli", (A.double(), X.double())) + b


def sample_grid(volume, grid):
    """
    Sample values from a 3D volume at specified 3D coordinates using grid sampling.

    This function maps a grid of 3D coordinates to the corresponding values in a 3D volume,
    assuming that the center of the volume is at (0, 0, 0) and that both the grid and volume
    share the same resolution.

    Parameters
    ----------
    volume : torch.Tensor
        A 3D tensor representing the volume data, with dimensions corresponding to depth (d),
        height (h), and width (w).
    grid : torch.Tensor
        A 3D tensor specifying the coordinates to sample from the volume, with dimensions
        (d1, d2, 3) where each element represents (x, y, z) coordinates.

    Returns
    -------
    torch.Tensor
        A 2D tensor with dimensions (d1, d2) that contains the values sampled from `volume` at
        the coordinates specified in `grid`.

    Example
    -------
    >>> volume = torch.randn(10, 10, 10)
    >>> grid = torch.tensor([[[0, 0, 0], [1, 1, 1]], [[-1, -1, -1], [2, 2, 2]]])
    >>> sampled = sample_grid(volume, grid)
    >>> print(sampled.shape)  # Output: (2, 2)

    Notes
    -----
    - Coordinates in `grid` are assumed to be in the same unit as the volume dimensions.
    - This function uses `torch.nn.functional.grid_sample` which expects normalized grid
      coordinates ranging from -1 to 1, so appropriate normalization is applied internally.

    """
    # Make sure input is tensor
    volume = torch.as_tensor(volume, dtype=torch.float32)
    grid = torch.as_tensor(grid, dtype=torch.float32)

    # Rescale grid so it ranges from -1 to 1 (as expected by F.grid_sample)
    norm_factor = torch.as_tensor([s / 2 - 0.5 for s in volume.shape[::-1]])
    norm_grid = grid / norm_factor

    # Resample
    resampled = F.grid_sample(
        volume[None, None, ...],
        norm_grid[None, None, ...],
        padding_mode="zeros",
        align_corners=True,
    )
    resampled = resampled.squeeze()  # drop batch and channel dimension

    return resampled
