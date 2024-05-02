import numpy as np
from scipy import ndimage


def lcn(image, sigmas=(12, 12)):
    """
    Local contrast normalization using Gaussian filters.

    Normalize each pixel using mean and standard deviation computed on a local neighborhood
    defined by Gaussian filters. This approach is softer on edges compared to using uniform filters.

    Parameters
    ----------
    image : np.array
        2D array with raw two-photon images.
    sigmas : tuple of int
        Gaussian kernel standard deviations for x and y dimensions. Smaller values result in more
        localized neighborhoods. Typical values might be between 15-30 microns.

    Returns
    -------
    np.array
        Normalized image.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(256, 256)
    >>> normalized_image = lcn(image, sigmas=(15, 15))

    """
    local_mean = ndimage.gaussian_filter(image, sigmas)
    local_var = ndimage.gaussian_filter(image ** 2, sigmas) - local_mean ** 2
    local_std = np.sqrt(np.clip(local_var, a_min=0, a_max=None))
    norm = (image - local_mean) / (local_std + 1e-7)

    return norm


def sharpen_2pimage(image, laplace_sigma=0.7, low_percentile=3, high_percentile=99.9):
    """
    Enhance edges in an image using Laplacian of Gaussian.

    Apply a Laplacian of Gaussian filter to enhance edges, then clip the pixel values to the specified
    percentiles and normalize the image.

    Parameters
    ----------
    image : np.array
        2D array with raw two-photon images.
    laplace_sigma : float
        Sigma of the Gaussian used in the Laplace filter.
    low_percentile : float
        Lower bound percentile to clip pixel values.
    high_percentile : float
        Upper bound percentile to clip pixel values.

    Returns
    -------
    np.array
        Sharpened and normalized image.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(256, 256)
    >>> sharpened_image = sharpen_2pimage(image, laplace_sigma=0.7, low_percentile=3, high_percentile=99.9)

    """
    sharpened = image - ndimage.gaussian_laplace(image, laplace_sigma)
    clipped = np.clip(sharpened, *np.percentile(sharpened, [low_percentile, high_percentile]))
    norm = (clipped - clipped.mean()) / (clipped.max() - clipped.min() + 1e-7)
    return norm


def create_correlation_image(scan):
    """
    Compute the correlation image for a given scan.

    For each pixel, compute the correlation over time with each of its eight neighboring pixels and
    average these correlations to produce a correlation image.

    Parameters
    ----------
    scan : np.array
        3D scan array (height, width, num_frames).

    Returns
    -------
    np.array
        2D correlation image (height x width).

    Notes
    -----
    This implementation computes correlations directly rather than reusing computations between pixels,
    which might seem less efficient but allows better use of vectorization, hence can be faster and
    slightly more memory efficient than a dynamic programming approach.

    Examples
    --------
    >>> import numpy as np
    >>> scan = np.random.rand(100, 100, 10)  # 100x100 image with 10 time points
    >>> corr_img = create_correlation_image(scan)

    """
    from itertools import product

     # Get image dimensions
    image_height, image_width, num_frames = scan.shape

    # Compute deviations from the mean (in place)
    mean_image = np.mean(scan, axis=-1, keepdims=True)
    scan -= mean_image # in place
    deviations = scan

    # Calculate (unnormalized) standard deviation per pixel
    stddev_image = np.empty([image_height, image_width])
    for y, x in product(range(image_height), range(image_width)):
        stddev_image[y, x] = np.sum(deviations[y, x] ** 2)
    stddev_image = np.sqrt(stddev_image)
    # we don't use np.sum(deviations**2, axis=-1) because it creates a copy of the scan

    # Cut a 3 x 3 square around each pixel and compute their (mean) pair-wise correlation.
    correlation_image = np.empty([image_height, image_width])
    for y, x in product(range(image_height), range(image_width)):
            yslice = slice(max(y - 1, 0), min(y + 2, image_height))
            xslice = slice(max(x - 1, 0), min(x + 2, image_width))

            numerator = np.inner(deviations[yslice, xslice], deviations[y, x])
            correlations = numerator / stddev_image[yslice, xslice]
            correlations[min(1, y), min(1, x)] = 0
            correlation_image[y, x] = np.sum(correlations) /  (correlations.size - 1)
    correlation_image /= stddev_image

    # Return scan back to original values
    scan += mean_image

    return correlation_image