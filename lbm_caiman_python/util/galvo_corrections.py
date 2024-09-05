"""
Galvo Corrections
=================

Utilities for motion and raster correction of resonant scans.

.. currentmodule:: util.galvo_corrections

"""
import numpy as np
from scipy import interpolate as interp
from scipy import ndimage
from scipy import signal
from tqdm import tqdm

from .exceptions import PipelineException
from .signal import mirrconv


def compute_raster_phase(image, temporal_fill_fraction):
    """
    Compute raster correction for bidirectional resonant scanners by estimating the phase shift
    that best aligns even and odd rows of an image. This function assumes that the distortion
    is primarily due to the resonant mirror returning before reaching the nominal turnaround point.

    Parameters
    ----------
    image : np.array
        The image to be corrected. This should be a 2D numpy array where rows correspond to
        successive lines scanned by the resonant mirror.
    temporal_fill_fraction : float
        The fraction of the total line scan period during which the actual image acquisition occurs.
        This is used to calculate the effective scan range in angles. It typically ranges from 0 to 1,
        where 1 means the entire period is used for acquiring image data.

    Returns
    -------
    float
        An estimated phase shift angle (in radians) that indicates the discrepancy between the
        expected and actual initial angles of the bidirectional scan. Positive values suggest that
        even rows should be shifted to the right.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100)  # Simulated image
    >>> temporal_fill_fraction = 0.9
    >>> angle_shift = compute_raster_phase(image, temporal_fill_fraction)
    >>> print(f"Calculated angle shift: {angle_shift} radians")

    Notes
    -----
    - The function uses linear interpolation for shifting the pixel rows and assumes that even rows
      and odd rows should be symmetric around the central scan line.
    - The phase shift is found using a greedy algorithm that iteratively refines the estimate.
    - Artifacts near the edges of the image can significantly affect the accuracy of the phase
      estimation, hence a default 5% of the rows and 10% of the columns are skipped during the
      calculation.
    - This function depends on `numpy` for numerical operations and `scipy.interpolate` for
      interpolation of row data.

    """

    # Make sure image has even number of rows (so number of even and odd rows is the same)
    image = image[:-1] if image.shape[0] % 2 == 1 else image

    # Get some params
    image_height, image_width = image.shape
    skip_rows = round(image_height * 0.05)  # rows near the top or bottom have artifacts
    skip_cols = round(image_width * 0.10)  # so do columns

    # Create images with even and odd rows
    even_rows = image[::2][skip_rows:-skip_rows]
    odd_rows = image[1::2][skip_rows:-skip_rows]

    # Scan angle at which each pixel was recorded.
    max_angle = (np.pi / 2) * temporal_fill_fraction
    scan_angles = np.linspace(-max_angle, max_angle, image_width + 2)[1:-1]
    # sin_index = np.sin(scan_angles)

    # Greedy search for the best raster phase: starts at coarse estimates and refines them
    even_interp = interp.interp1d(scan_angles, even_rows, fill_value="extrapolate")
    odd_interp = interp.interp1d(scan_angles, odd_rows, fill_value="extrapolate")
    angle_shift = 0
    for scale in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        angle_shifts = angle_shift + scale * np.linspace(-9, 9, 19)
        match_values = []
        for new_angle_shift in angle_shifts:
            shifted_evens = even_interp(scan_angles + new_angle_shift)
            shifted_odds = odd_interp(scan_angles - new_angle_shift)
            match_values.append(
                np.sum(
                    shifted_evens[:, skip_cols:-skip_cols]
                    * shifted_odds[:, skip_cols:-skip_cols]
                )
            )
        angle_shift = angle_shifts[np.argmax(match_values)]

    return angle_shift


def compute_motion_shifts(scan, template, in_place=True, num_threads=8):
    """
    Compute subpixel motion correction shifts for a series of images against a template.

    This function calculates the x and y shifts required to align each frame in a given
    scan with a reference template image. It employs a Fourier-based approach to determine
    the translation that best aligns each frame with the template. The shifts are computed
    in a rigid and subpixel precise manner, meaning that the correction does not involve
    rotation or scaling.

    Parameters
    ----------
    scan : np.array
        A 3D numpy array (image_height, image_width, num_frames) representing the stack of
        images to be aligned. If only a single 2D image is provided, it is automatically
        converted into a 3D array with one frame.
    template : np.array
        A 2D numpy array (image_height, image_width) that serves as the reference image
        to which all frames in the scan are aligned.
    in_place : bool, optional
        A flag that determines whether the input 'scan' array can be overwritten during
        the computation. Setting this to True can reduce memory usage but will alter the
        input data. Default is True.
    num_threads : int, optional
        The number of threads to use for FFT calculations, which can improve performance
        on multicore systems. Default is 8.

    Returns
    -------
    tuple of np.array
        A tuple containing two arrays, (y_shifts, x_shifts), each of length num_frames.
        'y_shifts' and 'x_shifts' contain the vertical and horizontal shifts (in pixels),
        respectively. Positive values indicate a shift to the right or downward, while
        negative values indicate a shift to the left or upward.

    Notes
    -----
    - This function is based on the `imreg_dft.translation` function from the `imreg_dft` module,
      which is used for image registration using discrete Fourier transforms.
    - Edge artifacts are mitigated by applying a Tukey window (taper) to both the template
      and the frames before computing the Fourier transforms.
    - The shifts are calculated to subpixel accuracy by interpolating the peak of the cross
      power spectrum around its maximum.

    Example
    -------
    >>> import numpy as np
    >>> scan = np.random.rand(256, 256, 10)  # Simulated stack of 10 images
    >>> template = np.random.rand(256, 256)  # Simulated template image
    >>> y_shifts, x_shifts = compute_motion_shifts(scan, template, in_place=False, num_threads=4)
    >>> print(y_shifts, x_shifts)

    """
    import pyfftw
    from imreg_dft import utils

    # Add third dimension if scan is a single image
    if scan.ndim == 2:
        scan = np.expand_dims(scan, -1)

    # Get some params
    image_height, image_width, num_frames = scan.shape
    taper = np.outer(signal.tukey(image_height, 0.2), signal.tukey(image_width, 0.2))

    # Prepare fftw
    frame = pyfftw.empty_aligned((image_height, image_width), dtype="complex64")
    fft = pyfftw.builders.fft2(
        frame, threads=num_threads, overwrite_input=in_place, avoid_copy=True
    )
    ifft = pyfftw.builders.ifft2(
        frame, threads=num_threads, overwrite_input=in_place, avoid_copy=True
    )

    # Get fourier transform of template
    template_freq = fft(template * taper).conj()  # we only need the conjugate
    abs_template_freq = abs(template_freq)
    eps = abs_template_freq.max() * 1e-15

    # Compute subpixel shifts per image
    y_shifts = np.empty(num_frames)
    x_shifts = np.empty(num_frames)
    for i in range(num_frames):
        # Compute correlation via cross power spectrum
        image_freq = fft(scan[:, :, i] * taper)
        cross_power = (image_freq * template_freq) / (
            abs(image_freq) * abs_template_freq + eps
        )
        shifted_cross_power = np.fft.fftshift(abs(ifft(cross_power)))

        # Get best shift
        shifts = np.unravel_index(
            np.argmax(shifted_cross_power), shifted_cross_power.shape
        )
        shifts = utils._interpolate(shifted_cross_power, shifts, rad=3)

        # Map back to deviations from center
        y_shifts[i] = shifts[0] - image_height // 2
        x_shifts[i] = shifts[1] - image_width // 2

    return y_shifts, x_shifts


def fix_outliers(y_shifts, x_shifts, max_y_shift=20, max_x_shift=20, method="median"):
    """Look for spikes in motion shifts and set them to a sensible value.

    Reject any shift whose y or x shift is higher than max_y_shift/max_x_shift pixels
    from the median/linear estimate/moving average. Outliers filled by interpolating
    valid points; in the edges filled with the median/linear estimate/moving average.

    Parametrs
    ---------
    :param np.array y_shifts/x_shifts: Shifts in y, x.
    :param float max_y_shift/max_x_shifts: Number of pixels used as threshold to classify
        a point as an outlier in y, x.
    :param string method: One of 'mean' or 'trend'.
        'median': Detect outliers as deviations from the median of the shifts.
        'linear': Detect outliers as deviations from a line estimated from the shifts.
        'trend': Detect outliers as deviations from the shift trend computed as a moving
            average over the entire scan.

    :returns: (y_shifts, x_shifts) Two arrays (num_frames) with the fixed motion shifts.
    :returns: (outliers) A boolean array (num_frames) with True for outlier frames.
    """
    # Basic checks
    num_frames = len(y_shifts)
    if num_frames < 5:
        return y_shifts, x_shifts, np.full(num_frames, False)

    # Copy shifts to avoid changing originals
    y_shifts, x_shifts = y_shifts.copy(), x_shifts.copy()

    # Detrend shifts
    if method == "median":
        y_trend = np.median(y_shifts)
        x_trend = np.median(x_shifts)
    elif method == "linear":
        x_trend = _fit_robust_line(x_shifts)
        y_trend = _fit_robust_line(y_shifts)
    else:  # trend
        window_size = min(101, num_frames)
        window_size -= 1 if window_size % 2 == 0 else 0
        y_trend = mirrconv(y_shifts, np.ones(window_size) / window_size)
        x_trend = mirrconv(x_shifts, np.ones(window_size) / window_size)

    # Subtract trend from shifts
    y_shifts -= y_trend
    x_shifts -= x_trend

    # Get outliers
    outliers = np.logical_or(abs(y_shifts) > max_y_shift, abs(x_shifts) > max_x_shift)

    # Interpolate outliers
    num_outliers = np.sum(outliers)
    if (
        num_outliers < num_frames - 1
    ):  # at least two good points needed for interpolation
        # indices = np.arange(len(x_shifts))
        # y_shifts = np.interp(indices, indices[~outliers], y_shifts[~outliers], left=0, right=0)
        # x_shifts = np.interp(indices, indices[~outliers], x_shifts[~outliers], left=0, right=0)
        y_shifts[outliers] = 0
        x_shifts[outliers] = 0
    else:
        print(
            "Warning: {} out of {} frames were outliers.".format(
                num_outliers, num_frames
            )
        )
        y_shifts = 0
        x_shifts = 0

    # Add trend back to shifts
    y_shifts += y_trend
    x_shifts += x_trend

    return y_shifts, x_shifts, outliers


def _fit_robust_line(shifts):
    """
    Use a robust linear regression algorithm to fit a line to the data.

    Parameters
    ----------
    shifts

    """
    from sklearn.linear_model import TheilSenRegressor

    X = np.arange(len(shifts)).reshape(-1, 1)
    y = shifts
    model = TheilSenRegressor()  # robust regression
    model.fit(X, y)
    line = model.predict(X)

    return line


def correct_raster(scan, raster_phase, temporal_fill_fraction, in_place=True):
    """
    Perform raster correction for resonant scans by adjusting even and odd lines based
    on a specified phase shift. This function is designed to correct geometric distortions
    in multi-photon images caused by the scanning mechanism's characteristics.

    Parameters
    ----------
    scan : np.array
        A numpy array representing the scan data, where the first two dimensions correspond to
        image height and width, respectively. The array can have additional dimensions,
        typically representing different frames or slices.
    raster_phase : float
        The phase shift angle (in radians) to be applied for correction. Positive values shift
        even lines to the left and odd lines to the right, while negative values shift even lines
        to the right and odd lines to the left.
    temporal_fill_fraction : float
        The ratio of the actual imaging time to the total time of one scan line. This parameter
        helps in determining the effective scan range that needs correction.
    in_place : bool, optional
        If True (default), modifies the input `scan` array in-place. If False, a corrected copy
        of the scan is returned, preserving the original data.

    Returns
    -------
    np.array
        The raster-corrected scan. The return type matches the input `scan` data type if it's a
        subtype of np.float. Otherwise, it is converted to np.float32 for processing.

    Raises
    ------
    PipelineException
        If input validations fail such as non-matching dimensions, incorrect data types, or if
        the `scan` does not have at least two dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> scan = np.random.rand(256, 256, 30)  # Simulate a 3D scan volume
    >>> raster_phase = 0.01  # Small phase shift
    >>> temporal_fill_fraction = 0.9
    >>> corrected_scan = correct_raster(scan, raster_phase, temporal_fill_fraction, in_place=False)
    >>> print(corrected_scan.shape)
    (256, 256, 30)

    Notes
    -----
    The raster correction is essential for improving the accuracy of image analyses and
    quantification in studies involving resonant scanning microscopy. Adjusting the phase
    shift accurately according to the resonant mirror's characteristics can significantly
    enhance the image quality.

    """

    # Basic checks
    if not hasattr(scan, 'shape'):
        raise PipelineException("Scan needs to be np.array-like")
    if scan.ndim < 2:
        raise PipelineException("Scan with less than 2 dimensions.")

    # Assert scan is float
    if not np.issubdtype(scan.dtype, np.floating):
        print("Warning: Changing scan type from", str(scan.dtype), "to np.float32")
        scan = scan.astype(np.float32, copy=(not in_place))
    elif not in_place:
        scan = scan.copy()  # copy it anyway preserving the original float dtype

    # Get some dimensions
    original_shape = scan.shape
    image_height = original_shape[0]
    image_width = original_shape[1]

    # Scan angle at which each pixel was recorded.
    max_angle = (np.pi / 2) * temporal_fill_fraction
    scan_angles = np.linspace(-max_angle, max_angle, image_width + 2)[1:-1]

    # We iterate over every image in the scan (first 2 dimensions). Same correction
    # regardless of what channel, slice or frame they belong to.
    reshaped_scan = np.reshape(scan, (image_height, image_width, -1))
    num_images = reshaped_scan.shape[-1]
    for i in range(num_images):
        # Get current image
        image = reshaped_scan[:, :, i]

        # Correct even rows of the image (0, 2, ...)
        interp_function = interp.interp1d(
            scan_angles,
            image[::2, :],
            bounds_error=False,
            fill_value=0,
            copy=(not in_place),
        )
        reshaped_scan[::2, :, i] = interp_function(scan_angles + raster_phase)

        # Correct odd rows of the image (1, 3, ...)
        interp_function = interp.interp1d(
            scan_angles,
            image[1::2, :],
            bounds_error=False,
            fill_value=0,
            copy=(not in_place),
        )
        reshaped_scan[1::2, :, i] = interp_function(scan_angles - raster_phase)

    scan = np.reshape(reshaped_scan, original_shape)
    return scan


def correct_motion(scan, x_shifts, y_shifts, in_place=True):
    """
    Apply motion correction to a multi-photon imaging dataset by shifting each frame according to specified x and y shifts.

    This function adjusts each image in a given scan volume to compensate for motion artifacts. Each image is shifted
    according to provided x and y motion shifts (left for x, up for y). It can optionally modify the array in-place to
    save memory.

    Parameters
    ----------
    scan : np.array
        A 3D or higher-dimensional numpy array representing the imaging scan data.
        The first two dimensions are treated as image height and width, respectively.
    x_shifts : list or np.array
        A 1D array or list containing the x-axis shift for each image in the scan.
        Positive values shift the image to the left.
    y_shifts : list or np.array
        A 1D array or list containing the y-axis shift for each image in the scan.
        Positive values shift the image upward.
    in_place : bool, optional
        If True (default), modifies the input array `scan` in-place, saving memory.
        If False, a copy of `scan` is created and modified, preserving the original data.

    Returns
    -------
    np.array
        The motion-corrected scan volume, which is the same numpy array if `in_place` is True,
        or a new array if `in_place` is False. The data type is preserved if it is a subtype of np.float,
        otherwise it is converted to np.float32 for precision.

    Raises
    ------
    PipelineException
        If input validations fail (e.g., non-matching dimensions between the scan and shift arrays,
        incorrect data types, or inadequate array dimensions).

    Example
    -------
    >>> import numpy as np
    >>> scan = np.random.rand(256, 256, 100)  # Simulate a 3D scan volume
    >>> x_shifts = np.random.randint(-5, 5, size=100)
    >>> y_shifts = np.random.randint(-5, 5, size=100)
    >>> corrected_scan = correct_motion(scan, x_shifts, y_shifts, in_place=False)
    >>> print(corrected_scan.shape)
    (256, 256, 100)

    Notes
    -----
    The function is optimized for handling large data volumes typically encountered in imaging applications.
    Using in-place modification helps in managing memory usage effectively, especially with large datasets.
    However, if the original data must be preserved, set `in_place` to False.

    """

    # Basic checks
    if not isinstance(scan, np.ndarray):
        raise PipelineException("Scan needs to be a numpy array.")
    if scan.ndim < 2:
        raise PipelineException("Scan with less than 2 dimensions.")
    if np.ndim(y_shifts) != 1 or np.ndim(x_shifts) != 1:
        raise PipelineException(
            "Dimension of one or both motion arrays differs from 1."
        )
    if len(x_shifts) != len(y_shifts):
        raise PipelineException("Length of motion arrays differ.")

    # Assert scan is float (integer precision is not good enough)
    if not np.issubdtype(scan.dtype, np.floating):
        print("Warning: Changing scan type from", str(scan.dtype), "to np.float32")
        scan = scan.astype(np.float32, copy=(not in_place))
    elif not in_place:
        scan = scan.copy()  # copy it anyway preserving the original dtype

    # Get some dimensions
    original_shape = scan.shape
    image_height = original_shape[0]
    image_width = original_shape[1]

    # Reshape input (to deal with more than 2-D volumes)
    reshaped_scan = np.reshape(scan, (image_height, image_width, -1))
    if reshaped_scan.shape[-1] != len(x_shifts):
        raise PipelineException("Scan and motion arrays have different dimensions")

    # Ignore NaN values (present in some older data)
    y_clean, x_clean = y_shifts.copy(), x_shifts.copy()
    y_clean[np.logical_or(np.isnan(y_shifts), np.isnan(x_shifts))] = 0
    x_clean[np.logical_or(np.isnan(y_shifts), np.isnan(x_shifts))] = 0

    # Shift each frame
    for i, (y_shift, x_shift) in enumerate(zip(y_clean, x_clean)):
        image = reshaped_scan[:, :, i].copy()
        ndimage.interpolation.shift(
            image, (-y_shift, -x_shift), order=1, output=reshaped_scan[:, :, i]
        )

    scan = np.reshape(reshaped_scan, original_shape)
    return scan


def low_memory_motion_correction(scan, raster_phase, fill_fraction, x_shifts, y_shifts):
    """
    Runs an in memory version of our current motion correction found in
    pipeline.utils.galvo_correction. This uses far less memory than the
    parallel motion correction used in motion_correction_method=1.
    """

    chunk_size_in_GB = 1
    single_frame_size = scan[:, :, 0].nbytes
    chunk_size = int(chunk_size_in_GB * 1024**3 / (single_frame_size))

    start_indices = np.arange(0, scan.shape[-1], chunk_size)
    if start_indices[-1] != scan.shape[-1]:
        start_indices = np.insert(start_indices, len(start_indices), scan.shape[-1])

    for start_idx, end_idx in tqdm(
        zip(start_indices[:-1], start_indices[1:]), total=len(start_indices) - 1
    ):
        scan_fragment = scan[:, :, start_idx:end_idx]
        if abs(raster_phase) > 1e-7:
            scan_fragment = correct_raster(
                scan_fragment, raster_phase, fill_fraction
            )  # raster
        scan_fragment = correct_motion(
            scan_fragment, x_shifts[start_idx:end_idx], y_shifts[start_idx:end_idx]
        )  # motion
        scan[:, :, start_idx:end_idx] = scan_fragment

    return scan


def return_scan_offset(image_in, dim):
    """
    Compute the scan offset correction between interleaved lines or columns in an image.

    This function calculates the scan offset correction by analyzing the cross-correlation
    between interleaved lines or columns of the input image. The cross-correlation peak
    determines the amount of offset between the lines or columns, which is then used to
    correct for any misalignment in the imaging process.

    Parameters
    ----------
    image_in : ndarray | ndarray-like
        Input image or volume. It can be 2D, 3D, or 4D. 
    
    .. note:: 
        
        Dimensions: [height, width], [height, width, time], or [height, width, time, channel/plane].
        The input array must be castable to numpy. e.g. np.shape, np.ravel.

    dim : int
        Dimension along which to compute the scan offset correction. 1 for vertical (along height), 2 for horizontal (along width).

    Returns
    -------
    int
        The computed correction value, based on the peak of the cross-correlation.

    Examples
    --------
    >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> return_scan_offset(img, 1)

    Notes
    -----
    This function assumes that the input image contains interleaved lines or columns that
    need to be analyzed for misalignment. The cross-correlation method is sensitive to
    the similarity in pattern between the interleaved lines or columns. Hence, a strong
    and clear peak in the cross-correlation result indicates a good alignment, and the
    corresponding lag value indicates the amount of misalignment.
    """

    if len(image_in.shape) == 3:
        image_in = np.mean(image_in, axis=2)
    elif len(image_in.shape) == 4:
        image_in = np.mean(np.mean(image_in, axis=3), axis=2)

    n = 8

    Iv1 = None
    Iv2 = None
    if dim == 1:
        Iv1 = image_in[::2, :]
        Iv2 = image_in[1::2, :]

        min_len = min(Iv1.shape[0], Iv2.shape[0])
        Iv1 = Iv1[:min_len, :]
        Iv2 = Iv2[:min_len, :]

        buffers = np.zeros((Iv1.shape[0], n))

        Iv1 = np.hstack((buffers, Iv1, buffers))
        Iv2 = np.hstack((buffers, Iv2, buffers))

        Iv1 = Iv1.T.ravel(order='F')
        Iv2 = Iv2.T.ravel(order='F')

    elif dim == 2:
        Iv1 = image_in[:, ::2]
        Iv2 = image_in[:, 1::2]

        min_len = min(Iv1.shape[1], Iv2.shape[1])
        Iv1 = Iv1[:, :min_len]
        Iv2 = Iv2[:, :min_len]

        buffers = np.zeros((n, Iv1.shape[1]))

        Iv1 = np.vstack((buffers, Iv1, buffers))
        Iv2 = np.vstack((buffers, Iv2, buffers))

        Iv1 = Iv1.ravel()
        Iv2 = Iv2.ravel()

    # Zero-center and clip negative values to zero
    Iv1 = Iv1 - np.mean(Iv1)
    Iv1[Iv1 < 0] = 0

    Iv2 = Iv2 - np.mean(Iv2)
    Iv2[Iv2 < 0] = 0

    Iv1 = Iv1[:, np.newaxis]
    Iv2 = Iv2[:, np.newaxis]

    r_full = signal.correlate(Iv1[:, 0], Iv2[:, 0], mode='full', method='auto')
    unbiased_scale = len(Iv1) - np.abs(np.arange(-len(Iv1) + 1, len(Iv1)))
    r = r_full / unbiased_scale

    mid_point = len(r) // 2
    lower_bound = mid_point - n
    upper_bound = mid_point + n + 1
    r = r[lower_bound:upper_bound]
    lags = np.arange(-n, n + 1)

    # Step 3: Find the correction value
    correction_index = np.argmax(r)
    return lags[correction_index]


def fix_scan_phase(data_in, offset, dim):
    """
    Corrects the scan phase of the data based on a given offset along a specified dimension.

    Parameters
    ----------
    dataIn : ndarray
        The input data of shape (sy, sx, sc, sz).
    offset : int
        The amount of offset to correct for.
    dim : int
        Dimension along which to apply the offset.
        1 for vertical (along height/sy), 2 for horizontal (along width/sx).

    Returns
    -------
    ndarray
        The data with corrected scan phase, of shape (sy, sx, sc, sz).
    """
    sy, sx, sc, sz = data_in.shape
    data_out = None
    if dim == 1:
        if offset > 0:
            data_out = np.zeros([sy, sx + offset, sc, sz])
            data_out[0::2, :sx, :, :] = data_in[0::2, :, :, :]
            data_out[1::2, offset:offset + sx, :, :] = data_in[1::2, :, :, :]
        elif offset < 0:
            offset = abs(offset)
            data_out = np.zeros((sy, sx + offset, sc, sz))
            data_out[0::2, offset:offset + sx, :, :] = data_in[0::2, :, :, :]
            data_out[1::2, :sx, :, :] = data_in[1::2, :, :, :]
        else:
            half_offset = int(offset / 2)
            data_out = np.zeros((sy, sx + 2 * half_offset, sc, sz))
            data_out[:, half_offset:half_offset + sx, :, :] = data_in

    elif dim == 2:
        data_out = np.zeros((sy, sx, sc, sz))
        if offset > 0:
            sy = min(data_out.shape[0] - offset, data_in.shape[0])
            data_out[:, 0::2, :, :] = data_in[:, 0::2, :, :]
            usable = data_in[:sy, 1::2, :, :]
            data_out[offset:(offset + sy), 1::2, :, :] = usable

            # data_out[offset:(offset + sy), 1::2, :, :] = data_in[:, 1::2, :, :]
        elif offset < 0:
            offset = abs(offset)
            sy = min(sy, data_out.shape[0] - offset)  # account for the 0's padded during the shift
            data_out[offset:(offset + sy), 0::2, :, :] = data_in[:, 0::2, :, :]
            data_out[:, 1::2, :, :] = data_in[:, 1::2, :, :]
        else:
            data_out[int(offset / 2):sy + int(offset / 2), :, :, :] = data_in

    return data_out


def correct_scan_stack(num_px, num_channels, num_rois, num_frames, roi_data):
    """
    Extract 4D image data from scanimage tiff recordings.

    This function takes the original multidimensional data acquired from microscopy scans and
    reorganizes it into a structured format, applying corrections and adjustments to each channel,
    region of interest (ROI), and frame to prepare it for further analysis.

    Args:
        num_px (tuple): The pixel resolution in the XY plane, given as a tuple of two integers.
        num_channels (int): The total number of channels to process.
        num_rois (int): The number of regions of interest (ROIs) to process.
        num_frames (int): The total number of frames to process.
        roi_data (list): A list containing ROI data objects, where each object contains image data
                         for different channels and frames.

    Returns:
        np.ndarray: A numpy array containing the reorganized image data, with adjustments applied to
                    each channel, ROI, and frame. The array has a data type of int.


    .. deprecated:: 0.1.0
          `correct_scan_stack` will be removed in 0.2.0, it is replaced by
          `scanreader` along with `fix_scan_phase`
    """

    imageData = []
    roi_x, roi_y = num_px

    for channel in range(num_channels):
        print(f"Assembling channel {channel + 1} of {num_channels}...")
        frameTemp = []
        for strip in range(num_rois):
            stripTemp = np.empty((roi_y, roi_x, 1, num_frames), dtype=np.float32)
            for frame in range(num_frames):
                stripTemp[:, :, 0, frame] = roi_data[strip].imageData[channel][frame][0]
            corr = return_scan_offset(stripTemp, 1).astype(int)

            stripTemp = fix_scan_phase(np.array(stripTemp), corr, 1)
            val = round(stripTemp.shape[0] * 0.03)
            stripTemp = stripTemp[val - 1 :, 6:138, :, :]
            if len(frameTemp) != 0:
                frameTemp = np.concatenate((frameTemp, stripTemp), axis=1)
            else:
                frameTemp = stripTemp
        if len(imageData) != 0:
            imageData = np.concatenate((imageData, frameTemp), axis=2)
        else:
            imageData = frameTemp
    return imageData.astype(int)
