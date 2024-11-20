import os
from pathlib import Path
from scanreader import scans


def get_reader(datapath: os.PathLike):
    filepath = Path(datapath)
    if filepath.is_file():
        filepath = datapath
    else:
        filepath = [
            files for files in datapath.glob("*.tif")
        ]  # this accumulates a list of every filepath which contains a .tif file
    return filepath


def read_scan(
    pathnames: os.PathLike | str | list,
    trim_roi_x: list | tuple = (0, 0),
    trim_roi_y: list | tuple = (0, 0),
) -> scans.ScanLBM:
    """
    Reads a ScanImage scan.

    Parameters
    ----------
    pathnames: os.PathLike
        Pathname(s) or pathname pattern(s) to read.
    trim_roi_x: tuple, list, optional
        Indexable (trim_roi_x[0], trim_roi_x[1]) item with 2 integers denoting the amount of pixels to trim on the left [0] and right [1] side of **each roi**.
    trim_roi_y: tuple, list, optional
        Indexable (trim_roi_y[0], trim_roi_y[1]) item with 2 integers denoting the amount of pixels to trim on the top [0] and bottom [1] side of **each roi**.

    Returns
    -------
    ScanLBM
        A Scan object (subclass of ScanMultiROI) with metadata and different offset correction methods.
        See Readme for details.

    """
    # Expand wildcards
    filenames = lbm_io.get_files(pathnames)
    if isinstance(filenames, (list, tuple)):
        if len(filenames) == 0:
            raise FileNotFoundError(
                f"Pathname(s) {filenames} do not match any files in disk."
            )

    # Get metadata from first file
    return scans.ScanLBM(filenames, trim_roi_x=trim_roi_x, trim_roi_y=trim_roi_y)


def return_scan_offset(image_in, nvals: int = 8):
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

        Dimensions: [height, width], [time, height, width], or [time, plane, height, width].
        The input array must be castable to numpy. e.g. np.shape, np.ravel.

    nvals : int
        Number of pixel-wise shifts to include in the search for best correlation.

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
    from scipy import signal

    image_in = image_in.squeeze()

    if len(image_in.shape) == 3:
        image_in = np.mean(image_in, axis=0)
    elif len(image_in.shape) == 4:
        image_in = np.mean(np.mean(image_in, axis=0), axis=0)

    n = nvals

    in_pre = image_in[::2, :]
    in_post = image_in[1::2, :]

    min_len = min(in_pre.shape[0], in_post.shape[0])
    in_pre = in_pre[:min_len, :]
    in_post = in_post[:min_len, :]

    buffers = np.zeros((in_pre.shape[0], n))

    in_pre = np.hstack((buffers, in_pre, buffers))
    in_post = np.hstack((buffers, in_post, buffers))

    in_pre = in_pre.T.ravel(order="F")
    in_post = in_post.T.ravel(order="F")

    # Zero-center and clip negative values to zero
    # Iv1 = Iv1 - np.mean(Iv1)
    in_pre[in_pre < 0] = 0

    in_post = in_post - np.mean(in_post)
    in_post[in_post < 0] = 0

    in_pre = in_pre[:, np.newaxis]
    in_post = in_post[:, np.newaxis]

    r_full = signal.correlate(in_pre[:, 0], in_post[:, 0], mode="full", method="auto")
    unbiased_scale = len(in_pre) - np.abs(np.arange(-len(in_pre) + 1, len(in_pre)))
    r = r_full / unbiased_scale

    mid_point = len(r) // 2
    lower_bound = mid_point - n
    upper_bound = mid_point + n + 1
    r = r[lower_bound:upper_bound]
    lags = np.arange(-n, n + 1)

    # Step 3: Find the correction value
    correction_index = np.argmax(r)
    return lags[correction_index]


def fix_scan_phase(
    data_in,
    offset,
):
    """
    Corrects the scan phase of the data based on a given offset along a specified dimension.

    Parameters:
    -----------
    dataIn : ndarray
        The input data of shape (sy, sx, sc, sz).
    offset : int
        The amount of offset to correct for.

    Returns:
    --------
    ndarray
        The data with corrected scan phase, of shape (sy, sx, sc, sz).
    """
    dims = data_in.shape
    ndim = len(dims)
    if ndim == 2:
        raise NotImplementedError("Array must be > 2 dimensions.")
    if ndim == 4:
        st, sc, sy, sx = data_in.shape
        if offset != 0:
            data_out = np.zeros((st, sc, sy, sx + abs(offset)))
        else:
            print("Phase = 0, no correction applied.")
            return data_in

        if offset > 0:
            data_out[:, :, 0::2, :sx] = data_in[:, :, 0::2, :]
            data_out[:, :, 1::2, offset : offset + sx] = data_in[:, :, 1::2, :]
            data_out = data_out[:, :, :, : sx + offset]
        elif offset < 0:
            offset = abs(offset)
            data_out[:, :, 0::2, offset : offset + sx] = data_in[:, :, 0::2, :]
            data_out[:, :, 1::2, :sx] = data_in[:, :, 1::2, :]
            data_out = data_out[:, :, :, offset:]

        return data_out

    if ndim == 3:
        st, sy, sx = data_in.shape
        if offset != 0:
            # Create output array with appropriate shape adjustment
            data_out = np.zeros((st, sy, sx + abs(offset)))
        else:
            print("Phase = 0, no correction applied.")
            return data_in

        if offset > 0:
            # For positive offset
            data_out[:, 0::2, :sx] = data_in[:, 0::2, :]
            data_out[:, 1::2, offset : offset + sx] = data_in[:, 1::2, :]
            # Trim output by excluding columns that contain only zeros
            data_out = data_out[:, :, : sx + offset]
        elif offset < 0:
            # For negative offset
            offset = abs(offset)
            data_out[:, 0::2, offset : offset + sx] = data_in[:, 0::2, :]
            data_out[:, 1::2, :sx] = data_in[:, 1::2, :]
            # Trim output by excluding the first 'offset' columns
            data_out = data_out[:, :, offset:]

        return data_out

    raise NotImplementedError()
