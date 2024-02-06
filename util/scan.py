import numpy as np
from scipy.signal import correlate


def return_scan_offset(image_in, dim):
    """
    Compute the scan offset correction between interleaved lines or columns in an image.

    This function calculates the scan offset correction by analyzing the cross-correlation
    between interleaved lines or columns of the input image. The cross-correlation peak
    determines the amount of offset between the lines or columns, which is then used to
    correct for any misalignment in the imaging process.

    Parameters:
    -----------
    image_in : ndarray
        Input image or volume. It can be 2D, 3D, or 4D. The dimensions represent
        [height, width], [height, width, time], or [height, width, time, channel/plane],
        respectively.
    dim : int
        Dimension along which to compute the scan offset correction.
        1 for vertical (along height), 2 for horizontal (along width).

    Returns:
    --------
    int
        The computed correction value, based on the peak of the cross-correlation.

    Examples:
    ---------
    >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> return_scan_offset(img, 1)

    Notes:
    ------
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

    r_full = correlate(Iv1[:, 0], Iv2[:, 0], mode='full', method='auto')
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

    Parameters:
    -----------
    dataIn : ndarray
        The input data of shape (sy, sx, sc, sz).
    offset : int
        The amount of offset to correct for.
    dim : int
        Dimension along which to apply the offset.
        1 for vertical (along height/sy), 2 for horizontal (along width/sx).

    Returns:
    --------
    ndarray
        The data with corrected scan phase, of shape (sy, sx, sc, sz).
    """

    sy, sx, sc, sz = data_in.shape
    data_out = None
    if dim == 1:
        if offset > 0:
            data_out = np.zeros((sy, sx + offset, sc, sz))
            data_out[0::2, :sx, :, :] = data_in[0::2, :, :, :]
            data_out[1::2, offset:offset + sx, :, :] = data_in[1::2, :, :, :]
        elif offset < 0:
            offset = abs(offset)
            data_out = np.zeros((sy, sx + offset, sc, sz))  # This initialization is key
            data_out[0::2, offset:offset + sx, :, :] = data_in[0::2, :, :, :]
            data_out[1::2, :sx, :, :] = data_in[1::2, :, :, :]
        else:
            half_offset = int(offset / 2)
            data_out = np.zeros((sy, sx + 2 * half_offset, sc, sz))
            data_out[:, half_offset:half_offset + sx, :, :] = data_in

    elif dim == 2:
        data_out = np.zeros(sy, sx, sc, sz)
        if offset > 0:
            data_out[:, 0::2, :, :] = data_in[:, 0::2, :, :]
            data_out[offset:(offset + sy), 1::2, :, :] = data_in[:, 1::2, :, :]
        elif offset < 0:
            offset = abs(offset)
            data_out[offset:(offset + sy), 0::2, :, :] = data_in[:, 0::2, :, :]
            data_out[:, 1::2, :, :] = data_in[:, 1::2, :, :]
        else:
            data_out[int(offset / 2):sy + int(offset / 2), :, :, :] = data_in

    return data_out
