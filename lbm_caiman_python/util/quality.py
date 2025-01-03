import numpy as np
from scipy import signal
from tqdm import tqdm


def mean_psd(y, method="logmexp"):
    """
    Averaging the PSD

    Args:
        y: np.ndarray
             PSD values

        method: string
            method of averaging the noise.
            Choices:
             'mean': Mean
             'median': Median
             'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
        mp: array
            mean psd
    """

    if method == "mean":
        mp = np.sqrt(np.mean(y / 2, axis=-1))
    elif method == "median":
        mp = np.sqrt(np.median(y / 2, axis=-1))
    else:
        mp = np.log((y + 1e-10) / 2)
        mp = np.mean(mp, axis=-1)
        mp = np.exp(mp)
        mp = np.sqrt(mp)

    return mp


def get_noise_fft(
    Y, noise_range=None, noise_method="logmexp", max_num_samples_fft=3072
):
    """
    Compute the noise level in the Fourier domain for a given signal.

    Parameters
    ----------
    Y : ndarray
        Input data array. The last dimension is treated as time.
    noise_range : list of float, optional
        Frequency range to estimate noise, by default [0.25, 0.5].
    noise_method : str, optional
        Method to compute the mean noise power spectral density (PSD), by default "logmexp".
    max_num_samples_fft : int, optional
        Maximum number of samples to use for FFT computation, by default 3072.

    Returns
    -------
    tuple
        - sn : float or ndarray
            Estimated noise level.
        - psdx : ndarray
            Power spectral density of the input data.
    """
    if noise_range is None:
        noise_range = [0.25, 0.5]
    T = Y.shape[-1]
    # Y=np.array(Y,dtype=np.float64)

    if T > max_num_samples_fft:
        Y = np.concatenate(
            (
                Y[..., 1 : max_num_samples_fft // 3 + 1],
                Y[
                    ...,
                    int(T // 2 - max_num_samples_fft / 3 / 2) : int(
                        T // 2 + max_num_samples_fft / 3 / 2
                    ),
                ],
                Y[..., -max_num_samples_fft // 3 :],
            ),
            axis=-1,
        )
        T = np.shape(Y)[-1]

    # we create a map of what is the noise on the FFT space
    ff = np.arange(0, 0.5 + 1.0 / T, 1.0 / T)
    ind1 = ff > noise_range[0]
    ind2 = ff <= noise_range[1]
    ind = np.logical_and(ind1, ind2)
    # we compute the mean of the noise spectral density s
    if Y.ndim > 1:
        xdft = np.fft.rfft(Y, axis=-1)
        xdft = xdft[..., ind[: xdft.shape[-1]]]
        psdx = 1.0 / T * abs(xdft) ** 2
        psdx *= 2
        sn = mean_psd(psdx, method=noise_method)

    else:
        xdft = np.fliplr(np.fft.rfft(Y))
        psdx = 1.0 / T * (xdft**2)
        psdx[1:] *= 2
        sn = mean_psd(psdx[ind[: psdx.shape[0]]], method=noise_method)

    return sn, psdx


def find_peaks(trace):
    """Find local peaks in the signal and compute prominence and width at half
    prominence. Similar to Matlab's findpeaks.

    :param np.array trace: 1-d signal vector.

    :returns: np.array with indices for each peak.
    :returns: list with prominences per peak.
    :returns: list with width per peak.
    """
    # Get peaks (local maxima)
    peak_indices = signal.argrelmax(trace)[0]

    # Compute prominence and width per peak
    prominences = []
    widths = []
    for index in peak_indices:
        # Find the level of the highest valley encircling the peak
        for left in range(index - 1, -1, -1):
            if trace[left] > trace[index]:
                break
        for right in range(index + 1, len(trace)):
            if trace[right] > trace[index]:
                break
        contour_level = max(min(trace[left:index]), min(trace[index + 1 : right + 1]))

        # Compute prominence
        prominence = trace[index] - contour_level
        prominences.append(prominence)

        # Find left and right indices at half prominence
        half_prominence = trace[index] - prominence / 2
        for k in range(index - 1, -1, -1):
            if trace[k] <= half_prominence:
                left = k + (half_prominence - trace[k]) / (trace[k + 1] - trace[k])
                break
        for k in range(index + 1, len(trace)):
            if trace[k] <= half_prominence:
                right = (
                    k - 1 + (half_prominence - trace[k - 1]) / (trace[k] - trace[k - 1])
                )
                break

        # Compute width
        width = right - left
        widths.append(width)

    return peak_indices, prominences, widths


def _reshape_spatial(A, model, title):
    c = np.zeros((model.dims[1], model.dims[0], 4))
    with tqdm(total=A.shape[0], desc=f"Processing {title}", leave=True) as pbar:
        for a in A:
            ar = a.toarray().reshape(model.dims[1], model.dims[0])
            rows, cols = np.where(ar > 0.1)
            c[rows, cols, :-1] = np.random.rand(3)
            c[rows, cols, -1] = ar[rows, cols]
            pbar.update(1)
    return c


def get_cnmf_plots(model, title=None):
    """
    Reshapes spatial footprints to overlay.

    Parameters
    ----------
    model : object
        A CNMF model object with `dims` and `estimates` attributes.

    Returns
    -------
    np.ndarray
        A 3D array representing the spatial footprints with the last channel containing the thresholded values.
    """
    if title is None:
        title = 'spatial footprint'
    else:
        title = title
    comp_good = _reshape_spatial(model.estimates.A.T[model.estimates.idx_components, :], model, title)
    comp_bad = _reshape_spatial(model.estimates.A.T[model.estimates.idx_components_bad, :], model, title)
    return comp_good, comp_bad