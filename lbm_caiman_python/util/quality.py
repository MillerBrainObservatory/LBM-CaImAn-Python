import logging
import os
import sys

import cv2
import numpy as np
from sklearn.linear_model import TheilSenRegressor
from caiman.motion_correction import high_pass_filter_space, bin_median
from caiman import load
import matplotlib.pyplot as plt
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
    Y, noise_range=[0.25, 0.5], noise_method="logmexp", max_num_samples_fft=3072
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


def compute_metrics_motion_correction(fname, final_size_x=None, final_size_y=None, swap_dim=False, pyr_scale=.5, levels=3,
                                      winsize=100, iterations=15, poly_n=5, poly_sigma=1.2 / 5, flags=0,
                                      play_flow=False, resize_fact_flow=.2, template=None,
                                      opencv=True, resize_fact_play=3, fr_play=30, max_flow=1,
                                      gSig_filt=None):
    """
    Compute evaluation metrics for motion correction results using optical flow.

    Parameters
    ----------
    fname : str
        Path to the file containing the registration result.
    final_size_x : int, optional
        Final size of the x-dimension after cropping, by default None.
    final_size_y : int, optional
        Final size of the y-dimension after cropping, by default None.
    swap_dim : bool, optional
        Whether to swap dimensions, by default False.
    pyr_scale : float, optional
        Scale between pyramid levels in optical flow calculation, by default 0.5.
    levels : int, optional
        Number of pyramid levels in optical flow calculation, by default 3.
    winsize : int, optional
        Size of the window used in optical flow calculation, by default 100.
    iterations : int, optional
        Number of iterations at each pyramid level, by default 15.
    poly_n : int, optional
        Size of the pixel neighborhood used for polynomial expansion, by default 5.
    poly_sigma : float, optional
        Standard deviation of the Gaussian used for polynomial expansion, by default 1.2/5.
    flags : int, optional
        Flags for the optical flow calculation, by default 0.
    play_flow : bool, optional
        Whether to play back the flow visualization, by default False.
    resize_fact_flow : float, optional
        Resize factor for the input during flow calculation, by default 0.2.
    template : ndarray, optional
        Template image for motion correction comparison, by default None.
    opencv : bool, optional
        Whether to use OpenCV for flow visualization, by default True.
    resize_fact_play : int, optional
        Resize factor for playback visualization, by default 3.
    fr_play : int, optional
        Frame rate for playback visualization, by default 30.
    max_flow : float, optional
        Maximum flow magnitude for visualization, by default 1.
    gSig_filt : float, optional
        Spatial filter size for high-pass filtering, by default None.

    Returns
    -------
    tuple
        - tmpl : ndarray
            Template used for motion correction.
        - flows : list of ndarray
            Optical flow results for each frame.
        - norms : list of float
            Norm of the optical flow for each frame.
    """
    logger = logging.getLogger("caiman")
    if os.environ.get('ENABLE_TQDM') == 'True':
        disable_tqdm = False
    else:
        disable_tqdm = True

    vmin, vmax = -max_flow, max_flow
    m = load(fname)
    if final_size_x is None:
        final_size_x = m.shape[1]
    if final_size_y is None:
        final_size_y = m.shape[2]
    if gSig_filt is not None:
        m = high_pass_filter_space(m, gSig_filt)
    mi, ma = m.min(), m.max()
    m_min = mi + (ma - mi) / 100
    m_max = mi + (ma - mi) / 4

    max_shft_x = int(np.ceil((np.shape(m)[1] - final_size_x) / 2))
    max_shft_y = int(np.ceil((np.shape(m)[2] - final_size_y) / 2))
    max_shft_x_1 = - ((np.shape(m)[1] - max_shft_x) - (final_size_x))
    max_shft_y_1 = - ((np.shape(m)[2] - max_shft_y) - (final_size_y))
    if max_shft_x_1 == 0:
        max_shft_x_1 = None

    if max_shft_y_1 == 0:
        max_shft_y_1 = None
    logger.info([max_shft_x, max_shft_x_1, max_shft_y, max_shft_y_1])
    m = m[:, max_shft_x:max_shft_x_1, max_shft_y:max_shft_y_1]

    if template is None:
        tmpl = bin_median(m)
    else:
        tmpl = template

    m = m.resize(1, 1, resize_fact_flow)
    norms = []
    flows = []
    count = 0
    sys.stdout.flush()
    for fr in tqdm(m, desc="Optical flow", disable=disable_tqdm):
        if disable_tqdm:
            if count % 100 == 0:
                logger.debug(count)

        count += 1
        flow = cv2.calcOpticalFlowFarneback(
            tmpl, fr, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

        if play_flow:
            if opencv:
                dims = tuple(np.array(flow.shape[:-1]) * resize_fact_play)
                vid_frame = np.concatenate([
                    np.repeat(np.clip((cv2.resize(fr, dims)[..., None] - m_min) /
                                      (m_max - m_min), 0, 1), 3, -1),
                    np.transpose([cv2.resize(np.clip(flow[:, :, 1] / vmax, 0, 1), dims),
                                  np.zeros(dims, np.float32),
                                  cv2.resize(np.clip(flow[:, :, 1] / vmin, 0, 1), dims)],
                                 (1, 2, 0)),
                    np.transpose([cv2.resize(np.clip(flow[:, :, 0] / vmax, 0, 1), dims),
                                  np.zeros(dims, np.float32),
                                  cv2.resize(np.clip(flow[:, :, 0] / vmin, 0, 1), dims)],
                                 (1, 2, 0))], 1).astype(np.float32)
                cv2.putText(vid_frame, 'movie', (10, 20), fontFace=5, fontScale=0.8, color=(
                    0, 255, 0), thickness=1)
                cv2.putText(vid_frame, 'y_flow', (dims[0] + 10, 20), fontFace=5, fontScale=0.8, color=(
                    0, 255, 0), thickness=1)
                cv2.putText(vid_frame, 'x_flow', (2 * dims[0] + 10, 20), fontFace=5, fontScale=0.8, color=(
                    0, 255, 0), thickness=1)
                cv2.imshow('frame', vid_frame)
                cv2.waitKey(1 / fr_play)  # to pause between frames
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                plt.subplot(1, 3, 1)
                plt.cla()
                plt.imshow(fr, vmin=m_min, vmax=m_max, cmap='gray')
                plt.title('movie')
                plt.subplot(1, 3, 3)
                plt.cla()
                plt.imshow(flow[:, :, 1], vmin=vmin, vmax=vmax)
                plt.title('y_flow')
                plt.subplot(1, 3, 2)
                plt.cla()
                plt.imshow(flow[:, :, 0], vmin=vmin, vmax=vmax)
                plt.title('x_flow')
                plt.pause(1 / fr_play)

        n = np.linalg.norm(flow)
        flows.append(flow)
        norms.append(n)
    if play_flow and opencv:
        cv2.destroyAllWindows()

    return tmpl, flows, norms

def reshape_spatial(model):
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
    A = model.estimates.A.T
    c = np.zeros((model.dims[1], model.dims[0], 4))
    for a in tqdm.tqdm(A, total=A.shape[0]):
        ar = a.toarray().reshape(model.dims[1], model.dims[0])
        rows, cols = np.where(ar > 0.1)
        c[rows, cols, -1] = ar[rows, cols]
    return c
