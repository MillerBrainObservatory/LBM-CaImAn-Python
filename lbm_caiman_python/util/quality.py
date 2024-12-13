import logging
import os
import sys

import cv2
import numpy as np
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import pearsonr
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
    """Estimate the noise level for each pixel by averaging the power spectral density.

    Args:
        Y: np.ndarray
            Input movie data with time in the last axis

        noise_range: np.ndarray [2 x 1] between 0 and 0.5
            Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
            default: [0.25,0.5]

        noise method: string
            method of averaging the noise.
            Choices:
                'mean': Mean
                'median': Median
                'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
        sn: np.ndarray
            Noise level for each pixel
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


def compute_quantal_size(scan):
    """
    Estimate the unit change in calcium response corresponding to a unit change in
    pixel intensity (dubbed quantal size, lower is better).

    Assumes images are stationary from one timestep to the next. Uses it to calculate a
    measure of noise per bright intensity (which increases linearly given that imaging
    noise is poisson), fits a line to it and uses the slope as the estimate.

    :param np.array scan: 3-dimensional scan (image_height, image_width, num_frames).

    :returns: int minimum pixel value in the scan (that appears a min number of times)
    :returns: int maximum pixel value in the scan (that appears a min number of times)
    :returns: np.array pixel intensities used for the estimation.
    :returns: np.array noise variances used for the estimation.
    :returns: float the estimated quantal size
    :returns: float the estimated zero value
    """
    # Set some params
    num_frames = scan.shape[2]
    min_count = num_frames * 0.1  # pixel values with fewer appearances will be ignored
    max_acceptable_intensity = 3000  # pixel values higher than this will be ignored

    # Make sure field is at least 32 bytes (int16 overflows if summed to itself)
    scan = scan.astype(np.float32, copy=False)

    # Create pixel values at each position in field
    eps = 1e-4  # needed for np.round to not be biased towards even numbers (0.5 -> 1, 1.5 -> 2, 2.5 -> 3, etc.)
    pixels = np.round((scan[:, :, :-1] + scan[:, :, 1:]) / 2 + eps)
    pixels = pixels.astype(np.int16 if np.max(abs(pixels)) < 2**15 else np.int32)

    # Compute a good range of pixel values (common, not too bright values)
    unique_pixels, counts = np.unique(pixels, return_counts=True)
    min_intensity = min(unique_pixels[counts > min_count])
    max_intensity = max(unique_pixels[counts > min_count])
    max_acceptable_intensity = min(max_intensity, max_acceptable_intensity)
    pixels_mask = np.logical_and(
        pixels >= min_intensity, pixels <= max_acceptable_intensity
    )

    # Select pixels in good range
    pixels = pixels[pixels_mask]
    unique_pixels, counts = np.unique(pixels, return_counts=True)

    # Compute noise variance
    variances = ((scan[:, :, :-1] - scan[:, :, 1:]) ** 2 / 2)[pixels_mask]
    pixels -= min_intensity
    variance_sum = np.zeros(len(unique_pixels))  # sum of variances per pixel value
    for i in range(0, len(pixels), int(1e8)):  # chunk it for memory efficiency
        variance_sum += np.bincount(
            pixels[i : i + int(1e8)],
            weights=variances[i : i + int(1e8)],
            minlength=np.ptp(unique_pixels) + 1,
        )[unique_pixels - min_intensity]
    unique_variances = variance_sum / counts  # average variance per intensity

    # Compute quantal size (by fitting a linear regressor to predict the variance from intensity)
    X = unique_pixels.reshape(-1, 1)
    y = unique_variances
    model = TheilSenRegressor()  # robust regression
    model.fit(X, y)
    quantal_size = model.coef_[0]
    zero_level = -model.intercept_ / model.coef_[0]

    return (
        min_intensity,
        max_intensity,
        unique_pixels,
        unique_variances,
        quantal_size,
        zero_level,
    )


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
