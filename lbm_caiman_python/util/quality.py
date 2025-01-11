import scipy
from scipy import signal
from sklearn.decomposition import NMF
import cv2
import numpy as np
from scipy.ndimage import correlate
import sys


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
                Y[..., 1: max_num_samples_fft // 3 + 1],
                Y[
                ...,
                int(T // 2 - max_num_samples_fft / 3 / 2): int(
                    T // 2 + max_num_samples_fft / 3 / 2
                ),
                ],
                Y[..., -max_num_samples_fft // 3:],
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
        psdx = 1.0 / T * (xdft ** 2)
        psdx[1:] *= 2
        sn = mean_psd(psdx[ind[: psdx.shape[0]]], method=noise_method)

    return sn, psdx


def find_peaks(trace):
    """
    Find local peaks in the signal and compute prominence and width at half
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
        contour_level = max(min(trace[left:index]), min(trace[index + 1: right + 1]))

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


def _imblur(Y, sig=5, siz=11, nDimBlur=None, kernel=None, opencv=True):
    """
    Spatial filtering with a Gaussian or user defined kernel

    The parameters are specified in GreedyROI

    Args:
        Y: np.ndarray
            d1 x d2 [x d3] x T movie, raw data.

        sig: [optional] list,tuple
            half size of neurons

        siz: [optional] list,tuple
            size of filter kernel (default 2*sig + 1).

        nDimBlur: [optional]
            if you want to specify the number of dimension

        kernel: [optional]
            if you want to specify a kernel

        opencv: [optional]
            if you want to process to the blur using OpenCV method

    Returns:
        the blurred image
    """
    # TODO: document (jerem)
    if kernel is None:
        if nDimBlur is None:
            nDimBlur = Y.ndim - 1
        else:
            nDimBlur = np.min((Y.ndim, nDimBlur))

        if np.isscalar(sig):
            sig = sig * np.ones(nDimBlur)

        if np.isscalar(siz):
            siz = siz * np.ones(nDimBlur)

        X = Y.copy()
        if opencv and nDimBlur == 2:
            if X.ndim > 2:
                # if we are on a video we repeat for each frame
                for frame in range(X.shape[-1]):
                    if sys.version_info >= (3, 0):
                        X[:, :, frame] = cv2.GaussianBlur(X[:, :, frame], tuple(
                            siz), sig[0], None, sig[1], cv2.BORDER_CONSTANT)
                    else:
                        X[:, :, frame] = cv2.GaussianBlur(X[:, :, frame], tuple(siz), sig[
                            0], sig[1], cv2.BORDER_CONSTANT, 0)

            else:
                if sys.version_info >= (3, 0):
                    X = cv2.GaussianBlur(
                        X, tuple(siz), sig[0], None, sig[1], cv2.BORDER_CONSTANT)
                else:
                    X = cv2.GaussianBlur(
                        X, tuple(siz), sig[0], sig[1], cv2.BORDER_CONSTANT, 0)
        else:
            for i in range(nDimBlur):
                h = np.exp(-np.arange(-np.floor(siz[i] / 2),
                                      np.floor(siz[i] / 2) + 1) ** 2 / (2 * sig[i] ** 2))
                h /= np.sqrt(h.dot(h))
                shape = [1] * len(Y.shape)
                shape[i] = -1
                X = correlate(X, h.reshape(shape), mode='constant')

    else:
        X = correlate(Y, kernel[..., np.newaxis], mode='constant')
        # for t in range(np.shape(Y)[-1]):
        #    X[:,:,t] = correlate(Y[:,:,t],kernel,mode='constant', cval=0.0)

    return X


def finetune(Y, cin, nIter=5):
    """compute a initialized version of A and C

    Args:
        Y:  D1*d2*T*K patches

        c: array T*K
            the initial calcium traces

        nIter: int
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

    Returns:
    a: array (d1,D2) the computed A as l2(Y*C)/Y*C

    c: array(T) C as the sum of As on x*y axis
"""
    debug_ = False
    if debug_:
        import os
        f = open('_LOG_1_' + str(os.getpid()), 'w+')
        f.write('Y:' + str(np.mean(Y)) + '\n')
        f.write('cin:' + str(np.mean(cin)) + '\n')
        f.close()

    # we compute the multiplication of patches per traces ( non negatively )
    for _ in range(nIter):
        a = np.maximum(np.dot(Y, cin), 0)
        a = a / np.sqrt(np.sum(a**2) + np.finfo(np.float32).eps)  # compute the l2/a
        # c as the variation of those patches
        cin = np.sum(Y * a[..., np.newaxis], tuple(np.arange(Y.ndim - 1)))

    return a, cin


def greedyROI(Y, nr=30, gSig=[5, 5], gSiz=[11, 11], nIter=5, kernel=None, nb=1,
              rolling_sum=False, rolling_length=100, seed_method='auto'):
    """
    Greedy initialization of spatial and temporal components using spatial Gaussian filtering

    Args:
        Y: np.array
            3d or 4d array of fluorescence data with time appearing in the last axis.

        nr: int
            number of components to be found

        gSig: scalar or list of integers
            standard deviation of Gaussian kernel along each axis

        gSiz: scalar or list of integers
            size of spatial component

        nIter: int
            number of iterations when refining estimates

        kernel: np.ndarray
            User specified kernel to be used, if present, instead of Gaussian (default None)

        nb: int
            Number of background components

        rolling_max: boolean
            Detect new components based on a rolling sum of pixel activity (default: True)

        rolling_length: int
            Length of rolling window (default: 100)

        seed_method: str {'auto', 'manual', 'semi'}
            methods for choosing seed pixels
            'semi' detects nr components automatically and allows to add more manually
            if running as notebook 'semi' and 'manual' require a backend that does not
            inline figures, e.g. %matplotlib tk

    Returns:
        A: np.array
            2d array of size (# of pixels) x nr with the spatial components. Each column is
            ordered columnwise (matlab format, order='F')

        C: np.array
            2d array of size nr X T with the temporal components

        center: np.array
            2d array of size nr x 2 [ or 3] with the components centroids

    Author:
        Eftychios A. Pnevmatikakis and Andrea Giovannucci based on a matlab implementation by Yuanjun Gao
            Simons Foundation, 2015

    See Also:
        http://www.cell.com/neuron/pdf/S0896-6273(15)01084-3.pdf


    """
    d = np.shape(Y)
    Y[np.isnan(Y)] = 0
    med = np.median(Y, axis=-1)
    Y = Y - med[..., np.newaxis]
    gHalf = np.array(gSiz) // 2
    gSiz = 2 * gHalf + 1
    # we initialize every values to zero
    if seed_method.lower() == 'manual':
        nr = 0
    A = np.zeros((np.prod(d[0:-1]), nr), dtype=np.float32)
    C = np.zeros((nr, d[-1]), dtype=np.float32)
    center = np.zeros((nr, Y.ndim - 1), dtype='uint16')

    rho = _imblur(Y, sig=gSig, siz=gSiz, nDimBlur=Y.ndim - 1, kernel=kernel)

    if rolling_sum:
        rolling_filter = np.ones(
            (rolling_length), dtype=np.float32) / rolling_length
        rho_s = scipy.signal.lfilter(rolling_filter, 1., rho ** 2)
        return rho_s
        v = np.amax(rho_s, axis=-1)
    else:
        v = np.sum(rho ** 2, axis=-1)

    if seed_method.lower() != 'manual':
        for k in range(nr):
            # we take the highest value of the blurred total image and we define it as
            # the center of the neuron
            ind = np.argmax(v)
            ij = np.unravel_index(ind, d[0:-1])
            for c, i in enumerate(ij):
                center[k, c] = i

            # we define a squared size around it
            ijSig = [[np.maximum(ij[c] - gHalf[c], 0), np.minimum(ij[c] + gHalf[c] + 1, d[c])]
                     for c in range(len(ij))]
            # we create an array of it (fl like) and compute the trace like the pixel ij through time
            dataTemp = np.array(
                Y[tuple([slice(*a) for a in ijSig])].copy(), dtype=np.float32)
            traceTemp = np.array(np.squeeze(rho[ij]), dtype=np.float32)

            coef, score = finetune(dataTemp, traceTemp, nIter=nIter)
            C[k, :] = np.squeeze(score)
            dataSig = coef[..., np.newaxis] * \
                      score.reshape([1] * (Y.ndim - 1) + [-1])
            xySig = np.meshgrid(*[np.arange(s[0], s[1])
                                  for s in ijSig], indexing='xy')
            arr = np.array([np.reshape(s, (1, np.size(s)), order='F').squeeze()
                            for s in xySig], dtype=int)
            indices = np.ravel_multi_index(arr, d[0:-1], order='F')

            A[indices, k] = np.reshape(
                coef, (1, np.size(coef)), order='C').squeeze()
            Y[tuple([slice(*a) for a in ijSig])] -= dataSig.copy()
            if k < nr - 1 or seed_method.lower() != 'auto':
                Mod = [[np.maximum(ij[c] - 2 * gHalf[c], 0),
                        np.minimum(ij[c] + 2 * gHalf[c] + 1, d[c])] for c in range(len(ij))]
                ModLen = [m[1] - m[0] for m in Mod]
                Lag = [ijSig[c] - Mod[c][0] for c in range(len(ij))]
                dataTemp = np.zeros(ModLen)
                dataTemp[tuple([slice(*a) for a in Lag])] = coef
                dataTemp = _imblur(dataTemp[..., np.newaxis],
                                   sig=gSig, siz=gSiz, kernel=kernel)
                temp = dataTemp * score.reshape([1] * (Y.ndim - 1) + [-1])
                rho[tuple([slice(*a) for a in Mod])] -= temp.copy()
                if rolling_sum:
                    rho_filt = scipy.signal.lfilter(
                        rolling_filter, 1., rho[tuple([slice(*a) for a in Mod])] ** 2)
                    v[tuple([slice(*a) for a in Mod])] = np.amax(rho_filt, axis=-1)
                else:
                    v[tuple([slice(*a) for a in Mod])] = \
                        np.sum(rho[tuple([slice(*a) for a in Mod])] ** 2, axis=-1)
        center = center.tolist()
    else:
        center = []

    res = np.reshape(Y, (np.prod(d[0:-1]), d[-1]),
                     order='F') + med.flatten(order='F')[:, None]
    #    model = NMF(n_components=nb, init='random', random_state=0)
    model = NMF(n_components=nb, init='nndsvdar')
    b_in = model.fit_transform(np.maximum(res, 0)).astype(np.float32)
    f_in = model.components_.astype(np.float32)

    return A, C, np.array(center, dtype='uint16'), b_in, f_in