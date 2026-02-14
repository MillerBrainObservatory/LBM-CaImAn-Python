"""
postprocessing utilities for caiman results.
"""

from pathlib import Path
from typing import Union

import numpy as np


def load_ops(ops_input) -> dict:
    """
    load ops from file or return dict as-is.

    parameters
    ----------
    ops_input : str, Path, dict, or None
        path to ops.npy file, dictionary, or None.

    returns
    -------
    dict
        ops dictionary.
    """
    if ops_input is None:
        return {}

    if isinstance(ops_input, dict):
        return ops_input.copy()

    if isinstance(ops_input, (str, Path)):
        ops_path = Path(ops_input)

        # handle directory input
        if ops_path.is_dir():
            ops_path = ops_path / "ops.npy"

        if ops_path.exists():
            ops = np.load(ops_path, allow_pickle=True)
            if isinstance(ops, np.ndarray):
                ops = ops.item()
            return dict(ops)
        else:
            return {}

    return {}


def load_planar_results(plane_dir) -> dict:
    """
    load all results from a plane directory.

    parameters
    ----------
    plane_dir : str or Path
        path to plane output directory.

    returns
    -------
    dict
        dictionary containing ops, estimates, F, dff, etc.
    """
    plane_dir = Path(plane_dir)
    results = {}

    # load ops
    ops_file = plane_dir / "ops.npy"
    if ops_file.exists():
        results["ops"] = load_ops(ops_file)

    # load estimates
    estimates_file = plane_dir / "estimates.npy"
    if estimates_file.exists():
        results["estimates"] = np.load(estimates_file, allow_pickle=True).item()

    # load fluorescence
    F_file = plane_dir / "F.npy"
    if F_file.exists():
        results["F"] = np.load(F_file)

    # load dff
    dff_file = plane_dir / "dff.npy"
    if dff_file.exists():
        results["dff"] = np.load(dff_file)

    # load spikes
    spks_file = plane_dir / "spks.npy"
    if spks_file.exists():
        results["spks"] = np.load(spks_file)

    # load motion correction results
    shifts_file = plane_dir / "mcorr_shifts.npy"
    if shifts_file.exists():
        results["shifts"] = np.load(shifts_file, allow_pickle=True)

    template_file = plane_dir / "mcorr_template.npy"
    if template_file.exists():
        results["template"] = np.load(template_file)

    return results


def dff_rolling_percentile(
    F: np.ndarray,
    window_size: int = None,
    percentile: int = 20,
    smooth_window: int = None,
    fs: float = 30.0,
    tau: float = 1.0,
) -> np.ndarray:
    """
    compute dF/F using rolling percentile baseline.

    parameters
    ----------
    F : np.ndarray
        fluorescence traces, shape (n_cells, n_frames).
    window_size : int, optional
        frames for rolling percentile. default: ~10*tau*fs.
    percentile : int, default 20
        percentile for baseline F0.
    smooth_window : int, optional
        smoothing window for dF/F.
    fs : float, default 30.0
        frame rate in Hz.
    tau : float, default 1.0
        decay time constant in seconds.

    returns
    -------
    np.ndarray
        dF/F traces, same shape as F.
    """
    if F is None or F.size == 0:
        return np.array([])

    # ensure 2d
    if F.ndim == 1:
        F = F[np.newaxis, :]

    n_cells, n_frames = F.shape

    # auto-calculate window size
    if window_size is None:
        window_size = int(10 * tau * fs)
    window_size = max(10, min(window_size, n_frames // 2))

    # compute baseline using rolling percentile
    from scipy.ndimage import percentile_filter

    F0 = np.zeros_like(F)
    for i in range(n_cells):
        F0[i] = percentile_filter(F[i], percentile, size=window_size)

    # avoid division by zero
    F0 = np.maximum(F0, 1e-6)

    # compute dF/F
    dff = (F - F0) / F0

    # optional smoothing
    if smooth_window is not None and smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        dff = uniform_filter1d(dff, size=smooth_window, axis=1)

    return dff


def compute_roi_stats(plane_dir) -> dict:
    """
    compute roi quality statistics.

    parameters
    ----------
    plane_dir : str or Path
        path to plane output directory.

    returns
    -------
    dict
        dictionary of roi statistics.
    """
    plane_dir = Path(plane_dir)
    results = load_planar_results(plane_dir)

    stats = {}

    # get estimates
    estimates = results.get("estimates", {})
    A = estimates.get("A")
    C = estimates.get("C")
    SNR = estimates.get("SNR_comp")
    r_values = estimates.get("r_values")

    if A is not None:
        from scipy import sparse
        if sparse.issparse(A):
            A = A.toarray()

        n_cells = A.shape[1]
        stats["n_cells"] = n_cells

        # compute cell sizes (number of pixels)
        cell_sizes = np.array([(A[:, i] > 0).sum() for i in range(n_cells)])
        stats["cell_sizes"] = cell_sizes
        stats["mean_cell_size"] = float(np.mean(cell_sizes))
        stats["median_cell_size"] = float(np.median(cell_sizes))

    if C is not None:
        n_cells, n_frames = C.shape
        stats["n_frames"] = n_frames

        # compute signal statistics
        stats["mean_signal"] = float(np.mean(C))
        stats["max_signal"] = float(np.max(C))

    if SNR is not None:
        stats["snr"] = SNR
        stats["mean_snr"] = float(np.mean(SNR))
        stats["median_snr"] = float(np.median(SNR))

    if r_values is not None:
        stats["r_values"] = r_values
        stats["mean_r_value"] = float(np.mean(r_values))

    # save stats
    np.save(plane_dir / "roi_stats.npy", stats)

    return stats


def get_accepted_cells(plane_dir) -> tuple:
    """
    get indices of accepted and rejected cells.

    parameters
    ----------
    plane_dir : str or Path
        path to plane output directory.

    returns
    -------
    tuple
        (accepted_indices, rejected_indices)
    """
    plane_dir = Path(plane_dir)
    estimates_file = plane_dir / "estimates.npy"

    if not estimates_file.exists():
        return np.array([]), np.array([])

    estimates = np.load(estimates_file, allow_pickle=True).item()

    accepted = estimates.get("idx_components")
    rejected = estimates.get("idx_components_bad")

    if accepted is None:
        # if no evaluation done, accept all
        A = estimates.get("A")
        if A is not None:
            accepted = np.arange(A.shape[1])
        else:
            accepted = np.array([])

    if rejected is None:
        rejected = np.array([])

    return np.asarray(accepted), np.asarray(rejected)


def get_contours(plane_dir, threshold: float = 0.5) -> list:
    """
    get cell contours for visualization.

    parameters
    ----------
    plane_dir : str or Path
        path to plane output directory.
    threshold : float, default 0.5
        threshold for contour extraction (fraction of max).

    returns
    -------
    list
        list of contour coordinates for each cell.
    """
    plane_dir = Path(plane_dir)
    results = load_planar_results(plane_dir)

    estimates = results.get("estimates", {})
    ops = results.get("ops", {})

    A = estimates.get("A")
    if A is None:
        return []

    from scipy import sparse
    if sparse.issparse(A):
        A = A.toarray()

    Ly = ops.get("Ly", int(np.sqrt(A.shape[0])))
    Lx = ops.get("Lx", int(np.sqrt(A.shape[0])))

    contours = []
    for i in range(A.shape[1]):
        component = A[:, i].reshape((Ly, Lx), order="F")

        # find contour
        from skimage import measure
        thresh = component.max() * threshold
        contour_list = measure.find_contours(component, thresh)

        if contour_list:
            contours.append(contour_list[0])
        else:
            contours.append(np.array([]))

    return contours
