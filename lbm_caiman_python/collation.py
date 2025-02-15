from pathlib import Path
import numpy as np
from scipy.sparse import hstack
import copy
import scipy.signal
import json
import lbm_mc as mc

import lbm_caiman_python


def combine_z_planes(results: dict):
    """
    Combines all z-planes in the results dictionary into a single estimates object.

    Parameters
    ----------
    results (dict): Dictionary with estimates for each z-plane.

    Returns
    -------
    estimates.Estimates: Combined estimates for all z-planes.
    """
    from caiman.source_extraction.cnmf import estimates
    keys = sorted(results.keys())
    e_list = [results[k].estimates for k in keys]

    # Initialize lists to collect components
    A_list = []
    b_list = []
    C_list = []
    f_list = []
    R_list = []

    for e in e_list:
        A_list.append(e.A)
        b_list.append(e.b)
        C_list.append(e.C)
        f_list.append(e.f)
        R_list.append(e.R)

    # Combine the components
    A_new = hstack(A_list).tocsr()
    b_new = np.concatenate(b_list, axis=0)
    C_new = np.concatenate(C_list, axis=0)
    f_new = np.concatenate(f_list, axis=0)
    R_new = np.concatenate(R_list, axis=0)

    # Assuming all z-planes have the same spatial dimensions
    dims_new = e_list[0].dims  # e.g., (height, width)

    # Create new estimates object
    e_new = estimates.Estimates(
        A=A_new,
        C=C_new,
        b=b_new,
        f=f_new,
        R=R_new,
        dims=dims_new
    )

    return e_new


def calculate_interplane_shifts(volume, n_planes, params, json_logger=None):
    """
    """
    interplane_shifts = np.zeros((n_planes - 1, 2), dtype=int)
    accumulated_shifts = np.zeros((n_planes - 1, 2), dtype=int)

    for i_plane in range(n_planes - 1):
        im1_copy = copy.deepcopy(volume[0, :, :, i_plane])
        im2_copy = copy.deepcopy(volume[0, :, :, i_plane + 1])

        # Removing NaNs
        nonan_mask = np.stack(
            (~np.isnan(im1_copy), ~np.isnan(im2_copy)), axis=0
        )
        nonan_mask = np.all(nonan_mask, axis=0)
        coord_nonan_pixels = np.where(nonan_mask)
        min_x, max_x = np.min(coord_nonan_pixels[0]), np.max(coord_nonan_pixels[0])
        min_y, max_y = np.min(coord_nonan_pixels[1]), np.max(coord_nonan_pixels[1])

        im1_nonan = im1_copy[min_x:max_x + 1, min_y:max_y + 1]
        im2_nonan = im2_copy[min_x:max_x + 1, min_y:max_y + 1]

        # Normalize intensities
        im1_nonan -= np.min(im1_nonan)
        im2_nonan -= np.min(im2_nonan)

        # Cross-correlation
        cross_corr_img = scipy.signal.fftconvolve(
            im1_nonan, im2_nonan[::-1, ::-1], mode="same"
        )

        # Find correlation peak
        corr_img_peak_x, corr_img_peak_y = np.unravel_index(
            np.argmax(cross_corr_img), cross_corr_img.shape
        )
        self_corr_peak_x, self_corr_peak_y = [
            dim / 2 for dim in cross_corr_img.shape
        ]
        interplane_shift = [
            corr_img_peak_x - self_corr_peak_x,
            corr_img_peak_y - self_corr_peak_y,
        ]

        interplane_shifts[i_plane] = copy.deepcopy(interplane_shift)
        accumulated_shifts[i_plane] = np.sum(
            interplane_shifts, axis=0, dtype=int
        )

    # Normalize shifts to start at zero
    min_accumulated_shift = np.min(accumulated_shifts, axis=0)
    for xy in range(2):
        accumulated_shifts[:, xy] -= min_accumulated_shift[xy]

    # Optional JSON logging
    if params.get("json_logging") and json_logger:
        json_logger.info(
            json.dumps({"accumulated_shifts": accumulated_shifts.tolist()})
        )

    return accumulated_shifts


if __name__ == "__main__":
    path = Path().home() / 'lbm_data'
    files = lbm_caiman_python.get_files(path, ".tiff", 3)
    print(files)