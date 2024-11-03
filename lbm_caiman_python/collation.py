import numpy as np
from scipy.sparse import hstack


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
