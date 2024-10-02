import numpy as np
from scipy.sparse import hstack
from caiman.source_extraction.cnmf import estimates


def combine_z_planes(results: dict) -> estimates.Estimates:
    """
    Combines consecutive z-planes in the results dictionary.

    Parameters:
        results (dict): Dictionary of CNMF results, with keys representing z-planes.

    Returns:
        combined_results (dict): Dictionary with combined estimates for each pair of z-planes.
    """
    combined_results = {}
    keys = sorted(results.keys())

    for idx in range(len(keys) - 1):
        i = keys[idx]
        j = keys[idx + 1]

        e1 = results[i].estimates
        e2 = results[j].estimates

        A1, b1, C1, f1, R1, dims1 = e1.A, e1.b, e1.C, e1.f, e1.R, e1.dims
        A2, b2, C2, f2, R2, dims2 = e2.A, e2.b, e2.C, e2.f, e2.R, e2.dims

        # Combine the 2D arrays along the spatial dimension
        b_new = np.concatenate((b1, b2), axis=0)
        C_new = np.concatenate((C1, C2), axis=0)
        f_new = np.concatenate((f1, f2), axis=0)
        R_new = np.concatenate((R1, R2), axis=0)

        # Sparse matrices require scipy.sparse.hstack
        A_new = hstack([A1, A2]).tocsr()

        dims_new = dims1
        e_new = estimates.Estimates(
            A=A_new,
            C=C_new,
            b=b_new,
            f=f_new,
            R=R_new,
            dims=dims_new
        )

        combined_key = f"{i}_{j}"
        combined_results[combined_key] = e_new

        # TODO: Log these with logging module
        print(f"Combined z-plane {i} and {j}:")
        print(f"b_new shape: {b_new.shape}")
        print(f"C_new shape: {C_new.shape}")
        print(f"f_new shape: {f_new.shape}")
        print(f"R_new shape: {R_new.shape}")
        print(f"A_new shape: {A_new.shape}")
        print("-" * 50)

    return combined_results
