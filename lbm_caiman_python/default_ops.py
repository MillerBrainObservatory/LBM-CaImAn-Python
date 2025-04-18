import numpy as np


def params_from_metadata(metadata):
    """
    Generate parameters for CNMF from metadata.

    Based on the pixel resolution and frame rate, the parameters are set to reasonable values.

    - Sets overlaps and max-shifts to 16 micron.
    - Sets gSig to 8 micron using your pixel-resolution.
    - Sets gSiz to 4 times gSig.
    - Sets max_shifts to 10 micron.
    - Sets strides to 64 micron.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary resulting from `lcp.get_metadata()`.

    Returns
    -------
    dict
        Dictionary of parameters for lbm_mc.

    """
    params = default_params()

    if metadata is None:
        print('No metadata found. Using default parameters.')
        return params

    params["main"]["fr"] = metadata["frame_rate"]
    params["main"]["dxy"] = metadata["pixel_resolution"]

    # typical neuron ~16 microns
    gSig = round(16 / metadata["pixel_resolution"][0]) / 2
    params["main"]["gSig"] = (gSig, gSig)

    gSiz = (4 * gSig[0] + 1, 4 * gSig[0] + 1)
    params["main"]["gSiz"] = gSiz

    max_shifts = [int(round(10 / px)) for px in metadata["pixel_resolution"]]
    params["main"]["max_shifts"] = max_shifts

    strides = [int(round(64 / px)) for px in metadata["pixel_resolution"]]
    params["main"]["strides"] = strides

    overlaps = [int(gSig[0])] * 2
    params["main"]["overlaps"] = overlaps

    rf_0 = (strides[0] + overlaps[0]) // 2
    rf_1 = (strides[1] + overlaps[1]) // 2
    rf = int(np.mean([rf_0, rf_1]))

    stride = int(np.mean([overlaps[0], overlaps[1]]))

    params["main"]["rf"] = rf
    params["main"]["stride"] = stride

    return params


def default_params():
    """
    Default parameters for both registration and CNMF.
    The exception is gSiz being set relative to gSig.

    Returns
    -------
    dict
        Dictionary of default parameter values for registration and segmentation.

    Notes
    -----
    This will likely change as CaImAn is updated.
    """
    gSig = (5, 5)
    gSiz = (4 * gSig[0] + 1, 4 * gSig[0] + 1)
    return {
        "main": {
            # Motion correction parameters
            "pw_rigid": True,
            "max_shifts": [6, 6],
            "strides": [64, 64],
            "overlaps": [8, 8],
            "min_mov": None,
            "gSig_filt": [2, 2],
            "max_deviation_rigid": 3,
            "border_nan": "copy",
            "splits_els": 14,
            "upsample_factor_grid": 4,
            "use_cuda": False,
            "num_frames_split": 50,
            "niter_rig": 1,
            "is3D": False,
            "splits_rig": 14,
            "num_splits_to_process_rig": None,
            # CNMF parameters
            'fr': 10,
            'dxy': (1., 1.),
            'decay_time': 0.5,
            'p': 2,
            'nb': 3,
            'K': 20,
            'rf': 64,
            'stride': [8, 8],
            'gSig': gSig,
            'gSiz': gSiz,
            'method_init': 'greedy_roi',
            'rolling_sum': True,
            'use_cnn': True,
            'ssub': 1,
            'tsub': 1,
            'merge_thr': 0.7,
            'bas_nonneg': True,
            'min_SNR': 1.4,
            'rval_thr': 0.4,
        },
        "refit": True
    }
