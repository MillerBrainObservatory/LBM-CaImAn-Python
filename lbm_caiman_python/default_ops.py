def default_ops():
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
    gSig = 6
    gSiz = (4 * gSig + 1, 4 * gSig + 1)
    return {
        "main": {
            # Motion correction parameters
            "pw_rigid": True,
            "max_shifts": [6, 6],
            "strides": [96, 96],
            "overlaps": [32, 32],
            "min_mov": None,
            "gsig_Filt": [0, 0],
            "max_deviation_rigid": 3,
            "border_nan": "copy",
            "splits_els": 14,
            "upsample_factor_grid": 4,
            "use_cuda": False,
            "num_frames_split": 50,
            "niter_rig": 1,
            "is3D": False,
            "indices": (slice(None), slice(None)),
            "splits_rig": 14,
            "num_splits_to_process_rig": None,
            # CNMF parameters
            'fr': 10,
            'dxy': (1., 1.),
            'decay_time': 0.4,
            'p': 2,
            'nb': 1,
            'rf': 13,
            'K': 20,
            'gSig': gSig,
            'gSiz': gSiz,
            'stride': [50, 50],
            'method_init': 'greedy_roi',
            'rolling_sum': True,
            'use_cnn': False,
            'ssub': 1,
            'tsub': 1,
            'merge_thr': 0.7,
            'bas_nonneg': True,
            'min_SNR': 1.4,
            'rval_thr': 0.8,
        },
    }
