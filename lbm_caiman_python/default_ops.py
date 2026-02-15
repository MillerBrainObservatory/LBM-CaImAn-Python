"""
default caiman parameters for lbm data processing.
"""


def default_ops() -> dict:
    """
    return default caiman parameters optimized for lbm microscopy data.

    returns
    -------
    dict
        dictionary of parameters for motion correction and cnmf.
    """
    return {
        # motion correction parameters
        "do_motion_correction": True,
        "max_shifts": (6, 6),
        "strides": (48, 48),
        "overlaps": (24, 24),
        "max_deviation_rigid": 3,
        "pw_rigid": True,
        "gSig_filt": (2, 2),
        "border_nan": "copy",
        "niter_rig": 1,
        "splits_rig": 14,
        "num_splits_to_process_rig": None,
        "splits_els": 14,
        "num_splits_to_process_els": None,
        "upsample_factor_grid": 4,
        "max_deviation_rigid": 3,
        "use_cuda": False,

        # cnmf parameters
        "do_cnmf": True,
        "K": 50,
        "gSig": (4, 4),
        "gSiz": None,
        "p": 1,
        "merge_thresh": 0.8,
        "min_SNR": 2.5,
        "rval_thr": 0.85,
        "decay_time": 0.4,
        "method_init": "greedy_roi",
        "ssub": 1,
        "tsub": 1,
        "rf": None,
        "stride": None,
        "nb": 1,
        "gnb": 1,
        "low_rank_background": True,
        "update_background_components": True,
        "rolling_sum": True,
        "only_init": False,
        "normalize_init": True,
        "ring_size_factor": 1.5,

        # component evaluation
        "min_cnn_thr": 0.9,
        "cnn_lowest": 0.1,
        "use_cnn": False,

        # general parameters
        "fr": 30.0,
        "n_processes": None,
        "dxy": (1.0, 1.0),
    }


def mcorr_ops() -> dict:
    """return only motion correction parameters."""
    ops = default_ops()
    return {k: v for k, v in ops.items() if k in (
        "do_motion_correction", "max_shifts", "strides", "overlaps",
        "max_deviation_rigid", "pw_rigid", "gSig_filt", "border_nan",
        "niter_rig", "splits_rig", "num_splits_to_process_rig",
        "splits_els", "num_splits_to_process_els", "upsample_factor_grid",
        "use_cuda", "fr", "n_processes", "dxy",
    )}


def cnmf_ops() -> dict:
    """return only cnmf parameters."""
    ops = default_ops()
    return {k: v for k, v in ops.items() if k in (
        "do_cnmf", "K", "gSig", "gSiz", "p", "merge_thresh", "min_SNR",
        "rval_thr", "decay_time", "method_init", "ssub", "tsub", "rf",
        "stride", "nb", "gnb", "low_rank_background",
        "update_background_components", "rolling_sum", "only_init",
        "normalize_init", "ring_size_factor", "min_cnn_thr", "cnn_lowest",
        "use_cnn", "fr", "n_processes", "dxy",
    )}
