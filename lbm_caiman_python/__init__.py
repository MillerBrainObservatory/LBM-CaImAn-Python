from . import _version

# core pipeline api
from .default_ops import default_ops, mcorr_ops, cnmf_ops
from .run_lcp import (
    pipeline,
    run_volume,
    run_plane,
    add_processing_step,
    generate_plane_dirname,
)
from .postprocessing import (
    load_ops,
    load_planar_results,
    dff_rolling_percentile,
    compute_roi_stats,
    get_accepted_cells,
    get_contours,
)

# existing utilities
from .collation import combine_z_planes
from .util.transform import vectorize, unvectorize, calculate_centers
from .util.quality import get_noise_fft, greedyROI
from .util.signal import smooth_data, norm_minmax
from .helpers import (
    generate_patch_view,
    get_single_patch_coords,
    extract_center_square,
)

__version__ = _version.get_versions()['version']

__all__ = [
    # core pipeline
    "pipeline",
    "run_volume",
    "run_plane",
    "default_ops",
    "mcorr_ops",
    "cnmf_ops",
    "add_processing_step",
    "generate_plane_dirname",

    # postprocessing
    "load_ops",
    "load_planar_results",
    "dff_rolling_percentile",
    "compute_roi_stats",
    "get_accepted_cells",
    "get_contours",

    # existing utilities
    "combine_z_planes",
    "generate_patch_view",
    "get_noise_fft",
    "calculate_centers",
    "greedyROI",
    "get_single_patch_coords",
    "extract_center_square",
    "smooth_data",
    "norm_minmax",
]
