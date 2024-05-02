"""
General utilities.
"""
import os

from .enhancement import (
    create_correlation_image,
    sharpen_2pimage,
    lcn
)
from .caiman_interface import (
    extract_masks,
    greedy_roi,
    deconvolve,
    deconvolve_detrended,
    classify_masks,
    get_centroids
)
from .galvo_corrections import (
    compute_raster_phase,
    compute_motion_shifts,
    fix_outliers,
    correct_raster,
    correct_motion
)
from .registration import (
    create_grid,
    resize,
    affine_product,
    sample_grid
)


def extract_common_key(filepath):
    parts = filepath.stem.split("_")
    return "_".join(parts[:-1])


class CacheDict(dict):
    """
    A dictionary that prevents itself from growing too much.
    """

    def __init__(self, maxentries):
        self.maxentries = maxentries
        super().__init__(self)

    def __setitem__(self, key, value):
        # Protection against growing the cache too much
        if len(self) > self.maxentries:
            # Remove a 10% of (arbitrary) elements from the cache
            entries_to_remove = self.maxentries / 10
            for k in list(self)[:entries_to_remove]:
                super().__delitem__(k)
        super().__setitem__(key, value)


def detect_number_of_cores():
    """Detects the number of cores on a system."""

    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1  # Default


__all__ = [
    'detect_number_of_cores',
    # 'determine_chunk_size',
    'CacheDict',
    'h5',
    'enhancement',
    'caiman_stats',
    'settings',
    'caiman_interface',
    'mask_classification',
    'quality',
    'registration',
    'signal',
    'stitching',
    'galvo_corrections'
]
