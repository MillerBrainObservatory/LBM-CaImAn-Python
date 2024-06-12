"""
General utilities.

.. currentmodule:: .


"""
import os
import sys

from .caiman_interface import (
    extract_masks,
    greedy_roi,
    deconvolve,
    deconvolve_detrended,
    classify_masks,
    get_centroids
)
from .enhancement import (
    create_correlation_image,
    sharpen_2pimage,
    lcn
)
from .galvo_corrections import (
    compute_raster_phase,
    compute_motion_shifts,
    fix_outliers,
    correct_raster,
    correct_motion, 
    fix_scan_phase, 
    return_scan_offset
)
from .performance import (
    map_frames,
    parallel_quality_metrics,
    parallel_motion_shifts
)
from .quality import (
    compute_quantal_size,
    find_peaks
)
from .registration import (
    create_grid,
    resize,
    affine_product,
    sample_grid
)
from .mask_classification import (
    classify_manual,
    classify_manual_extended,
    detect_peaks
)
from .stitching import (
    Position,
    StitchedROI,
    StitchedSlice, 
    linear_stitch
)
from .exceptions import (
    PipelineException
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


def get_size(obj, seen=None, unit="gb"):
    """Recursively finds size of objects"""
    unit = unit.lower()
    if unit not in ["gb", "mb", "kb", "b"]:
        raise ValueError("unit must be one of 'gb', 'mb', 'kb', 'b'")
    elif unit == "gb":
        factor = 1024**3
    elif unit == "mb":
        factor = 1024**2
    elif unit == "kb":
        factor = 1024
    else:
        factor = 1

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return f"{size // factor} {unit}"


__all__ = [
    'enhancement',
    'Position',
    'StitchedROI',
    'StitchedSlice',
    'linear_stitch',
    'map_frames',
    'parallel_quality_metrics',
    'parallel_motion_shifts',
    'find_peaks',
    'detect_peaks',
    'PipelineException',
    'extract_masks',
    'greedy_roi',
    'deconvolve',
    'deconvolve_detrended',
    'classify_masks',
    'get_centroids',
    'create_correlation_image',
    'sharpen_2pimage',
    'lcn',
    'detect_number_of_cores',
    'CacheDict',
    'create_correlation_image',
    'sharpen_2pimage',
    'lcn',
    'caiman_interface',
    'caiman_stats',
    'mask_classification',
    'quality',
    'registration',
    'signal',
    'stitching',
    'correct_raster',
    'correct_motion',
    'compute_quantal_size',
    'compute_motion_shifts',
    'compute_raster_phase',
    'fix_outliers',
    'fix_scan_phase',
    'classify_manual',
    'classify_manual_extended',
    'create_grid',
    'resize',
    'sample_grid',
    'affine_product',
    'galvo_corrections',
    'return_scan_offset',
    'get_size',
]
