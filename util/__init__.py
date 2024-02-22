"""
Utilities for the MaXiMuM project.
"""

import os

import reshaping_stitching_aligning
from .get_mroi_from_tiff import get_mroi_data_from_tiff
from .io import save_to_disk, determine_chunk_size, load_from_disk, read_data_chunk, save_single
from .metadata import parse
from .reorg import reorganize
from .roi_data_simple import RoiDataSimple
from .scan import return_scan_offset, fix_scan_phase


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
        else:  # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1  # Default


def _test():
    """Run ``doctest``"""
    import doctest
    doctest.testmod()


__all__ = [
    'return_scan_offset',
    'fix_scan_phase',
    'reorganize',
    'RoiDataSimple',
    'parse',
    'get_mroi_data_from_tiff',
    'save_to_disk',
    'determine_chunk_size',
    'load_from_disk',
    'read_data_chunk',
    'save_single',
    'reshaping_stitching_aligning'
]
