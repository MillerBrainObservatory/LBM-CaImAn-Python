import os
import re

import numpy as np
import tifffile
from pathlib import Path
from glob import glob
from os import path
from scanreader import scans


def expand_wildcard(wildcard):
    """ Expands a list of pathname patterns to form a sorted list of absolute filenames.

    Args:
        wildcard: String or list of strings. Pathname pattern(s) to be extended with glob.

    Returns:
        A list of string. Absolute filenames.
    """
    if isinstance(wildcard, str):
        wildcard_list = [wildcard]
    elif isinstance(wildcard, (tuple, list)):
        wildcard_list = wildcard
    else:
        error_msg = 'Expected string or list of strings, received {}'.format(wildcard)
        raise TypeError(error_msg)

    # Expand wildcards
    rel_filenames = [glob(wildcard) for wildcard in wildcard_list]
    rel_filenames = [item for sublist in rel_filenames for item in sublist]  # flatten list

    # Make absolute filenames
    abs_filenames = [path.abspath(filename) for filename in rel_filenames]

    # Sort
    sorted_filenames = sorted(abs_filenames, key=path.basename)

    return sorted_filenames


def make_json_serializable(obj):
    """Convert metadata to JSON serializable format."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def read_scan(pathnames, dtype=np.int16, join_contiguous=False):
    """ Reads a ScanImage scan. """
    # Expand wildcards
    filenames = expand_wildcard(pathnames)
    if len(filenames) == 0:
        error_msg = 'Pathname(s) {} do not match any files in disk.'.format(pathnames)
        raise FileNotFoundError(error_msg)

    scan = ScanMultiROIReordered(join_contiguous=join_contiguous)

    # Read metadata and data (lazy operation)
    scan.read_data(filenames, dtype=dtype)

    return scan


class ScanMultiROIReordered(scans.ScanMultiROI):
    """
    A subclass of ScanMultiROI that ignores the num_fields dimension
    and reorders the output to [time, z, x, y].
    """

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Call the parent class's __getitem__ with the reordered key
        item = super().__getitem__((0, key[2], key[3], key[1], key[0]))
        if item.ndim == 2:
            return item
        elif item.ndim == 3:
            return np.transpose(item, (2, 0, 1))
        elif item.ndim == 4:
            return np.transpose(item, (3, 2, 0, 1))
        else:
            raise ValueError(f"Unexpected number of dimensions: {item.ndim}")

    @property
    def shape(self):
        return self.num_frames, self.num_channels, self.field_heights[0], self.field_widths[0]

    @property
    def ndim(self):
        return 4

    @property
    def size(self):
        return self.num_frames * self.num_channels * self.field_heights[0] * self.field_widths[0]


def is_raw_scanimage(file: os.PathLike | str):
    """
    Check if a TIFF file is a raw ScanImage TIFF.

    Parameters
    ----------
    file: os.PathLike
        Path to the TIFF file.

    Returns
    -------
    bool
        True if the TIFF file is a raw ScanImage TIFF; False otherwise.
    """
    if not file:
        return False

    tiff_file = tifffile.TiffFile(file)
    if (
            hasattr(tiff_file, 'shaped_metadata')
            and tiff_file.shaped_metadata is not None
            and isinstance(tiff_file.shaped_metadata, (list, tuple))
            and tiff_file.shaped_metadata
            and tiff_file.shaped_metadata[0] not in ([], (), None)
    ):
        if 'image' in tiff_file.shaped_metadata[0]:
            return True
        else:
            return False
    else:
        return False


def get_metadata(file: os.PathLike | str):
    """
    Extract metadata from a TIFF file. This can be a raw ScanImage TIFF or one
    processed via [lbm_caiman_python.save_as()](#save_as).

    Parameters
    ----------
    file: os.PathLike
        Path to the TIFF file.

    Returns
    -------
    dict
        Metadata extracted from the TIFF file.

    Raises
    ------
    ValueError
        If no metadata is found in the TIFF file. This can occur when the file is not a ScanImage TIFF.
    """
    tiff_file = tifffile.TiffFile(file)
    if is_raw_scanimage(file):
        return tiff_file.shaped_metadata[0]['image']
    elif hasattr(tiff_file, 'scanimage_metadata'):
        meta = tiff_file.scanimage_metadata
        if meta is None:
            return None

        si = meta.get('FrameData', {})
        if not si:
            print(f"No FrameData found in {file}.")
            return None
        print("Reading tiff series data...")
        series = tiff_file.series[0]
        print("Reading tiff pages...")
        pages = tiff_file.pages
        print("Raw tiff fully read.")

        # Extract ROI and imaging metadata
        roi_group = meta["RoiGroups"]["imagingRoiGroup"]["rois"]

        if isinstance(roi_group, dict):
            num_rois = 1
            roi_group = [roi_group]
        else:
            num_rois = len(roi_group)

        num_planes = len(si["SI.hChannels.channelSave"])

        if num_rois > 1:
            try:
                sizes = [roi_group[i]["scanfields"][i]["sizeXY"] for i in range(num_rois)]
                num_pixel_xys = [roi_group[i]["scanfields"][i]["pixelResolutionXY"] for i in range(num_rois)]
            except KeyError:
                sizes = [roi_group[i]["scanfields"]["sizeXY"] for i in range(num_rois)]
                num_pixel_xys = [roi_group[i]["scanfields"]["pixelResolutionXY"] for i in range(num_rois)]

            # see if each item in sizes is the same
            assert all([sizes[0] == size for size in sizes]), "ROIs have different sizes"
            assert all([num_pixel_xys[0] == num_pixel_xy for num_pixel_xy in
                        num_pixel_xys]), "ROIs have different pixel resolutions"
            size_xy = sizes[0]
            num_pixel_xy = num_pixel_xys[0]
        else:
            size_xy = [roi_group[0]["scanfields"]["sizeXY"]][0]
            num_pixel_xy = [roi_group[0]["scanfields"]["pixelResolutionXY"]][0]

        # TIFF header-derived metadata
        sample_format = pages[0].dtype.name
        objective_resolution = si["SI.objectiveResolution"]
        frame_rate = si["SI.hRoiManager.scanFrameRate"]

        # Field-of-view calculations
        # TODO: We may want an FOV measure that takes into account contiguous ROIs
        # As of now, this is for a single ROI
        fov_x = round(objective_resolution * size_xy[0])
        fov_y = round(objective_resolution * size_xy[1])
        fov_xy = (int(fov_x), int(fov_y / num_rois))

        # Pixel resolution (dxy) calculation
        pixel_resolution = (fov_x / num_pixel_xy[0], fov_y / num_pixel_xy[1])

        return {
            "num_planes": num_planes,
            "num_frames": int(len(pages) / num_planes),
            "fov": fov_xy,  # in microns
            "num_rois": num_rois,
            "frame_rate": frame_rate,
            "pixel_resolution": np.round(pixel_resolution, 2),
            "ndim": series.ndim,
            "dtype": 'uint16',
            "size": series.size,
            "raw_height": pages[0].shape[0],
            "raw_width": pages[0].shape[1],
            "tiff_pages": len(pages),
            "roi_width_px": num_pixel_xy[0],
            "roi_height_px": num_pixel_xy[1],
            "sample_format": sample_format,
            "objective_resolution": objective_resolution,
        }
    else:
        raise ValueError(f"No metadata found in {file}.")


def get_files_ext(base_dir, str_contains, max_depth, sorted=False) -> list:
    """
    Recursively searches for files with a specific extension up to a given depth and stores their paths in a pickle file.

    Parameters
    ----------
    base_dir : str or Path
        The base directory to start searching.
    str_contains : str
        The file extension to look for (e.g., '.txt').
    max_depth : int
        The maximum depth of subdirectories to search.
    sorted : bool, optional
        Whether to sort files alphanumerically by filename, with digits in ascending order (i.e. 1, 2, 10) (default is False).

    Returns
    -------
    list
        A list of full file paths matching the given extension.
    """
    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Directory '{base_path}' does not exist.")
    if not base_path.is_dir():
        raise NotADirectoryError(f"'{base_path}' is not a directory.")

    files = [
        file for file in base_path.rglob(f'*{str_contains}*')
        if len(file.relative_to(base_path).parts) <= max_depth + 1
    ]

    if sorted:
        def numerical_sort_key(path):
            match = re.search(r'\d+', path.name)
            return int(match.group()) if match else float('inf')

        files.sort(key=numerical_sort_key)

    return [str(file) for file in files]


def get_metrics_path(fname: Path) -> Path:
    """
    Get the path to the computed metrics file for a given data file.
    Assumes the metrics file is to be stored in the same directory as the data file,
    with the same name stem and a '_metrics.npz' suffix.

    Parameters
    ----------
    fname : Path
        The path to the input data file.

    Returns
    -------
    metrics_path : Path
        The path to the computed metrics file.
    """
    fname = Path(fname)
    return fname.with_stem(fname.stem + '_metrics').with_suffix('.npz')


def stack_from_files(files: list):
    """Stacks a list of TIFF files into a Dask array. Can be 3D Tyx or 4D Tzyx.

    Parameters
    ----------
    files : list
        List of TIFF files to stack.

    Returns
    -------
    dask.array.core.Array
        Dask array of the stacked files.

    """
    lazy_arrays = []
    for file in files:
        if Path(file).suffix not in [".tif", ".tiff"]:
            continue
        arr = tifffile.memmap(file)
        lazy_arrays.append(arr)
        # dask_arr = da.from_array(arr, chunks="auto")

    zstack = np.stack(lazy_arrays, axis=1)
    return zstack
