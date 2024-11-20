import os

import numpy as np
import tifffile
from pathlib import Path


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

def get_metadata(file: os.PathLike):
    if not file:
        return None

    tiff_file = tifffile.TiffFile(file)
    meta = tiff_file.scanimage_metadata
    str_f = tiff_file.filename

    if 'plane_' in str_f and meta is None:
        raise ValueError(f"No metadata found in {str_f}. This appears to be a processed file. "
                         f"The called function operates only on raw scanimage tiff files.")

    si = meta.get('FrameData', {})
    if not si:
        print(f"No FrameData found in {file}.")
        return None

    series = tiff_file.series[0]
    pages = tiff_file.pages

    # Extract ROI and imaging metadata
    roi_group = meta["RoiGroups"]["imagingRoiGroup"]["rois"]

    num_rois = len(roi_group)
    num_planes = len(si["SI.hChannels.channelSave"])
    scanfields = roi_group[0]["scanfields"]  # assuming single ROI scanfield configuration

    # ROI metadata
    center_xy = scanfields["centerXY"]
    size_xy = scanfields["sizeXY"]
    num_pixel_xy = scanfields["pixelResolutionXY"]

    # TIFF header-derived metadata
    sample_format = pages[0].dtype.name
    objective_resolution = si["SI.objectiveResolution"]
    frame_rate = si["SI.hRoiManager.scanFrameRate"]

    # Field-of-view calculations
    fov_x = round(objective_resolution * size_xy[0])  # adjusted for number of ROIs
    fov_y = round(objective_resolution * size_xy[1])
    fov_xy = (fov_x, fov_y)

    # Pixel resolution calculation
    pixel_resolution = (fov_x / num_pixel_xy[0], fov_y / num_pixel_xy[1])

    # Assembling metadata
    return {
        "image_height": pages[0].shape[0],
        "image_width": pages[0].shape[1],
        "num_pages": len(pages),
        "dims": series.dims,
        "ndim": series.ndim,
        "dtype": 'uint16',
        "is_multifile": series.is_multifile,
        "nbytes": series.nbytes,
        "size": series.size,
        "dim_labels": series.sizes,
        "shape": series.shape,
        "num_planes": num_planes,
        "num_rois": num_rois,
        "num_frames": len(pages) / num_planes,
        "frame_rate": frame_rate,
        "fov": fov_xy,  # in microns
        "pixel_resolution": np.round(pixel_resolution, 2),
        "roi_width_px": num_pixel_xy[0],
        "roi_height_px": num_pixel_xy[1],
        "sample_format": sample_format,
        "num_lines_between_scanfields": round(si["SI.hScan2D.flytoTimePerScanfield"] / si["SI.hRoiManager.linePeriod"]),
        "center_xy": center_xy,
        "line_period": si["SI.hRoiManager.linePeriod"],
        "size_xy": size_xy,
        "objective_resolution": objective_resolution,
        "si": si,
        "roi_info": roi_group
    }

def get_files(
        pathnames: os.PathLike | str | list[os.PathLike | str],
        ext: str = 'tif',
        exclude_pattern: str = '_plane_',
) -> list[os.PathLike | str] | os.PathLike:
    """
    Expands a list of pathname patterns to form a sorted list of absolute filenames.

    Parameters
    ----------
    pathnames: os.PathLike
        Pathname(s) or pathname pattern(s) to read.
    ext: str
        Extention, string giving the filetype extention.
    exclude_pattern: str | list
        A string or list of strings that match to files marked as excluded from processing.

    Returns
    -------
    List[PathLike[AnyStr]]
        List of absolute filenames.
    """
    if '.' in ext or 'tiff' in ext:
        ext = 'tif' #glob tiff and tif
    if isinstance(pathnames, (list, tuple)):
        out_files = []
        excl_files = []
        for fpath in pathnames:
            if exclude_pattern not in str(fpath):
                if Path(fpath).is_file():
                    out_files.extend([fpath])
                elif Path(fpath).is_dir():
                    fnames = [x for x in Path(fpath).expanduser().glob(f"*{ext}*")]
                    out_files.extend(fnames)
            else:
                excl_files.extend(fpath)
        return sorted(out_files)
    if isinstance(pathnames, (os.PathLike, str)):
        pathnames = Path(pathnames).expanduser()
        if pathnames.is_dir():
            files_with_ext = [x for x in pathnames.glob(f"*{ext}*")]
            return sorted(files_with_ext)
        elif pathnames.is_file():
            if exclude_pattern not in str(pathnames):
                return pathnames
            else:
                raise FileNotFoundError(f"No {ext} files found in directory: {pathnames}")
    else:
        raise ValueError(
            f"Input path should be an iterable list/tuple or PathLike object (string, pathlib.Path), not {pathnames}")
