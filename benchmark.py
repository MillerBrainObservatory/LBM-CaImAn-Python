# %%[markdown]
# 1. Import Libraries
# %%[code]
import glob
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint

import dask.array as da
import tifffile
from icecream import ic


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


def get_time():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def get_time_str():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# enable or disable logging
# ic.disable()
ic.enable()


def extract_metadata(filename):
    metadata = {}
    with open(filename, "rb") as fh:
        metadata = tifffile.read_scanimage_metadata(fh)
    static_metadata = metadata[0]
    frame_metadata = metadata[1]["RoiGroups"]["imagingRoiGroup"]["rois"]
    rois = [x["scanfields"] for x in frame_metadata]
    raw_arr = da.from_zarr(tifffile.imread(filename, aszarr=True))
    return {
        "fname": filename,
        "arr": raw_arr,
        "center_xy": rois[0]["centerXY"],
        "size_xy": rois[0]["sizeXY"],
        "pixel_resolution_xy": rois[0]["pixelResolutionXY"],
        "num_frames": static_metadata["SI.hStackManager.framesPerSlice"],
        "num_planes": len(static_metadata["SI.hChannels.channelsActive"]),
        "frame_rate": static_metadata["SI.hRoiManager.scanVolumeRate"],
        "objective_resolution": static_metadata["SI.objectiveResolution"],
        "lines_per_frame": static_metadata["SI.hRoiManager.linesPerFrame"],
        "px_per_line": static_metadata["SI.hRoiManager.pixelsPerLine"],
        "file_size_gb": raw_arr.nbytes / 10**9,
        "array_dims": raw_arr.shape,
    }


def get_slice_coordinates(arr_shape, square_size=(142, 142)):
    """Get the start and end coordinates for slicing a square from the center of a 2D array."""
    # Get the dimensions of the input array
    rows, cols = arr_shape

    # grab a square at the center of the array
    start_row = (rows - square_size[0]) // 2
    end_row = start_row + square_size[0]
    start_col = (cols - square_size[1]) // 2
    end_col = start_col + square_size[1]

    return (start_row, start_col), (end_row, end_col)


# %%[code]

sandbox_filepath = "/data2/fpo/lbm/sandbox/"
sandbox_files = sorted(glob.glob(sandbox_filepath + "/*.tif", recursive=True))
# Notice how each file contains different metadata in the filename, i.e. .9mmx.9mm vs 3mm x 5mm
if len(sandbox_files) <= 1:
    print(f"No files found in {sandbox_filepath}")
else:
    print([f"{Path(x).name}" for x in sandbox_files])

# %% md
## Extract metadata without loading the image into memory
# %%
sandbox_metadata = extract_metadata(sandbox_files[0])
for i, filename in enumerate(sandbox_files):
    print(f"File {i+1}")
    pprint(extract_metadata(filename))
    print(" ")

# %%
##  Create a sample dataset
#   We need all of the files the were saved with scanimage in the directory, for reasons described above.
#   We can extract metadata from a single file, and use it to slice the whole dataset.
# We should have 10 files, with 844 frames each, totaling 8440 frames as shown in the metadata for this file
path_sample_files = "/data2/fpo/lbm/3mm_5mm/"
sample_files = sorted(glob.glob(path_sample_files + "/*.tif", recursive=True))
if len(sample_files) <= 1:
    print(f"No files found in {path_sample_files}")
else:
    print([f"{Path(x).name}" for x in sample_files])
sample_metadata = extract_metadata(sample_files[0])

# %%
dd = get_size(sample_metadata["arr"])
# %%
data = tifffile.imread(sample_files[0])

# %%
start, end = get_slice_coordinates((data.shape[1], data.shape[2]))
array_slice = data[:, start[0] : end[0], start[1] : end[1]]
print(array_slice.shape)

# %%
# Write .tif as h5, with metadata as attributes
file = Path(sample_files[0])
new_name = (
    file.name[:-16]
    + f"_{array_slice.shape[0]}_{array_slice.shape[1]}_{array_slice.shape[2]}"
)
new_file = file.parent / Path(new_name).with_suffix(".h5")
new_file
