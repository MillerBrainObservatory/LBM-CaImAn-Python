
from pathlib import Path

import numpy as np

# Globals
RAW_INPUT_DIRS = []
OUTPUT_DIRS = Path('/v-data4/foconnell/')


def init_params():
    """
    Return a dictionary with parameters.

    Parameters:
    ----------
    debug: bool : Whether to return debug messages.
    chans_order_{1, 15, 30} : Channel / plane reordering to put in order of tissue depth.
    save_output : If true, will save each plane in a separate folder, otherwise saves the full volume
    raw_data_dirs : Absolute path to the folder containing your data. Must be a list of 1 or more directories.
    fname_must_contain : Optional string to match filenames to exclude from the analysis.
    fname_must_NOT_contain: Optional string to match filenames to exclude from the analysis.
    make_template_seams_and_plane_alignment : Flag to start reconstruction.
    reconstruct_all_files : Whether to iterate over all files
    reconstruct_until_this_ifile : Iterate over this many files in each raw_data_dirs. Fallback
    list_files_for_template : TBD.
    seams_overlap : "calculate",
    make_nonan_volume :?? Isn't user set, checks for nan's in each plane
        - False if letaral_align_planes = True because it is going to be no-nan by definition, no need to check for it
    lateral_align_planes : Check if  the pipeline can work with int16, and do it if so. If NaN
        handling is required, float32 will be used instead.
    add_1000_for_nonegative_volume : Correct for the first 1000 planes being used by the resonant scanner?
    save_mp4 : Saves the plane across time,
    save_meanf_png : Saves the image of an individual plane.

    TODO: Add checks with clear warnings or errors for each parameter
    TODO: detect the number of planes based on the file metadata and not on the filename
    """
    params = {
        "debug": True,
        "chans_order_1plane": np.array([0]),
        "chans_order_15planes": (np.array([1, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 13, 14, 15]) - 1),
        "chans_order_30planes": (np.array([
                1, 5, 6, 7, 8, 9, 2, 10, 11, 12, 13, 14, 15, 16,
                17, 3, 18, 19, 20, 21, 22, 23, 4, 24, 25, 26, 27, 28, 29, 30,]) - 1),
        "flynn_temp_param": True,
        "raw_data_dirs": [r"/v-data4/foconnell/data/lbm/raw"],
        "output_dir": Path(""),
        "fname_must_contain": "0001",
        "fname_must_NOT_contain": "some_random_stuff",
        "make_template_seams_and_plane_alignment": True,
        "reconstruct_all_files": True,
        "list_files_for_template": [0],
        "json_logging": False,
        "seams_overlap": "calculate",
        "save_output": True,
        "make_nonan_volume": True,
        "lateral_align_planes": False,
        "add_1000_for_nonegative_volume": True,
        "save_mp4": True,
        "save_meanf_png": True,
    }

    return params
