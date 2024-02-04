import numpy as np


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
    make_template_seams_and_plane_alignment : True,
    econstruct_all_files : True,
    list_files_for_template : TBD.
    json_logging : False,  # replaced with debug param
    seams_overlap : "calculate",
    make_nonan_volume : True,
    lateral_align_planes : False,
    add_1000_for_nonegative_volume : True,
    save_mp4 : True,
    save_meanf_png : True,

    TODO: Add checks with clear warnings or errors for each parameter
    TODO: detect the number of planes based on the file metadata and not on the filename
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
        "raw_data_dirs": [r"/v-data4/foconnell/data/lbm/"],
        "fname_must_contain": "",
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
