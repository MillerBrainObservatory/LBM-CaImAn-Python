
from pathlib import Path

import numpy as np

# Globals
RAW_INPUT_DIRS = []
OUTPUT_DIRS = Path('/v-data4/foconnell/')


def init_params():
    """
    Initializes and returns a dictionary containing parameters for preprocessing, reconstruction,
    visualization, and saving of imaging data.

    Returns
    -------
    params : dict
        A dictionary with the following keys and default values:
        - debug (bool): Enable debug messages.
        - chans_order_{n}planes (np.array): Channel or plane reordering array to arrange data by tissue depth for n planes.
        - save_output (bool): If True, saves each plane in a separate folder; otherwise, saves the full volume.
        - raw_data_dirs (list): List of strings specifying the absolute paths to folders containing data.
        - fname_must_contain (str): Filenames must contain this string to be included in analysis.
        - fname_must_NOT_contain (str): Filenames must not contain this string to be included in analysis.
        - make_template_seams_and_plane_alignment (bool): Flag to indicate whether to start reconstruction.
        - reconstruct_all_files (bool): If True, iterate over all files; otherwise, use 'reconstruct_until_this_ifile'.
        - reconstruct_until_this_ifile (int): Number of files to process in each directory when 'reconstruct_all_files' is False.
        - list_files_for_template (list): Indices of files to use for creating a template.
        - seams_overlap (str or int or list): Strategy for calculating overlap. If "calculate", dynamically determine the optimal overlap; if int, use as fixed overlap; if list, specify overlap for each plane.
        - save_as_volume_or_planes (str): Specifies saving mode, either as "volume" or "planes".
        - concatenate_all_h5_to_tif (bool): If True, concatenate all .h5 files into a single .tif file.
        - n_ignored_pixels_sides (int): Number of pixels to ignore on each side of the MROI for overlap calculation.
        - min_seam_overlap (int): Minimum seam overlap in pixels for dynamic overlap calculation.
        - max_seam_overlap (int): Maximum seam overlap in pixels for dynamic overlap calculation.
        - alignment_plot_checks (bool): If True, generate plots to check alignment during processing.
        - gaps_columns (int), gaps_rows (int): Gap sizes in pixels for visualization.
        - intensity_percentiles (list): Percentiles for intensity scaling in visualization.
        - meanf_png_only_first_file (bool), video_only_first_file (bool): Flags to limit certain outputs to the first file processed.
        - video_play_speed (int), rolling_average_frames (int), video_duration_secs (int): Parameters for video visualization.
        - lateral_align_planes (bool): If True, perform lateral alignment across planes.
        - make_nonan_volume (bool): If True, ensure the volume does not contain NaNs by trimming or padding.
        - add_1000_for_nonegative_volume (bool): If True, add 1000 to pixel values to ensure non-negative volumes.
        - output_dir (Path): The directory where output files should be saved.
        - json_logging (bool): Enable JSON format for logging debug and process information.

    Notes
    -----
    This function should be modified to include any additional parameters required by the imaging processing
    and analysis pipeline. Users are encouraged to adjust the default values according to their specific needs.

    TODO: Implement checks with clear warnings or errors for parameter inconsistencies.
          Detect the number of planes based on file metadata instead of relying on filename conventions.
    """
    params = {
        "debug": True,
        "chans_order_1plane": np.array([0]),
        "chans_order_15planes": np.arange(1, 16) - 1,
        "chans_order_30planes": np.arange(1, 31) - 1,
        "raw_data_dirs": [r"/v-data4/foconnell/data/lbm/raw"],
        "output_dir": "preprocessed_4",
        "fname_must_contain": "",
        "fname_must_NOT_contain": "",
        "make_template_seams_and_plane_alignment": True,
        "reconstruct_all_files": True,
        "list_files_for_template": [0],
        "seams_overlap": "calculate",
        "save_output": True,
        "save_as_volume_or_planes": "planes",
        "concatenate_all_h5_to_tif": False,
        "n_ignored_pixels_sides": 5,
        "min_seam_overlap": 5,
        "max_seam_overlap": 20,
        "alignment_plot_checks": False,
        "gaps_columns": 5,
        "gaps_rows": 5,
        "intensity_percentiles": [15, 99.5],
        "meanf_png_only_first_file": True,
        "video_only_first_file": True,
        "video_play_speed": 1,
        "rolling_average_frames": 1,
        "video_duration_secs": 20,
        "lateral_align_planes": False,
        "make_nonan_volume": True,
        "add_1000_for_nonegative_volume": True,
        "json_logging": False,
    }

    return params
