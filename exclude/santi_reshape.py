# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 19:02:03 2023

@author: otero
"""

# %%
# Pre-processing pipeline of 2pRAM and LBM data
# Steps:
# - reshapes the axis to have an x,y,z,t volume (LBM)
# - sorts the z-planes (LBM)
# - calculates and corrects the MROI seams
# - calculates and corrects the X-Y shifts across planes (LBM)
# - outputs data as x-y-t planes or x-y-z-t volumes)


# %%

# If a silent crash happens (and the Terminal closes) despite not reaching 100% RAM, it could be due to the system-oom-process-killer being too sensitive (or acting weird)
# It might be worth trying to increase the thresholds for when to kill processes, or to turn it off altogether...
# https://askubuntu.com/questions/1404888/how-do-i-disable-the-systemd-oom-process-killer-in-ubuntu-22-04
# To turned it off:
# $ systemctl disable --now systemd-oomd
# $ systemctl mask systemd-oomd
# It can be turned back on with:
# $ systemctl enable systemd-oomd
# $ systemctl unmask systemd-oomd

import copy
import datetime
import glob
import json
import logging
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 900
import numpy as np
import os
import scipy.signal
import skimage
import sys
import tifffile
import time

home = Path().home() / 'caiman_data' / 'session_01' / ''
contents = home.glob("*")
print([x for x in contents])

# %% USER-DEFINED PARAMETERS

params = dict()

params["make_template_seams_and_plane_alignment"] = True
params["list_files_for_template"] = [0]

params["reconstruct_all_files"] = True
if not params["reconstruct_all_files"]:
    params["reconstruct_until_this_ifile"] = 10

# Directories
params["raw_data_dirs"] = [
    str(Path().home() / 'caiman_data' / 'animal_01' / 'session_01')
]  # Must be a list with 1 or more dirs
params[
    "fname_must_contain"
] = ""  # something you want to specify and that the desired filenames should contain
params[
    "fname_must_NOT_contain"
] = "some_random_stuff"  # if not needed, leave something you know it is not in the filename

# Plane order in raw files (it will only use the one matching your data's number of planes). This requires your filenames to contain either 'SP', 'Max15', or 'Max30'
# TODO: detect the number of planes based on the file metadata and not on the filename
params["chans_order_1plane"] = np.array([0])
params["chans_order_15planes"] = (
    np.array([1, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 13, 14, 15]) - 1
)
params["chans_order_30planes"] = (
    np.array(
        [
            1,
            5,
            6,
            7,
            8,
            9,
            2,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            3,
            18,
            19,
            20,
            21,
            22,
            23,
            4,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        ]
    )
    - 1
)

# Parameters output planes/volumes
params["save_output"] = True
if params["save_output"]:
    params[
        "save_as_volume_or_planes"
    ] = "planes"  # 'planes' will save individual planes in subfolders -- 'volume' will save a whole 4D hdf5 volume
    if params["save_as_volume_or_planes"] == "planes":
        params[
            "concatenate_all_h5_to_tif"
        ] = False  # If True, it will take all the time-chunked h5 files, concatenate, and save them as a single .tif

params[
    "make_nonan_volume"
] = True  # Whether to trim the edges so the output does not have nans. Also affects output as planes if lateral_aligned_planes==True (to compensate for X-Y shifts of MAxiMuM) or params['identical_mroi_overlaps_across_planes']==False (if seams from different planes are merged differently, then some planes will end up being larger than others)
params["lateral_align_planes"] = False  # Calculates and compensates the X-Y of MAxiMuM
params["add_1000_for_nonegative_volume"] = True

# Parameters MROIs seams
params[
    "seams_overlap"
] = "calculate"  # Should be either 'calculate', an integer, or a list of integers with length=n_planes
if params["seams_overlap"] == "calculate":
    params[
        "n_ignored_pixels_sides"
    ] = 5  # Useful if there is a delay or incorrect phase for when the EOM turns the laser on/off at the start/end of a resonant-scanner line
    params["min_seam_overlap"] = 5
    params[
        "max_seam_overlap"
    ] = 20  # Used if params['seams_overlap']_setting = 'calculate'
    params["alignment_plot_checks"] = False

# Logging
params["json_logging"] = False

# Video and mean-frame png
params["save_mp4"] = True
params["save_meanf_png"] = True
if params["save_mp4"] or params["save_meanf_png"]:
    params["gaps_columns"] = 5
    params["gaps_rows"] = 5
    params["intensity_percentiles"] = [15, 99.5]
    if params["save_meanf_png"]:
        params["meanf_png_only_first_file"] = True
    if params["save_mp4"]:
        params["video_only_first_file"] = True
        params["video_play_speed"] = 1
        params["rolling_average_frames"] = 1
        params["video_duration_secs"] = 20


# %%

# This will check if the pipeline can work with int16, and do it if possible. If NaN handling is required, float32 will be used instead
if not params["lateral_align_planes"]:
    initialize_volume_with_nans = False
    convert_volume_float32_to_int16 = True
    params[
        "make_nonan_volume"
    ] = False  # It is going to be no-nan by definition, no need to check for it
elif params["make_nonan_volume"]:
    initialize_volume_with_nans = True
    convert_volume_float32_to_int16 = True
else:
    initialize_volume_with_nans = True
    convert_volume_float32_to_int16 = False


# %% Set up logger and log parameters

if params["json_logging"]:
    now = datetime.datetime.now()
    date_string = now.strftime("%Y%m%dd_%H%M%St")
    json_filename = f"{params['raw_data_dirs'][0]}log_{date_string}.json"
    json_formatter = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
    )

    json_logger = logging.getLogger(__name__)
    json_logger.setLevel(logging.DEBUG)
    json_handler = logging.FileHandler(json_filename)
    print_handler = logging.StreamHandler(sys.stdout)
    json_handler.setFormatter(json_formatter)
    json_logger.addHandler(json_handler)
    json_logger.addHandler(print_handler)

    json_logger.info(json.dumps(str(params)))


# %% Look for files used to: 1) make a template and do seam-overlap handling and X-Y shift alignment; 2) pre-process
path_all_files = []
for i_dir in params["raw_data_dirs"]:
    tmp_paths = sorted(glob.glob(i_dir + "/**/*.tif*", recursive=True))
    for this_tmp_path in tmp_paths:
        if (
            params["fname_must_contain"] in this_tmp_path
            and params["fname_must_NOT_contain"] not in this_tmp_path
        ):
            path_all_files.append(this_tmp_path)

if params["json_logging"]:
    json_logger.info(json.dumps({"path_all_files": path_all_files}))

n_template_files = len(params["list_files_for_template"])
path_template_files = [
    path_all_files[file_idx] for file_idx in params["list_files_for_template"]
]

del (
    i_dir,
    params["raw_data_dirs"],
    params["fname_must_contain"],
    params["fname_must_NOT_contain"],
)

# %%
pipeline_steps = []
if params["make_template_seams_and_plane_alignment"]:
    pipeline_steps.append("make_template")
if params["reconstruct_all_files"]:
    pipeline_steps.append("reconstruct_all")


for current_pipeline_step in pipeline_steps:
    if current_pipeline_step == "make_template":
        path_input_files = path_template_files
    elif current_pipeline_step == "reconstruct_all":
        path_input_files = path_all_files

    # %% Iterate over files
    if params["reconstruct_all_files"]:
        list_files_for_reconstruction = range(len(path_input_files))
    else:
        list_files_for_reconstruction = range(params["reconstruct_until_this_ifile"])

    for i_file in list_files_for_reconstruction:
        tic = time.time()
        path_input_file = path_input_files[i_file]

        print("---------------------------------------------------")
        if params["json_logging"]:
            json_logger.debug(
                json.dumps(
                    {"debug_message": "Started working on: " + str(path_input_file)}
                )
            )

        # %% Determine if it is a single-plane, Max15, or Max30 recording

        if i_file == 0:
            if "SP" in path_input_file:
                n_planes = 1
            elif "Max15" in path_input_file:
                n_planes = 15
            elif "Max30" in path_input_file:
                n_planes = 30
            else:
                # n_planes = int(input("Check filename... Number of planes?"))
                n_planes = 30

            if n_planes == 1:
                chans_order = params["chans_order_1plane"]
                rows, columns = 1, 1  # For png and mp4
            elif n_planes == 15:
                chans_order = params["chans_order_15planes"]
                rows, columns = 3, 5
            elif n_planes == 30:
                chans_order = params["chans_order_30planes"]
                rows, columns = 6, 5

            if params["json_logging"]:
                json_logger.debug(
                    json.dumps({"debug_message": "Number of planes: " + str(n_planes)})
                )

            # %% Get MROI info from tif metadata

            with tifffile.TiffFile(path_input_file) as tif:
                metadata = {}
                for tag in tif.pages[0].tags.values():
                    tag_name, tag_value = tag.name, tag.value
                    metadata[tag_name] = tag_value

            mrois_si_raw = json.loads(metadata["Artist"])["RoiGroups"][
                "imagingRoiGroup"
            ]["rois"]
            if type(mrois_si_raw) != dict:
                mrois_si = []
                for roi in mrois_si_raw:
                    if type(roi["scanfields"]) != list:
                        scanfield = roi["scanfields"]
                    else:
                        scanfield = roi["scanfields"][
                            np.where(np.array(roi["zs"]) == 0)[0][0]
                        ]
                    roi_dict = {}
                    roi_dict["center"] = np.array(scanfield["centerXY"])
                    roi_dict["sizeXY"] = np.array(scanfield["sizeXY"])
                    roi_dict["pixXY"] = np.array(scanfield["pixelResolutionXY"])
                    mrois_si.append(roi_dict)
            else:
                scanfield = mrois_si_raw["scanfields"]
                roi_dict = {}
                roi_dict["center"] = np.array(scanfield["centerXY"])
                roi_dict["sizeXY"] = np.array(scanfield["sizeXY"])
                roi_dict["pixXY"] = np.array(scanfield["pixelResolutionXY"])
                mrois_si = [roi_dict]

            # Sort MROIs so they go from left-to-right (but keep the un-sorted because that matches how they were acquired and saved in the long-tif-strip)
            mrois_centers_si = np.array([mroi_si["center"] for mroi_si in mrois_si])
            x_sorted = np.argsort(mrois_centers_si[:, 0])
            mrois_si_sorted_x = [mrois_si[i] for i in x_sorted]
            mrois_centers_si_sorted_x = [mrois_centers_si[i] for i in x_sorted]

        # %% Load, reshape (so time and planes are 2 independent dimensions) and re-order (planes, fix Jeff's order)

        if params["json_logging"]:
            json_logger.debug(
                json.dumps(
                    {
                        "debug_message": "Loading file (expect warning if multi-file recording)"
                    }
                )
            )

        tiff_file = tifffile.imread(path_input_file)
        if n_planes > 1:
            tiff_file = np.reshape(
                tiff_file,
                (
                    int(tiff_file.shape[0] / n_planes),
                    n_planes,
                    tiff_file.shape[1],
                    tiff_file.shape[2],
                ),
                order="A",
            )  # warnings are expected if the recording is split into many files or incomplete
        else:
            tiff_file = np.expand_dims(tiff_file, 1)
        tiff_file = np.swapaxes(tiff_file, 1, 3)
        tiff_file = tiff_file[..., chans_order]

        if current_pipeline_step == "make_template":
            tiff_file = np.mean(tiff_file, axis=0, keepdims=True)

        # %% Separate tif into MROIs
        # Get the Y coordinates for mrois (and not flybacks)
        if i_file == 0:
            n_mrois = len(mrois_si)
            tif_pixels_Y = tiff_file.shape[2]
            mrois_pixels_Y = np.array([mroi_si["pixXY"][1] for mroi_si in mrois_si])
            each_flyback_pixels_Y = (tif_pixels_Y - mrois_pixels_Y.sum()) // (
                n_mrois - 1
            )

        if params["json_logging"]:
            json_logger.debug(
                json.dumps({"debug_message": "Separating tif into individual MROIs"})
            )

        # Divide long stripe into mrois
        planes_mrois = np.empty((n_planes, n_mrois), dtype=np.ndarray)
        for i_plane in range(n_planes):
            y_start = 0
            for i_mroi in range(
                n_mrois
            ):  # We go over the order in which they were acquired
                if i_plane == 0:
                    print(f'From: {y_start}, \n To: {y_start + mrois_pixels_Y[x_sorted[i_mroi]]}')
                planes_mrois[i_plane, i_mroi] = tiff_file[:, :, y_start : y_start + mrois_pixels_Y[x_sorted[i_mroi]], i_plane]
                y_start += mrois_pixels_Y[i_mroi] + each_flyback_pixels_Y

        del tiff_file

        if current_pipeline_step == "make_template": # combine multiple templates
            if n_template_files > 1:
                if i_file == 0:
                    template_accumulator = copy.deepcopy(planes_mrois)
                    continue
                elif i_file != n_template_files - 1:
                    template_accumulator += planes_mrois
                    continue
                else:
                    template_accumulator += planes_mrois
                    planes_mrois = template_accumulator / n_template_files

        # %% Get location of MROIs in final canvas based on MROI metadata
        if current_pipeline_step == "make_template":
            # Get pixel sizes, assumes they are the same for each plane (we only get pixel sizes from the first plane)
            sizes_mrois_pix = np.array(
                [mroi_pix.shape[1:] for mroi_pix in planes_mrois[0, :]]
            )
            sizes_mrois_si = np.array(
                [mroi_si["sizeXY"] for mroi_si in mrois_si_sorted_x]
            )
            pixel_sizes = sizes_mrois_si / sizes_mrois_pix
            psize_x, psize_y = np.mean(pixel_sizes[:, 0]), np.mean(pixel_sizes[:, 1])
            assert np.product(
                np.isclose(pixel_sizes[:, 1] - psize_y, 0)
            ), "Y-pixels resolution not uniform across MROIs"
            assert np.product(
                np.isclose(pixel_sizes[:, 0] - psize_x, 0)
            ), "X-pixels resolution not uniform across MROIs"
            # assert np.product(np.isclose(pixel_sizes[:,0]-pixel_sizes[:,1], 0)), "Pixels do not have squared resolution"

            # Calculate the pixel ranges (with their SI locations) that would fit all MROIs
            top_left_corners_si = mrois_centers_si_sorted_x - sizes_mrois_si / 2
            bottom_right_corners_si = mrois_centers_si_sorted_x + sizes_mrois_si / 2
            xmin_si, ymin_si = (
                top_left_corners_si[:, 0].min(),
                top_left_corners_si[:, 1].min(),
            )
            xmax_si, ymax_si = (
                bottom_right_corners_si[:, 0].max(),
                bottom_right_corners_si[:, 1].max(),
            )
            reconstructed_xy_ranges_si = [
                np.arange(xmin_si, xmax_si, psize_x),
                np.arange(ymin_si, ymax_si, psize_y),
            ]

            # Calculate the starting pixel for each MROI when placed in the reconstructed movie
            top_left_corners_pix = np.empty((n_mrois, 2), dtype=int)
            for i_xy in range(2):
                for i_mroi in range(n_mrois):
                    closest_xy_pix = np.argmin(
                        np.abs(
                            reconstructed_xy_ranges_si[i_xy]
                            - top_left_corners_si[i_mroi, i_xy]
                        )
                    )
                    top_left_corners_pix[i_mroi, i_xy] = int(closest_xy_pix)
                    closest_xy_si = reconstructed_xy_ranges_si[i_xy][closest_xy_pix]
                    if not np.isclose(closest_xy_si, top_left_corners_si[i_mroi, i_xy]):
                        if params["json_logging"]:
                            json_logger.debug(
                                json.dumps(
                                    {
                                        "debug_message": "ROI %d x does not fit perfectly into image, corner is %.4f but closest available is %.4f"
                                        % (
                                            i_mroi,
                                            closest_xy_si,
                                            top_left_corners_si[i_mroi, i_xy],
                                        )
                                    }
                                )
                            )
            # Sometimes an extra pixel is added because of pixel_size rounding
            for i_xy in range(2):
                if (
                    len(reconstructed_xy_ranges_si[i_xy])
                    == np.sum(sizes_mrois_pix[:, 0]) + 1
                ):
                    reconstructed_xy_ranges_si[i_xy] = reconstructed_xy_ranges_si[i_xy][
                        :-1
                    ]

        # %% Calculate optimal overlap for seams
        if current_pipeline_step == "make_template":
            if params["seams_overlap"] == "calculate":
                # Determine if all the MROIs are adjacent
                for i_mroi in range(n_mrois - 1):
                    if (
                        top_left_corners_pix[i_mroi][0] + sizes_mrois_pix[i_mroi][0]
                        != top_left_corners_pix[i_mroi + 1][0]
                    ):
                        raise Exception(
                            "MROIs number "
                            + str(i_mroi)
                            + " and number "
                            + str(i_mroi + 1)
                            + " (0-based idx) are not contiguous"
                        )

                # Combine meanf from differete template files:
                overlaps_planes_seams_scores = np.zeros(
                    (
                        n_planes,
                        n_mrois - 1,
                        params["max_seam_overlap"] - params["min_seam_overlap"],
                    )
                )  # We will avoid i_overlaps = 0

                for i_plane in range(n_planes):
                    for i_seam in range(n_mrois - 1):
                        for i_overlaps in range(
                            params["min_seam_overlap"], params["max_seam_overlap"]
                        ):
                            strip_left = planes_mrois[i_plane, i_seam][
                                0,
                                -params["n_ignored_pixels_sides"]
                                - i_overlaps : -params["n_ignored_pixels_sides"],
                            ]
                            strip_right = planes_mrois[i_plane, i_seam + 1][
                                0,
                                params["n_ignored_pixels_sides"] : i_overlaps
                                + params["n_ignored_pixels_sides"],
                            ]
                            subtract_left_right = abs(strip_left - strip_right)
                            overlaps_planes_seams_scores[
                                i_plane, i_seam, i_overlaps - params["min_seam_overlap"]
                            ] = np.mean(subtract_left_right)

                overlaps_planes_scores = np.mean(overlaps_planes_seams_scores, axis=(1))
                overlaps_planes = []
                for i_plane in range(n_planes):
                    overlaps_planes.append(
                        int(
                            np.argmin(overlaps_planes_scores[i_plane])
                            + params["min_seam_overlap"]
                            + 2 * params["n_ignored_pixels_sides"]
                        )
                    )
                if params["json_logging"]:
                    json_logger.info(json.dumps({"overlaps_planes": overlaps_planes}))

                # Plot the scores for the different planes and also potential shifts
                if params["alignment_plot_checks"]:
                    for i_plane in range(n_planes):
                        plt.plot(
                            range(
                                params["min_seam_overlap"], params["max_seam_overlap"]
                            ),
                            overlaps_planes_scores[i_plane],
                        )
                    plt.title("Score for all planes")
                    plt.xlabel("Overlap (pixels)")
                    plt.ylabel("Error (a.u.)")
                    plt.show()

                    for i_plane in range(n_planes):
                        for shift in range(-2, 3):
                            i_overlap = overlaps_planes[i_plane] + shift
                            canvas_alignment_check = np.zeros(
                                (
                                    len(reconstructed_xy_ranges_si[0])
                                    - (n_mrois - 1) * i_overlap,
                                    len(reconstructed_xy_ranges_si[1]),
                                    3,
                                ),
                                dtype=np.float32,
                            )
                            x_start = 0
                            for i_mroi in range(n_mrois):
                                x_start = (
                                    top_left_corners_pix[i_mroi][0] - i_mroi * i_overlap
                                )
                                x_end = x_start + sizes_mrois_pix[i_mroi][0]
                                y_start = top_left_corners_pix[i_mroi][1]
                                y_end = y_start + sizes_mrois_pix[i_mroi][1]
                                canvas_alignment_check[
                                    x_start:x_end, y_start:y_end, i_mroi % 2
                                ] = planes_mrois[0, i_plane, i_mroi] - np.min(
                                    planes_mrois[0, i_plane, i_mroi]
                                )

                            pct_low, pct_high = np.nanpercentile(
                                canvas_alignment_check, [80, 99.9]
                            )  # Consider that we are using 1/3 of pixels (RGB channels)
                            canvas_alignment_check = skimage.exposure.rescale_intensity(
                                canvas_alignment_check, in_range=(pct_low, pct_high)
                            )
                            plt.imshow(np.swapaxes(canvas_alignment_check, 0, 1))
                            plt.title("Plane: " + str(i_plane))
                            plt.xlabel(
                                "Original overlap: "
                                + str(overlaps_planes[i_plane])
                                + " + Shift: "
                                + str(shift)
                            )
                            plt.show()

                overlaps_planes = [int(round(np.mean(overlaps_planes)))] * n_planes

            elif type(params["seams_overlap"]) is int:
                overlaps_planes = [params["seams_overlap"]] * n_planes
            elif params["seams_overlap"] is list:
                overlaps_planes = params["seams_overlap"]
            else:
                raise Exception(
                    "params['seams_overlap'] should be set to 'calculate', an integer, or a list of length n_planes"
                )

        # %% Create a volume container

        if (
            current_pipeline_step == "make_template"
        ):  # For templatingMROIs, we will get here when working on the last file
            n_f = 1
        elif current_pipeline_step == "reconstruct_all":
            n_f = n_f = planes_mrois[0, 0].shape[0]

            # For template or if no need to align planes, initialize interplane shifts as 0s
        if (
            current_pipeline_step == "make_template"
            or not params["lateral_align_planes"]
        ):
            interplane_shifts = np.zeros((n_planes, 2), dtype=int)
            accumulated_shifts = np.zeros((n_planes, 2), dtype=int)

        max_shift_x = max(accumulated_shifts[:, 0])
        max_shift_y = max(accumulated_shifts[:, 1])

        n_x = (
            len(reconstructed_xy_ranges_si[0])
            - min(overlaps_planes) * (n_mrois - 1)
            + max_shift_x
        )
        n_y = len(reconstructed_xy_ranges_si[1]) + max_shift_y
        n_z = n_planes

        if params["json_logging"]:
            json_logger.debug(
                json.dumps(
                    {
                        "debug_message": "Creating volume of shape: "
                        + str([n_f, n_x, n_y, n_z])
                        + " (f,x,y,z)"
                    }
                )
            )

        if initialize_volume_with_nans:
            volume = np.full((n_f, n_x, n_y, n_z), np.nan, dtype=np.float32)
        else:
            volume = np.empty((n_f, n_x, n_y, n_z), dtype=np.int16)

        # %% Merge MROIs and place the plane in the volume (with lateral offsets)
        if params["json_logging"]:
            json_logger.debug(
                json.dumps(
                    {"debug_message": "Merging MROIs and placing them into the volume"}
                )
            )
        for i_plane in range(n_planes):
            overlap_seams_this_plane = overlaps_planes[i_plane]
            plane_width = len(
                reconstructed_xy_ranges_si[0]
            ) - overlap_seams_this_plane * (n_mrois - 1)
            plane_length = len(reconstructed_xy_ranges_si[1])
            plane_canvas = np.zeros((n_f, plane_width, plane_length), dtype=np.float32)
            for i_mroi in range(n_mrois):
                # The first and last MROIs require different handling
                if i_mroi == 0:
                    x_start_canvas = (
                        0  # This always works because the MROIs were sorted
                    )
                    x_end_canvas = (
                        x_start_canvas
                        + sizes_mrois_pix[i_mroi][0]
                        - int(np.trunc(overlap_seams_this_plane / 2))
                    )
                    x_start_mroi = x_start_canvas
                    x_end_mroi = x_end_canvas
                elif i_mroi != n_mrois - 1:
                    x_start_canvas = copy.deepcopy(x_end_canvas)
                    x_end_canvas = (
                        x_start_canvas
                        + sizes_mrois_pix[i_mroi][0]
                        - overlap_seams_this_plane
                    )
                    x_mroi_width = sizes_mrois_pix[i_mroi][0] - overlap_seams_this_plane
                    x_start_mroi = int(np.ceil(overlap_seams_this_plane / 2))
                    x_end_mroi = x_start_mroi + x_mroi_width
                else:
                    x_start_canvas = copy.deepcopy(x_end_canvas)
                    x_end_canvas = plane_width
                    x_start_mroi = int(np.ceil(overlap_seams_this_plane / 2))
                    x_end_mroi = sizes_mrois_pix[i_mroi][0]

                y_start_canvas = top_left_corners_pix[i_mroi][1]
                y_end_canvas = y_start_canvas + sizes_mrois_pix[i_mroi][1]

                plane_canvas[
                    :, x_start_canvas:x_end_canvas, y_start_canvas:y_end_canvas
                ] = planes_mrois[i_plane, i_mroi][:, x_start_mroi:x_end_mroi]

            shift_x_varied_seams = int(
                round(
                    (overlap_seams_this_plane - min(overlaps_planes))
                    * (n_mrois - 1)
                    / 2
                )
            )
            shift_x = accumulated_shifts[i_plane, 0] + shift_x_varied_seams
            shift_y = accumulated_shifts[i_plane, 1]

            end_x = shift_x + plane_canvas.shape[1]
            end_y = shift_y + plane_canvas.shape[2]

            volume[:, shift_x:end_x, shift_y:end_y, i_plane] = plane_canvas

        del planes_mrois
        # %% Calculate lateral offsets
        if current_pipeline_step == "make_template":
            # Calculate lateral offsets across planes and align planes
            # For the first file of each recording, we will calculate the lateral-shift vectors across planes
            # For that, we will do cross-correlation between mean-frame images from 2 adjacent planes
            # https://stackoverflow.com/questions/24768222/how-to-detect-a-shift-between-images
            for i_plane in range(n_planes - 1):
                # Calculate shifts
                im1_copy = copy.deepcopy(volume[0, :, :, i_plane])
                im2_copy = copy.deepcopy(volume[0, :, :, i_plane + 1])
                # Removing nans
                nonan_mask = np.stack(
                    (~np.isnan(im1_copy), ~np.isnan(im2_copy)), axis=0
                )
                nonan_mask = np.all(nonan_mask, axis=0)
                coord_nonan_pixels = np.where(nonan_mask)
                min_x, max_x = np.min(coord_nonan_pixels[0]), np.max(
                    coord_nonan_pixels[0]
                )
                min_y, max_y = np.min(coord_nonan_pixels[1]), np.max(
                    coord_nonan_pixels[1]
                )
                im1_nonan = im1_copy[min_x : max_x + 1, min_y : max_y + 1]
                im2_nonan = im2_copy[min_x : max_x + 1, min_y : max_y + 1]

                im1_nonan -= np.min(im1_nonan)
                im2_nonan -= np.min(im2_nonan)

                cross_corr_img = scipy.signal.fftconvolve(
                    im1_nonan, im2_nonan[::-1, ::-1], mode="same"
                )

                # Calculate vector
                corr_img_peak_x, corr_img_peak_y = np.unravel_index(
                    np.argmax(cross_corr_img), cross_corr_img.shape
                )
                self_corr_peak_x, self_corr_peak_y = [
                    dim / 2 for dim in cross_corr_img.shape
                ]
                interplane_shift = [
                    corr_img_peak_x - self_corr_peak_x,
                    corr_img_peak_y - self_corr_peak_y,
                ]

                interplane_shifts[i_plane] = copy.deepcopy(interplane_shift)
                accumulated_shifts[i_plane] = np.sum(
                    interplane_shifts, axis=0, dtype=int
                )

            min_accumulated_shift = np.min(accumulated_shifts, axis=0)
            for xy in range(2):
                accumulated_shifts[:, xy] -= min_accumulated_shift[xy]

            if params["json_logging"]:
                json_logger.info(
                    json.dumps({"accumulated_shifts": accumulated_shifts.tolist()})
                )
            continue

        # %% Select X,Y pixels that do not have nans for any plane
        if params["make_nonan_volume"]:
            if params["json_logging"]:
                json_logger.debug(
                    json.dumps({"debug_message": "Trimming volume to remove NaNs"})
                )
            volume_meanp = np.mean(volume[0], axis=2)
            non_nan = ~np.isnan(volume_meanp)
            coord_nonan_pixels = np.where(non_nan)
            min_x, max_x = (
                np.min(coord_nonan_pixels[0]),
                np.max(coord_nonan_pixels[0]) + 1,
            )
            min_y, max_y = (
                np.min(coord_nonan_pixels[1]),
                np.max(coord_nonan_pixels[1]) + 1,
            )
            volume = volume[:, min_x:max_x, min_y:max_y]
            if params["json_logging"]:
                json_logger.debug(
                    json.dumps(
                        {
                            "debug_message": "Shape of trimmed nonan volume: "
                            + str(volume.shape)
                        }
                    )
                )

        # %% Make volume non-negative
        if params["add_1000_for_nonegative_volume"]:
            if params["json_logging"]:
                json_logger.debug(
                    json.dumps({"debug_message": "Adding 1000 to make positive"})
                )
            volume += 1000

        # %% Convert volume from float to int
        if convert_volume_float32_to_int16:
            if volume.dtype != np.int16:
                if params["json_logging"]:
                    json_logger.debug(
                        json.dumps({"debug_message": "Transforming volume to int16"})
                    )
                volume = volume.astype(np.int16)

        volume = np.swapaxes(volume, 1, 2)

        # %% Saving outputs
        if current_pipeline_step == "reconstruct_all":
            if params["save_output"]:
                if params["json_logging"]:
                    json_logger.debug(
                        json.dumps({"debug_message": "Saving output file"})
                    )
                save_dir = os.path.dirname(path_input_file) + "/Preprocessed/"
                if params["save_as_volume_or_planes"] == "volume":
                    save_dir = os.path.dirname(path_input_file)
                    path_output_file = path_input_file[:-4] + "_preprocessed.h5"
                    h5file = h5py.File(path_output_file, "w")
                    h5file.create_dataset("mov", data=volume)
                    h5file.attrs.create(
                        "metadata", str(metadata)
                    )  # You can use json to load it as a dictionary
                    h5file.close()
                    del h5file
                elif params["save_as_volume_or_planes"] == "planes":
                    for i_plane in range(n_planes):
                        save_dir_this_plane = save_dir + "plane" f"{i_plane:02d}/"
                        if not os.path.isdir(save_dir_this_plane):
                            os.makedirs(save_dir_this_plane)
                        output_filename = os.path.basename(
                            path_input_file[:-4] + "_plane"
                            f"{i_plane:02d}_preprocessed.h5"
                        )
                        path_output_file = save_dir_this_plane + output_filename
                        h5file = h5py.File(path_output_file, "w")
                        h5file.create_dataset("mov", data=volume[:, :, :, i_plane])
                        h5file.attrs.create(
                            "metadata", str(metadata)
                        )  # You can use json to load it as a dictionary
                        h5file.close()
                        del h5file

                        if (
                            i_file == list_files_for_reconstruction[-1]
                            and params["concatenate_all_h5_to_tif"]
                        ):
                            files_to_concatenate = sorted(
                                glob.glob(save_dir_this_plane + "*preprocessed.h5")
                            )
                            data_to_concatenate = []
                            for this_file_to_concatenate in files_to_concatenate:
                                f = h5py.File(this_file_to_concatenate, "r")
                                data_to_concatenate.append(f["mov"])
                            data_to_concatenate = np.concatenate(
                                data_to_concatenate[:], axis=0
                            )
                            tifffile.imwrite(
                                save_dir_this_plane + "plane" f"{i_plane:02d}.tif",
                                data_to_concatenate,
                            )

                if params["json_logging"]:
                    json_logger.debug(json.dumps({"debug_message": "File(s) saved"}))

                # %% Save mean frame png
                if params["save_meanf_png"]:
                    if not params["meanf_png_only_first_file"] or i_file == 0:
                        if params["json_logging"]:
                            json_logger.debug(
                                json.dumps({"debug_message": "Saving png"})
                            )
                        canvas_png = np.zeros(
                            (
                                (volume.shape[1] + params["gaps_rows"]) * rows
                                - params["gaps_rows"],
                                (volume.shape[2] + params["gaps_columns"]) * columns
                                - params["gaps_columns"],
                            ),
                            dtype=np.uint8,
                        )
                        volume_meanf = np.nanmean(volume, axis=0)
                        for i_plane in range(n_planes):
                            plane_for_png = copy.deepcopy(volume_meanf[:, :, i_plane])
                            # Normalize to [0,1]
                            plane_for_png -= np.nanmin(plane_for_png)
                            plane_for_png = plane_for_png / np.nanmax(plane_for_png)

                            # Apply percentile-dynamic-range
                            pct_low, pct_high = np.nanpercentile(
                                plane_for_png, params["intensity_percentiles"]
                            )
                            plane_for_png = skimage.exposure.rescale_intensity(
                                plane_for_png, in_range=(pct_low, pct_high)
                            )

                            # Rescale and transform to uint8
                            plane_for_png = np.round(plane_for_png * 255)
                            plane_for_png = plane_for_png.astype(np.uint8)
                            # Place it on canvas
                            x_start = (
                                i_plane
                                % columns
                                * (plane_for_png.shape[1] + params["gaps_columns"])
                            )
                            y_start = (
                                i_plane
                                // columns
                                * (plane_for_png.shape[0] + params["gaps_rows"])
                            )
                            x_end = x_start + plane_for_png.shape[1]
                            y_end = y_start + plane_for_png.shape[0]
                            canvas_png[y_start:y_end, x_start:x_end] = plane_for_png
                        fig = plt.figure(dpi=1200)
                        plt.imshow(canvas_png, cmap="gray")
                        input_filename = os.path.basename(path_input_file)
                        plt.title(input_filename, fontsize=4)
                        plt.xticks(fontsize=4)
                        plt.yticks(fontsize=4)
                        fig.tight_layout()
                        plt.show()
                        output_filename_meanf_png = (
                            save_dir + input_filename[:-4] + ".png"
                        )
                        fig.savefig(output_filename_meanf_png, bbox_inches="tight")
                        del canvas_png, volume_meanf

                # %% Save mp4 clip for easy visual inspection
                if params["save_mp4"]:
                    if not params["video_only_first_file"] or i_file == 0:
                        if params["json_logging"]:
                            json_logger.debug(
                                json.dumps({"debug_message": "Saving mp4"})
                            )
                        metadata_software = metadata["Software"].split()
                        for i_line in range(len(metadata_software)):
                            this_line = metadata_software[i_line]
                            if "SI.hRoiManager.scanFrameRate" in this_line:
                                # The next line is the '=' and the next after that is the frame-rate
                                frame_rate = float(metadata_software[i_line + 2])
                        fps = frame_rate * params["video_play_speed"]
                        if params["video_duration_secs"] != 0:
                            use_until_frame_n = round(
                                fps * params["video_duration_secs"]
                            )  # -1 for entire recording
                        else:
                            use_until_frame_n = -1
                        canvas_video = np.zeros(
                            (
                                volume[:use_until_frame_n].shape[0]
                                - params["rolling_average_frames"]
                                + 1,
                                (volume.shape[1] + params["gaps_rows"]) * rows
                                - params["gaps_rows"],
                                (volume.shape[2] + params["gaps_columns"]) * columns
                                - params["gaps_columns"],
                            ),
                            dtype=np.uint8,
                        )
                        for i_plane in range(n_planes):
                            plane_for_video = copy.deepcopy(
                                volume[:use_until_frame_n, :, :, i_plane]
                            )
                            # Apply rolling average by convolving
                            plane_for_video = scipy.signal.convolve(
                                plane_for_video,
                                np.ones(([params["rolling_average_frames"], 1, 1])),
                                mode="valid",
                            )
                            # Apply percentile-dynamic-range
                            pct_low, pct_high = np.nanpercentile(
                                plane_for_video, params["intensity_percentiles"]
                            )
                            plane_for_video = skimage.exposure.rescale_intensity(
                                plane_for_video, in_range=(pct_low, pct_high)
                            )
                            # Normalize to [0,1]
                            plane_for_video -= np.nanmin(plane_for_video)
                            plane_for_video = plane_for_video / np.nanmax(
                                plane_for_video
                            )
                            # Rescale and transform to uint8
                            plane_for_video = np.round(plane_for_video * 255)
                            plane_for_video = plane_for_video.astype(np.uint8)
                            # Place it on canvas
                            x_start = (
                                i_plane
                                % columns
                                * (plane_for_video.shape[2] + params["gaps_columns"])
                            )
                            y_start = (
                                i_plane
                                // columns
                                * (plane_for_video.shape[1] + params["gaps_rows"])
                            )
                            x_end = x_start + plane_for_video.shape[2]
                            y_end = y_start + plane_for_video.shape[1]
                            canvas_video[
                                :, y_start:y_end, x_start:x_end
                            ] = plane_for_video
                        size_frame_video = (
                            canvas_video.shape[2],
                            canvas_video.shape[1],
                        )
                        input_filename = os.path.basename(path_input_file)
                        output_filename_video = (
                            save_dir
                            + input_filename[:-4]
                            + "_RollingAvg"
                            + str(params["rolling_average_frames"])
                            + "Frames_Speed"
                            + str(params["video_play_speed"])
                            + "x.mp4"
                        )
                        out = cv2.VideoWriter(
                            output_filename_video,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            size_frame_video,
                            False,
                        )
                        for f in range(canvas_video.shape[0]):
                            out.write(canvas_video[f])
                        out.release()
                        del canvas_video

        # %% This file is done

        toc = time.time()
        if params["json_logging"]:
            json_logger.debug(
                json.dumps(
                    {
                        "debug_message": "File processed and outputs saved. Time elapsed: "
                        + str(toc - tic)
                    }
                )
            )
        del volume
