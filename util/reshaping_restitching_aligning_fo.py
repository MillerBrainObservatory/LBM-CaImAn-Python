import copy
import glob
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import tifffile
from icecream import ic
from matplotlib import pyplot as plt

from params import init_params


def ic_format_nparray(obj):
    """Format numpy array for debug statements."""
    return f"ndarray, shape={obj.shape}, dtype={obj.dtype}"


def extract_scanimage_metadata(filepath):
    """
    Extracts and structures ScanImage Region of Interest (ROI) data from TIFF files.

    Parameters
    ----------
    filepath : str
        Path to the directory containing TIFF files.

    Returns
    -------
    metadata : dict
        Extracted metadata from the TIFF file.
    rois : list
        List of dictionaries, each representing a ROI with keys for center, size, and pixel resolution.
    rois_sorted_x : list
        ROIs sorted by their x-coordinate center.
    centers_sorted_x : list
        Centers of ROIs sorted by x-coordinate.
    x_sort_indices : ndarray
        Indices that would sort the ROIs by their x-coordinate center.

    Notes
    -----
    Assumes that the first TIFF file in the directory contains the necessary metadata in the 'Artist' tag, which is
    then processed to extract ROI information. Only ROIs at z=0 are considered.
    """
    tiff_files = list(Path(filepath).glob("*.tif"))

    # Extract metadata from the first TIFF file
    with tifffile.TiffFile(tiff_files[0]) as tif:
        metadata = {tag.name: tag.value for tag in tif.pages[0].tags.values()}

    # The 'Artist' tag is where ScanImage stores the images
    rois_raw = json.loads(metadata["Artist"])["RoiGroups"]["imagingRoiGroup"]["rois"]
    # The order in which these ROI's are extracted is important.
    # We retain that information within an array of indexes.
    rois = []
    for roi in rois_raw:
        #TODO: WHat condition leads to scanfields being saved as non-list/non-iterable types
        if type(roi["scanfields"]) != list:
            scanfield = roi["scanfields"]
        else:
            scanfield = roi["scanfields"][
                np.where(np.array(roi["zs"]) == 0)[0][0]
            ]
        rois.append({
            "center": np.array(scanfield["centerXY"]),
            "sizeXY": np.array(scanfield["sizeXY"]),
            "pixXY": np.array(scanfield["pixelResolutionXY"])
        })

    # Sort ROIs by x-coordinate of their center
    centers = np.array([roi["center"] for roi in rois])
    x_sort_indices = np.argsort(centers[:, 0])
    rois_sorted_x = [rois[i] for i in x_sort_indices]
    centers_sorted_x = [centers[i] for i in x_sort_indices]

    return metadata, rois, rois_sorted_x, centers_sorted_x, x_sort_indices


def set_params(params):
    """
    Set variables based on parameters.

    Parameters
    ----------
    params : dict
        A dictionary containing parameters for the script, including debugging flags,
        output preferences, and file processing options.

    Returns
    -------
    tuple
        A tuple containing configured variables including paths for template files,
        all files to process, number of template files, flags for initializing volumes,
        and pipeline steps to execute.
    """
    if params['debug']:
        ic.enable()
        ic.configureOutput(prefix='Debugger -> ', includeContext=True, contextAbsPath=True)
    else:
        ic.disable()

    path_all_files = []
    for i_dir in params["raw_data_dirs"]:
        path_all_files.extend(sorted(glob.glob(f"{i_dir}/**/*.tif", recursive=True)))

    if not params["lateral_align_planes"]:  # If no need to align planes, we can use int16
        # TODO: Is this param ever false?
        initialize_volume_with_nans = False
        convert_volume_float32_to_int16 = True
        # It is going to be no-nan by definition, no need to check for it
        params["make_nonan_volume"] = False
    elif params["make_nonan_volume"]:
        initialize_volume_with_nans = True
        convert_volume_float32_to_int16 = True
    else:
        initialize_volume_with_nans = True
        convert_volume_float32_to_int16 = False
    path_template_files = [path_all_files[idx] for idx in params["list_files_for_template"]]
    pipeline_steps = ["make_template"] if params["make_template_seams_and_plane_alignment"] else []
    pipeline_steps.append("reconstruct_all" if params["reconstruct_all_files"] else "")

    return path_template_files, path_all_files, params, pipeline_steps, initialize_volume_with_nans, convert_volume_float32_to_int16


def save_outputs(i_file, volume, path_input_file, metadata, n_planes, params, file_list):
    """
    Save processed data in various formats based on parameters.

    Parameters
    ----------
    volume : np.ndarray
        The volume data to be saved.
    path_input_file : str
        The path to the input file, used to derive output file names.
    metadata : dict
        Metadata associated with the input file, used for saving additional info.
    n_planes : int
        The number of planes in the volume data.
    params : dict
        Parameters dictating the output formats and configurations.

    Returns
    -------
    None
    """
    # Implementation for saving output in different formats based on `params`
    # Example for saving an H5 file:
    if params["save_as_volume_or_planes"] == "volume":
        output_file_path = path_input_file.replace('.tif', '_processed.h5')
        with h5py.File(output_file_path, 'w') as h5file:
            h5file.create_dataset('data', data=volume)
            h5file.attrs['metadata'] = json.dumps(metadata)

    save_dir = os.path.dirname(path_input_file) + '/Preprocessed_2f/'
    if params['save_as_volume_or_planes'] == 'volume':
        save_dir = os.path.dirname(path_input_file)
        path_output_file = path_input_file[:-4] + '_preprocessed.h5'
        h5file = h5py.File(path_output_file, 'w')
        h5file.create_dataset('mov', data=volume)
        h5file.attrs.create('metadata', str(metadata))  # You can use json to load it as a dictionary
        h5file.close()
        del h5file
    elif params['save_as_volume_or_planes'] == 'planes':
        for i_plane in range(n_planes):
            save_dir_this_plane = save_dir + 'plane'f'{i_plane:02d}/'
            if not os.path.isdir(save_dir_this_plane):
                os.makedirs(save_dir_this_plane)
            output_filename = os.path.basename(path_input_file[:-4] + '_plane'f'{i_plane:02d}_preprocessed.h5')
            path_output_file = save_dir_this_plane + output_filename
            h5file = h5py.File(path_output_file, 'w')
            h5file.create_dataset('mov', data=volume[:, :, :, i_plane])
            h5file.attrs.create('metadata', str(metadata))  # You can use json to load it as a dictionary
            h5file.close()
            del h5file

            if i_file == file_list[-1] and params['concatenate_all_h5_to_tif']:
                files_to_concatenate = sorted(glob.glob(save_dir_this_plane + '*preprocessed.h5'))
                data_to_concatenate = []
                for this_file_to_concatenate in files_to_concatenate:
                    f = h5py.File(this_file_to_concatenate, 'r')
                    data_to_concatenate.append(f['mov'])
                data_to_concatenate = np.concatenate(data_to_concatenate[:], axis=0)
                tifffile.imwrite(save_dir_this_plane + 'plane'f'{i_plane:02d}.tif', data_to_concatenate)


def load_tiff(path_input_file, n_planes):
    tiff_file = tifffile.imread(path_input_file)
    if n_planes > 1:
        # warnings are expected if the recording is split into many files or incomplete
        tiff_file = np.reshape(tiff_file,
                               (int(tiff_file.shape[0] / n_planes),
                                n_planes,
                                tiff_file.shape[1],
                                tiff_file.shape[2]),
                               order='A'
                               )
    else:
        tiff_file = np.expand_dims(tiff_file, 1)
    tiff_file = np.swapaxes(tiff_file, 1, 3)
    return tiff_file


def locate_mroi(planes_mrois, mrois_si_sorted_x, mrois_centers_si_sorted_x):
    """
    Calculates the spatial arrangement of Multi-Regions of Interest (MROIs) based on their pixel sizes
    and spatial coordinates. This function determines the pixel sizes for all MROIs and computes the
    reconstructed spatial ranges that fit all MROIs. It also calculates the starting pixel coordinates for
    each MROI in the reconstructed image space.

    Parameters
    ----------
    planes_mrois : np.ndarray
        An array of MROIs with shape (n_planes, n_mrois) containing imaging data for each MROI.
    mrois_si_sorted_x : list
        A list of dictionaries, each representing an MROI with spatial information, sorted by the x-coordinate.
    mrois_centers_si_sorted_x : np.ndarray
        An array of MROI center coordinates sorted by the x-coordinate.

    Returns
    -------
    reconstructed_xy_ranges_si : list of np.ndarray
        The reconstructed spatial ranges in SI units that fit all MROIs, in both x and y dimensions.
    top_left_corners_si : np.ndarray
        The top-left corner coordinates of each MROI in spatial coordinates.
    top_left_corners_pix : np.ndarray
        The starting pixel coordinates for each MROI in the reconstructed image.
    sizes_mrois_pix : np.ndarray
        The size of each MROI in pixels.
    sizes_mrois_si : np.ndarray
        The size of each MROI in spatial coordinates (SI units).

    Raises
    ------
    AssertionError
        If the resolution in pixels is not uniform across all MROIs either in the x or y dimensions.
    NotImplementedError
        If the function encounters misaligned MROI coordinates that it cannot reconcile.
    """
    n_mrois = len(mrois_si_sorted_x)
    sizes_mrois_pix = np.array([mroi_pix.shape[1:] for mroi_pix in planes_mrois[0, :]])
    sizes_mrois_si = np.array([mroi_si['sizeXY'] for mroi_si in mrois_si_sorted_x])
    pixel_sizes = sizes_mrois_si / sizes_mrois_pix
    psize_x, psize_y = np.mean(pixel_sizes[:, 0]), np.mean(pixel_sizes[:, 1])

    # Ensuring uniform pixel resolution across MROIs
    assert np.all(np.isclose(pixel_sizes[:, 1], psize_y)), "Y-pixels resolution not uniform across MROIs"
    assert np.all(np.isclose(pixel_sizes[:, 0], psize_x)), "X-pixels resolution not uniform across MROIs"

    # Calculating spatial ranges for reconstructed image
    top_left_corners_si = mrois_centers_si_sorted_x - sizes_mrois_si / 2
    bottom_right_corners_si = mrois_centers_si_sorted_x + sizes_mrois_si / 2
    xmin_si, ymin_si = np.min(top_left_corners_si, axis=0)
    xmax_si, ymax_si = np.max(bottom_right_corners_si, axis=0)
    reconstructed_xy_ranges_si = [np.arange(xmin_si, xmax_si, psize_x), np.arange(ymin_si, ymax_si, psize_y)]

    # Determining starting pixel for each MROI
    top_left_corners_pix = np.empty((n_mrois, 2), dtype=int)
    for i_xy in range(2):
        for i_mroi in range(n_mrois):
            closest_xy_pix = np.argmin(np.abs(reconstructed_xy_ranges_si[i_xy] - top_left_corners_si[i_mroi, i_xy]))
            top_left_corners_pix[i_mroi, i_xy] = closest_xy_pix
            closest_xy_si = reconstructed_xy_ranges_si[i_xy][closest_xy_pix]
            if not np.isclose(closest_xy_si, top_left_corners_si[i_mroi, i_xy]):
                raise NotImplementedError("Misaligned MROI coordinates cannot be reconciled.")

    # Adjusting ranges if necessary
    for i_xy in range(2):
        if len(reconstructed_xy_ranges_si[i_xy]) == np.sum(sizes_mrois_pix[:, i_xy]) + 1:
            reconstructed_xy_ranges_si[i_xy] = reconstructed_xy_ranges_si[i_xy][:-1]

    return reconstructed_xy_ranges_si, top_left_corners_si, top_left_corners_pix, sizes_mrois_pix, sizes_mrois_si


def calculate_overlap(n_mrois, n_planes, planes_mrois, params_dict, top_left_corners_pix, sizes_mrois_pix):
    """
    Calculates the optimal overlap for seams between adjacent Regions of Interest (mROIs) for each plane.
    This function evaluates the difference in pixel intensities at the seams and determines the optimal overlap
    to minimize this difference, facilitating a smoother transition between MROIs.

    Parameters
    ----------
    n_mrois : int
        The number of MROIs.
    n_planes : int
        The number of planes in the dataset.
    planes_mrois : np.ndarray
        Array containing the imaging data for each MROI, structured as (n_planes, n_mrois).
    params_dict : dict
        Dictionary containing parameters for the overlap calculation, including 'min_seam_overlap',
        'max_seam_overlap', 'n_ignored_pixels_sides', and 'alignment_plot_checks'.
    top_left_corners_pix : np.ndarray
        Array of starting pixel coordinates for each MROI in the reconstructed image.
    sizes_mrois_pix : np.ndarray
        Array of sizes for each MROI in pixels.

    Returns
    -------
    overlaps_planes : list
        A list containing the calculated optimal overlap for each plane.

    Raises
    ------
    Exception
        If adjacent MROIs are not contiguous, indicating a gap or overlap in their arrangement.
    """

    # Verify adjacency of MROIs
    for i_mroi in range(n_mrois - 1):
        if top_left_corners_pix[i_mroi][0] + sizes_mrois_pix[i_mroi][0] != top_left_corners_pix[i_mroi + 1][0]:
            raise Exception(f'MROIs number {i_mroi} and number {i_mroi + 1} (0-based idx) are not contiguous')

    overlaps_planes_seams_scores = np.zeros(
        (n_planes, n_mrois - 1, params_dict['max_seam_overlap'] - params_dict['min_seam_overlap']))
    for i_plane in range(n_planes):
        for i_seam in range(n_mrois - 1):
            for i_overlaps in range(params_dict['min_seam_overlap'], params_dict['max_seam_overlap']):
                strip_left = planes_mrois[i_plane, i_seam][0,
                             -params_dict['n_ignored_pixels_sides'] - i_overlaps:-params_dict['n_ignored_pixels_sides']]
                strip_right = planes_mrois[i_plane, i_seam + 1][0,
                              params_dict['n_ignored_pixels_sides']:i_overlaps + params_dict['n_ignored_pixels_sides']]
                subtract_left_right = abs(strip_left - strip_right)
                overlaps_planes_seams_scores[i_plane, i_seam, i_overlaps - params_dict['min_seam_overlap']] = np.mean(
                    subtract_left_right)

    overlaps_planes_scores = np.mean(overlaps_planes_seams_scores, axis=(1))
    overlaps_planes = [
        int(np.argmin(overlaps_planes_scores[i_plane]) + params_dict['min_seam_overlap'] + 2 * params_dict[
            'n_ignored_pixels_sides']) for i_plane in range(n_planes)]

    # Optionally visualize the alignment and overlap scoring
    if params_dict.get('alignment_plot_checks', False):
        for i_plane in range(n_planes):
            plt.figure(figsize=(10, 4))
            plt.plot(range(params_dict['min_seam_overlap'], params_dict['max_seam_overlap']),
                     overlaps_planes_scores[i_plane], label=f'Plane {i_plane}')
            plt.title('Overlap Score per Plane')
            plt.xlabel('Overlap (pixels)')
            plt.ylabel('Average Difference')
            plt.legend()
            plt.show()

    return overlaps_planes


def main():
    n_planes = 30

    parameters = init_params()
    path_input_files = parameters['raw_data_dirs'][0]
    metadata, mrois_si, mrois_si_sorted_x, mrois_centers_si_sorted_x, x_sorted = extract_scanimage_metadata(
        path_input_files)

    path_template_files, path_all_files, params, pipeline_steps, initialize_volume_with_nans, convert_volume_float32_to_int16 = set_params(
        parameters)
    for current_pipeline_step in pipeline_steps:

        # Gather input files.
        # Template files are created on the first pass "make_template"
        # You must have either "recon. all files" or "until this file" set as parameters
        path_input_files = path_template_files if current_pipeline_step == "make_template" else path_all_files
        list_files_for_reconstruction = range(parameters["reconstruct_until_this_ifile"]) if not parameters[
            "reconstruct_all_files"] else range(len(path_input_files))

        for i_file in list_files_for_reconstruction:
            tic = time.time()
            path_input_file = path_input_files[i_file]

            tiff = load_tiff(path_input_file, n_planes)

            if current_pipeline_step == 'make_template':
                tiff_file = np.mean(tiff, axis=0, keepdims=True)

            # %% Separate tif into MROIs
            # Get the Y coordinates for mrois (and not flybacks)

            if i_file == 0:

                # Calculate amount of flyback pixels to delete on each scan

                # if each XY image is (144, 1000)
                # we should have a total of 1000 pixels for each ROI, i.e. 5000 pixels
                # but the scanner flyback adds extra lines, for the case of 5000 px, our number of pixels is 5104
                # Calculate how many px we need to eliminate for each scanner flyback
                n_mrois = len(mrois_si)
                tif_pixels_Y = tiff.shape[2]  # this will be > num_pixels_y * n_roi
                mrois_pixels_Y = np.array([mroi_si['pixXY'][1] for mroi_si in mrois_si])

                each_flyback_pixels_Y = (tif_pixels_Y - mrois_pixels_Y.sum()) // (n_mrois - 1)

            # Divide long stripe into mrois
            planes_mrois = np.empty((n_planes, n_mrois), dtype=np.ndarray)
            for i_plane in range(n_planes):
                y_start = 0
                # We go over the order in which they were acquired
                for i_mroi in range(n_mrois):
                    planes_mrois[i_plane, i_mroi] = tiff[:, :, y_start:y_start + mrois_pixels_Y[x_sorted[i_mroi]], i_plane]
                    y_start += mrois_pixels_Y[i_mroi] + each_flyback_pixels_Y

            del tiff

            if current_pipeline_step == 'make_template':
                n_template_files = params['list_files_for_template']
                if len(n_template_files) > 1:
                    if i_file == 0:
                        template_accumulator = copy.deepcopy(planes_mrois)
                        continue
                    elif i_file != n_template_files - 1:
                        template_accumulator += planes_mrois
                        continue
                    else:
                        template_accumulator += planes_mrois
                        planes_mrois = template_accumulator / n_template_files

            # LOCATE MROIS
            reconstructed_xy_ranges_si, top_left_corners_si, top_left_corners_pix, sizes_mrois_pix, sizes_mrois_si = locate_mroi(planes_mrois, mrois_si_sorted_x, mrois_centers_si_sorted_x)

            if current_pipeline_step == 'make_template':
                if params['seams_overlap'] == 'calculate':
                    overlap_planes = calculate_overlap(n_mrois, len(planes_mrois),planes_mrois, params, top_left_corners_pix, sizes_mrois_pix)
                elif type(params['seams_overlap']) is int:
                    overlaps_planes = [params['seams_overlap']] * n_planes
                elif params['seams_overlap'] is list:
                    overlaps_planes = params['seams_overlap']
            else:
                raise Exception(
                    'params[\'seams_overlap\'] should be set to \'calculate\', an integer, or a list of length n_planes')

            # %% Create a volume container
            if (current_pipeline_step == "make_template"):  # For templating
                # MROIs, we will get here when working on the last file
                n_f = 1
            elif current_pipeline_step == "reconstruct_all":
                n_f = n_f = planes_mrois[0, 0].shape[0]

                # For template or if no need to align planes, initialize interplane shifts as 0s
            if (current_pipeline_step == "make_template" or not params["lateral_align_planes"]):
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

            print("Creating volume of shape: ")
            ic(str([n_f, n_x, n_y, n_y]))
            print(" (f,x,y,z)")

            if initialize_volume_with_nans:
                volume = np.full((n_f, n_x, n_y, n_z), np.nan, dtype=np.float32)
            else:
                volume = np.empty((n_f, n_x, n_y, n_z), dtype=np.int16)
            x = 5
            # save_outputs(i_file, volume, path_input_file, metadata, n_planes, parameters, path_input_files)
            # toc = time.time()
            # print(f"Processing time for file {i_file}: {toc - tic} seconds")


if __name__ == "__main__":
    main()
