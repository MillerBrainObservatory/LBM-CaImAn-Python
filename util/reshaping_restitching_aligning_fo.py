import copy
import glob
import json
import os
import time
from pathlib import Path
from pprint import pprint

import h5py
import numpy as np
import skimage
import tifffile
from icecream import ic
from matplotlib import pyplot as plt

import params  # Ensure this module is correctly imported


# Utility Functions
def ic_format_nparray(obj):
    """Format numpy array for debug statements."""
    return f"ndarray, shape={obj.shape}, dtype={obj.dtype}"


def init_params():
    """Initialize and return parameters."""
    return params.init_params()


def extract_mroi(filepath):
    """Process raw MROI data into a structured format."""
    tiffs = list(Path(filepath).glob("*.tif"))

    metadata = {}
    with tifffile.TiffFile(tiffs[0]) as tif:
        for tag in tif.pages[0].tags.values():
            tag_name, tag_value = tag.name, tag.value
            metadata[tag_name] = tag_value
    mrois_si_raw = json.loads(metadata["Artist"])["RoiGroups"]["imagingRoiGroup"]["rois"]
    if isinstance(mrois_si_raw, dict):
        mrois_si_raw = [mrois_si_raw]  # Ensure it's a list for uniform processing

    mrois_si = []
    for roi in mrois_si_raw:
        scanfield = roi["scanfields"][np.where(np.array(roi["zs"]) == 0)[0][0]] if isinstance(roi["scanfields"],
                                                                                              list) else roi[
            "scanfields"]
        mrois_si.append({
            "center": np.array(scanfield["centerXY"]),
            "sizeXY": np.array(scanfield["sizeXY"]),
            "pixXY": np.array(scanfield["pixelResolutionXY"])
        })

    mrois_centers_si = np.array([mroi_si["center"] for mroi_si in mrois_si])
    x_sorted = np.argsort(mrois_centers_si[:, 0])
    mrois_si_sorted_x = [mrois_si[i] for i in x_sorted]
    mrois_centers_si_sorted_x = [mrois_centers_si[i] for i in x_sorted]
    return metadata,mrois_si, mrois_si_sorted_x, mrois_centers_si_sorted_x


def set_vars(params):
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

    path_template_files = [path_all_files[idx] for idx in params["list_files_for_template"]]
    pipeline_steps = ["make_template"] if params["make_template_seams_and_plane_alignment"] else []
    pipeline_steps.append("reconstruct_all" if params["reconstruct_all_files"] else "")

    return path_template_files, path_all_files, params, pipeline_steps


def load_reshape(path_input_file, n_planes, params):
    """
    Main preprocessing function for loading data, preprocessing, and assembling MROI.

    Parameters
    ----------
    path_input_file : str
        Path to the input TIFF file.
    n_planes : int
        The number of planes in the TIFF file.
    params : dict
        A dictionary of parameters used for preprocessing.

    Returns
    -------
    np.ndarray
        The preprocessed data array, organized and ready for further processing.
    """

    # Read TIFF data
    tiff_data = tifffile.imread(path_input_file)

    # Assemble MROI
    mrois_si, mrois_centers_si_sorted_x, _, mrois_si_sorted_x, _, _ = extract_mroi(path_input_file)
    reshaped_data = np.reshape(tiff_data,
                               (tiff_data.shape[0] // n_planes, n_planes, tiff_data.shape[1], tiff_data.shape[2]))

    return reshaped_data


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
        tiff_file = np.reshape(tiff_file,
                               (int(tiff_file.shape[0] / n_planes), n_planes, tiff_file.shape[1], tiff_file.shape[2]),
                               order='A')  # warnings are expected if the recording is split into many files or incomplete
    else:
        tiff_file = np.expand_dims(tiff_file, 1)
    tiff_file = np.swapaxes(tiff_file, 1, 3)
    return tiff_file

def locate_mroi(planes_mrois, mrois_si_sorted_x, mrois_centers_si_sorted_x):
    # Get pixel sizes
    n_mrois = 5
    sizes_mrois_pix = np.array([mroi_pix.shape[1:] for mroi_pix in planes_mrois[0, :]])
    sizes_mrois_si = np.array([mroi_si['sizeXY'] for mroi_si in mrois_si_sorted_x])
    pixel_sizes = sizes_mrois_si / sizes_mrois_pix
    psize_x, psize_y = np.mean(pixel_sizes[:, 0]), np.mean(pixel_sizes[:, 1])
    assert np.product(np.isclose(pixel_sizes[:, 1] - psize_y, 0)), "Y-pixels resolution not uniform across MROIs"
    assert np.product(np.isclose(pixel_sizes[:, 0] - psize_x, 0)), "X-pixels resolution not uniform across MROIs"
    # assert np.product(np.isclose(pixel_sizes[:,0]-pixel_sizes[:,1], 0)), "Pixels do not have squared resolution"

    # Calculate the pixel ranges (with their SI locations) that would fit all MROIs
    top_left_corners_si = mrois_centers_si_sorted_x - sizes_mrois_si / 2
    bottom_right_corners_si = mrois_centers_si_sorted_x + sizes_mrois_si / 2
    xmin_si, ymin_si = top_left_corners_si[:, 0].min(), top_left_corners_si[:, 1].min()
    xmax_si, ymax_si = bottom_right_corners_si[:, 0].max(), bottom_right_corners_si[:, 1].max()
    reconstructed_xy_ranges_si = [np.arange(xmin_si, xmax_si, psize_x), np.arange(ymin_si, ymax_si, psize_y)]

    # Calculate the starting pixel for each MROI when placed in the reconstructed movie
    top_left_corners_pix = np.empty((n_mrois, 2), dtype=int)
    for i_xy in range(2):
        for i_mroi in range(n_mrois):
            closest_xy_pix = np.argmin(np.abs(reconstructed_xy_ranges_si[i_xy] - top_left_corners_si[i_mroi, i_xy]))
            top_left_corners_pix[i_mroi, i_xy] = int(closest_xy_pix)
            closest_xy_si = reconstructed_xy_ranges_si[i_xy][closest_xy_pix]
            if not np.isclose(closest_xy_si, top_left_corners_si[i_mroi, i_xy]):
                print("BAD RECONSTRUCTION")
                raise NotImplementedError("No way to deal with misaligned mroi coordinates")
    for i_xy in range(2):
        if len(reconstructed_xy_ranges_si[i_xy]) == np.sum(sizes_mrois_pix[:, 0]) + 1:
            reconstructed_xy_ranges_si[i_xy] = reconstructed_xy_ranges_si[i_xy][:-1]

    return reconstructed_xy_ranges_si

def calculate_overlap(n_mrois, n_planes, planes_mrois, params):

    # Determine if all the MROIs are adjacent
    for i_mroi in range(n_mrois - 1):
        if top_left_corners_pix[i_mroi][0] + sizes_mrois_pix[i_mroi][0] != \
                top_left_corners_pix[i_mroi + 1][0]:
            raise Exception('MROIs number ' + str(i_mroi) + ' and number ' + str(
                i_mroi + 1) + ' (0-based idx) are not contiguous')

    # Combine meanf from differete template files:
    overlaps_planes_seams_scores = np.zeros((n_planes, n_mrois - 1, params['max_seam_overlap'] - params[
        'min_seam_overlap']))  # We will avoid i_overlaps = 0

    for i_plane in range(n_planes):
        for i_seam in range(n_mrois - 1):
            for i_overlaps in range(params['min_seam_overlap'], params['max_seam_overlap']):
                strip_left = planes_mrois[i_plane, i_seam][0,
                             -params['n_ignored_pixels_sides'] - i_overlaps:-params[
                                 'n_ignored_pixels_sides']]
                strip_right = planes_mrois[i_plane, i_seam + 1][0,
                              params['n_ignored_pixels_sides']:i_overlaps + params[
                                  'n_ignored_pixels_sides']]
                subtract_left_right = abs(strip_left - strip_right)
                overlaps_planes_seams_scores[
                    i_plane, i_seam, i_overlaps - params['min_seam_overlap']] = np.mean(
                    subtract_left_right)

    overlaps_planes_scores = np.mean(overlaps_planes_seams_scores, axis=(1))
    overlaps_planes = []
    for i_plane in range(n_planes):
        overlaps_planes.append(
            int(np.argmin(overlaps_planes_scores[i_plane]) + params['min_seam_overlap'] + 2 * params[
                'n_ignored_pixels_sides']))

    # Plot the scores for the different planes and also potential shifts
    if params['alignment_plot_checks']:

        for i_plane in range(n_planes):
            plt.plot(range(params['min_seam_overlap'], params['max_seam_overlap']),
                     overlaps_planes_scores[i_plane])
        plt.title('Score for all planes')
        plt.xlabel('Overlap (pixels)')
        plt.ylabel('Error (a.u.)')
        plt.show()

        for i_plane in range(n_planes):
            for shift in range(-2, 3):
                i_overlap = overlaps_planes[i_plane] + shift
                canvas_alignment_check = np.zeros(
                    (len(reconstructed_xy_ranges_si[0]) - (n_mrois - 1) * i_overlap,
                     len(reconstructed_xy_ranges_si[1]),
                     3), dtype=np.float32)
                x_start = 0
                for i_mroi in range(n_mrois):
                    x_start = top_left_corners_pix[i_mroi][0] - i_mroi * i_overlap
                    x_end = x_start + sizes_mrois_pix[i_mroi][0]
                    y_start = top_left_corners_pix[i_mroi][1]
                    y_end = y_start + sizes_mrois_pix[i_mroi][1]
                    canvas_alignment_check[x_start:x_end, y_start:y_end, i_mroi % 2] = planes_mrois[
                                                                                           0, i_plane, i_mroi] - np.min(
                        planes_mrois[0, i_plane, i_mroi])

                pct_low, pct_high = np.nanpercentile(canvas_alignment_check, [80,
                                                                              99.9])  # Consider that we are using 1/3 of pixels (RGB channels)
                canvas_alignment_check = skimage.exposure.rescale_intensity(canvas_alignment_check,
                                                                            in_range=(
                                                                                pct_low, pct_high))
                plt.imshow(np.swapaxes(canvas_alignment_check, 0, 1))
                plt.title('Plane: ' + str(i_plane))
                plt.xlabel(
                    'Original overlap: ' + str(overlaps_planes[i_plane]) + ' + Shift: ' + str(shift))
                plt.show()


def main():
    n_planes = 30

    parameters = init_params()
    path_input_files = parameters['raw_data_dirs'][0]
    metadata, mrois_si, mrois_si_sorted_x, mrois_centers_si_sorted_x = extract_mroi(path_input_files)

    path_template_files, path_all_files, params, pipeline_steps = set_vars(parameters)
    for current_pipeline_step in pipeline_steps:

        pprint(current_pipeline_step)
        path_input_files = path_template_files if current_pipeline_step == "make_template" else path_all_files
        list_files_for_reconstruction = range(parameters["reconstruct_until_this_ifile"]) if not parameters[
            "reconstruct_all_files"] else range(len(path_input_files))

        for i_file in list_files_for_reconstruction:
            tic = time.time()
            path_input_file = path_input_files[i_file]

            tiff = load_tiff(path_input_file, n_planes)

            if current_pipeline_step == 'make_template':
                tiff_file = np.mean(tiff_file, axis=0, keepdims=True)

            # %% Separate tif into MROIs
            # Get the Y coordinates for mrois (and not flybacks)
            if i_file == 0:
                n_mrois = len(mrois_si)
                tif_pixels_Y = tiff.shape[2]
                mrois_pixels_Y = np.array([mroi_si['pixXY'][1] for mroi_si in mrois_si])
                each_flyback_pixels_Y = (tif_pixels_Y - mrois_pixels_Y.sum()) // (n_mrois - 1)

            # Divide long stripe into mrois
            planes_mrois = np.empty((n_planes, n_mrois), dtype=np.ndarray)
            for i_plane in range(n_planes):
                y_start = 0
                for i_mroi in range(n_mrois):  # We go over the order in which they were acquired
                    planes_mrois[i_plane, i_mroi] = tiff_file[:, :, y_start:y_start + mrois_pixels_Y[x_sorted[i_mroi]],
                                                    i_plane]
                    y_start += mrois_pixels_Y[i_mroi] + each_flyback_pixels_Y

            del tiff_file

            if current_pipeline_step == 'make_template':
                n_template_files = params['list_files_for_template']
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

            reconstructed_xy_ranges_si = locate_mroi(planes_mrois, mrois_si_sorted_x, mrois_centers_si_sorted_x)
            if current_pipeline_step == 'make_template':
                if params['seams_overlap'] == 'calculate':
                    overlap = calculate_overlap()
                elif type(params['seams_overlap']) is int:
                    overlaps_planes = [params['seams_overlap']] * n_planes
                elif params['seams_overlap'] is list:
                    overlaps_planes = params['seams_overlap']
            else:
                raise Exception(
                    'params[\'seams_overlap\'] should be set to \'calculate\', an integer, or a list of length n_planes')

            save_outputs(i_file, volume, path_input_file, metadata, n_planes, parameters, path_input_files)
            toc = time.time()
            print(f"Processing time for file {i_file}: {toc - tic} seconds")


if __name__ == "__main__":
    main()
