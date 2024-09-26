"""
This module contains a function that extracts multi-ROI (Region of Interest) data from a TIFF image

Structure (ScanImage ROIDataSimple:

roi_data - len(roi_data) = number of ROIs
    - RoiData Object (class)
    - zs (z stack)
    - channels (channels, the same for each ROI)
    - imageData (container)
    -   len(imageData) = number of planes (z)
        - len(imageData[0]) = number of frames/volumes (time)
            - len(imageData[0][0]) = number of "slices", but this will always be 1 for us
                - imageData[0][0][0] = the actual 2D cross-section of the image
"""
import logging
import os
from pathlib import Path

import numpy as np
import tifffile

from util.roi_data_simple import RoiDataSimple

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_mroi_data_from_tiff(
    metadata_json: dict,
    metadata_kv: dict,
    data: np.ndarray,
    num_channels: int,
    num_volumes: int,
    num_rois: int,
):
    """
    Extracts multi-ROI (Region of Interest) data from a TIFF image based on metadata.

    This function emulates the behavior of the MATLAB function `ScanImage.util.getMRoiFromTiff.m`.
    Given the metadata and the image data, this function processes and extracts the regions
    of interest from the image data.

    ScanImageTiffReader, the Python binding for ScanImage, does not provide a way to extract and collate the ROI data

    Parameters
    ----------
    metadata_json : dict
        JSON representation of the metadata.

    metadata_kv : dict
        Key-value pairs of specific metadata attributes.
    data : numpy.ndarray
        The 4D numpy array containing image data with dimensions corresponding to [height, width, channel, volume].
    num_channels : int
        Number of channels in the image data.
    num_volumes : int
        Number of volumes (or timepoints) in the image data.
    num_rois : int
        Number of Regions of Interest.

    Returns
    -------
    roi_data : dict
        A dictionary where keys are ROI indices and values are instances of RoiDataSimple,
        which store the extracted data for each ROI.
    roi_group : dict
        A dictionary containing the metadata for the ROI group.


    Notes
    -----
    The function currently assumes a certain structure for the `metadata_json` and `metadata_kv`
    arguments, as well as for the ROI data. The exact structure and any discrepancies between
    MATLAB and Python implementations should be taken into account before using this function.

    Some conditions in the function, especially those handling the 'zs' (z-stack) dimension,
    have been optimized for the current use case. Future use cases with different z-stack
    configurations might need additional adjustments.

    Example:
        >>> metadata = {"RoiGroups": {"imagingRoiGroup": ...}}
        >>> metadata_keyvals = {"SI.hScan2D.flytoTimePerScanfield": ...}
        >>> image_data = np.random.rand(512, 512, 2, 10)
        >>> rois, group = get_mroi_data_from_tiff(metadata_keyvals, metadata_kv, image_data, 2, 10, 5)
    """
    numLinesBetweenScanfields = np.round(
        metadata_kv["SI.hScan2D.flnytoTimePerScanfield"]
        / float(metadata_kv["SI.hRoiManager.linePeriod"]),
        0,
    )

    #  For us, always 0 ? Correspondes to number of slices in the stack
    stackZsAvailable = metadata_kv["SI.hStackManager.zs"]
    if isinstance(stackZsAvailable, (int, float)):
        stackZsAvailable = 1
    else:
        lenRoiZs = len(stackZsAvailable)

    # So not worrying about it now, heres the matlab translation nonetheless:
    # roi_info = np.zeros((num_rois, stackZsAvailable))
    roi_info = np.zeros((num_rois,))
    roi_img_height_info = np.zeros((num_rois, stackZsAvailable))

    roi_data = {}
    roi_group = metadata_json["RoiGroups"]["imagingRoiGroup"]
    for i in range(num_rois):
        rdata = RoiDataSimple()
        rdata.hRoi = roi_group["rois"][i]
        rdata.channels = metadata_kv["SI.hChannels.channelSave"]

        if isinstance(rdata.hRoi["zs"], (int, float)):
            lenRoiZs = 1
        else:
            lenRoiZs = len(rdata.hRoi["zs"])
        zsHasRoi = np.zeros_like(
            stackZsAvailable, dtype=int
        )  # This should likely go outside the for-loop

        # We can probably eliminate the else case to this if statement
        # ScanImage has other cases that don't pertain to us... keeping here for now
        if lenRoiZs == 1:
            # We can also probably eliminate this if statement
            if rdata.hRoi["discretePlaneMode"]:
                zsHasRoi = (stackZsAvailable == rdata.hRoi["zs"][0]).astype(int)
                roi_img_height_info[i, np.where(zsHasRoi == 1)] = rdata.hRoi[
                    "scanfields"
                ]["pixelResolutionXY"][1]
            else:
                # The roi extends from -Inf to Inf
                zsHasRoi = np.ones_like(stackZsAvailable, dtype=int)
                # The height doesn't change for the case of single-scanfields
                roi_img_height_info[i, :] = rdata.hRoi["scanfields"][
                    "pixelResolutionXY"
                ][1]
        else:
            minVal = rdata.hRoi["zs"][0]
            maxVal = rdata.hRoi["zs"][-1]
            idxRange = np.where(
                (stackZsAvailable >= minVal) & (stackZsAvailable <= maxVal)
            )[0]

            for j in idxRange:
                s = j
                sf = rdata.hRoi.get(stackZsAvailable[s])
                if sf:
                    roi_img_height_info[i, s] = sf["pixelResolution"][1]
                    zsHasRoi[s] = 1

        # We only need to fill in one row, this likely won't work for multiple zs
        roi_info[:] = zsHasRoi
        try:
            rdata.zs = np.array(stackZsAvailable)[np.where(zsHasRoi == 1)]
        except IndexError:
            rdata.zs = 0
        roi_data[i] = rdata

    for curr_channel in range(num_channels):
        logger.debug(f"Processing channel {curr_channel}")

        for curr_volume in range(num_volumes):
            roi_image_cnt = np.zeros(num_rois, dtype=int)
            num_slices = [1]  # Placeholder
            for curr_slice in num_slices:
                num_curr_image_rois = np.sum(roi_info).astype(
                    int
                )  # Adjust for other zs values
                roi_ids = np.where(roi_info[:] == 1)[0]

                cnt = 1
                img_offset_x, img_offset_y, roi_img_height = None, None, None
                for roi_idx in roi_ids:
                    if cnt == 1:
                        # The first one will be at the very top
                        img_offset_x = 0
                        img_offset_y = 0
                    else:
                        # For the rest of the rois, there will be a recurring numLinesBetweenScanfields spacing
                        img_offset_y = (
                            img_offset_y + roi_img_height + numLinesBetweenScanfields
                        )

                    # The width of the scanfield doesn't change
                    roi_img_width = roi_data[roi_idx].hRoi["scanfields"][
                        "pixelResolutionXY"
                    ][0]
                    # The height of the scanfield depends on the interpolation of scanfields within existing fields
                    roi_img_height = roi_img_height_info[roi_idx].astype(int)
                    roi_img_height_range = np.arange(0, roi_img_height)[:, None]
                    roi_img_width_range = np.arange(0, roi_img_width)
                    roi_image_cnt[roi_idx] += 1

                    y_indices = (img_offset_y + roi_img_height_range).astype(int)
                    x_indices = (img_offset_x + roi_img_width_range).astype(int)

                    extracted_data = data[
                        y_indices, x_indices, curr_channel, curr_volume
                    ]
                    roi_data[roi_idx].add_image_to_volume(
                        curr_channel, curr_volume, extracted_data
                    )
                    cnt += 1
    return roi_data, roi_group


def get_metadata_from_tiff(
    filename: os.PathLike,
):
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"File {filename} not found")
    with open(filename, "rb") as fh:
        metadata = tifffile.read_scanimage_metadata(fh)

    static_metadata = metadata[0]
    frame_metadata = metadata[1]["RoiGroups"]["imagingRoiGroup"]["rois"]
    rois = [x["scanfields"] for x in frame_metadata]

    num_rois = len(rois)

    num_planes = len(static_metadata["SI.hChannels.channelSave"])
    lines_per_frame = static_metadata["SI.hRoiManager.linesPerFrame"]
    px_per_line = static_metadata["SI.hRoiManager.pixelsPerLine"]

    num_lines_between_scanfields = np.round(static_metadata["SI.hScan2D.flytoTimePerScanfield"] / float(static_metadata["SI.hRoiManager.linePeriod"]), 0,)

    roi_center_xy = rois[0]["centerXY"]  # needed to realign the image
    roi_size_xy = rois[0]["sizeXY"]

    num_px_x = rois[0]["pixelResolutionXY"][0]  # number of pixels in x
    num_px_y = rois[0]["pixelResolutionXY"][1]  # number of pixels in y

    num_frames_total = static_metadata["SI.hStackManager.framesPerSlice"]  # number of frames
    num_frames_file = static_metadata["SI.hScan2D.logFramesPerFile"]  # number of frames per file
    if num_frames_file >= num_frames_total:
        num_files = 1
    else:
        num_files = int(num_frames_total / num_frames_file)  # number of files

    frame_rate = static_metadata["SI.hRoiManager.scanVolumeRate"]  # hz
    objective_resolution = static_metadata["SI.objectiveResolution"]  # deg/deg
    fovx = np.round(roi_size_xy[0] * objective_resolution, 0) # deg/um
    fovy = np.round(roi_size_xy[1] * objective_resolution, 0)  # deg/um
    pixel_resolution_x = np.round(fovx / num_px_x, 0)  # um/pixel
    pixel_resolution_y = np.round(fovy / num_px_y, 0)  # um/pixel

    # make a range of x and y values to use for indexing
    x_roi_range = np.arange(0, num_px_x)
    y_roi_range = np.arange(0, num_px_y)

    new_roi_width = len(x_roi_range)
    new_roi_height = len(y_roi_range)

    new_roi_sizes = {
        "new_roi_width": new_roi_width,
        "new_roi_height": new_roi_height,
        "x_roi_range": x_roi_range,
        "y_roi_range": y_roi_range
    }

    return {
        "raw_filename": filename,
        "static_metadata": static_metadata,
        "frame_metadata": frame_metadata,
        "num_frames_total": num_frames_total,
        "num_frames_file": num_frames_file,
        "num_files": num_files,
        "num_planes": num_planes,
        "num_rois": num_rois,
        "frame_rate": frame_rate,
        "num_px_x": num_px_x,
        "num_px_y": num_px_y,
        "fov_x": fovx,
        "fov_y": fovy,
        "pixel_resolution_x": pixel_resolution_x,
        "pixel_resolution_y": pixel_resolution_y,
        "lines_per_frame": lines_per_frame,
        "px_per_line": px_per_line,
        "objective_resolution": objective_resolution,
        "sizes": new_roi_sizes,
        "roi_center_xy": roi_center_xy,
        "roi_size_xy": roi_size_xy,
        "num_lines_between_scanfields": num_lines_between_scanfields
    }
