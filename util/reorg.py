import numpy as np
from util.scan import return_scan_offset
from util.scan import fix_scan_phase


def extract_roi_data(num_px, num_channels, num_rois, num_frames, roi_data):
    """
    Extract 4D image data from scanimage tiff recordings.

    This function takes the original multidimensional data acquired from microscopy scans and
    reorganizes it into a structured format, applying corrections and adjustments to each channel,
    region of interest (ROI), and frame to prepare it for further analysis.

    Args:
        num_px (tuple): The pixel resolution in the XY plane, given as a tuple of two integers.
        num_channels (int): The total number of channels to process.
        num_rois (int): The number of regions of interest (ROIs) to process.
        num_frames (int): The total number of frames to process.
        roi_data (list): A list containing ROI data objects, where each object contains image data
                         for different channels and frames.

    Returns:
        np.ndarray: A numpy array containing the reorganized image data, with adjustments applied to
                    each channel, ROI, and frame. The array has a data type of int.
    """

    imageData = []
    roi_x, roi_y = num_px

    for channel in range(num_channels):
        print(f"Assembling channel {channel + 1} of {num_channels}...")
        frameTemp = []
        for strip in range(num_rois):
            stripTemp = np.empty((roi_y, roi_x, 1, num_frames), dtype=np.float32)
            for frame in range(num_frames):
                frame_data = roi_data[strip].imageData[channel][frame][0]
                stripTemp[:, :, 0, frame] = roi_data[strip].imageData[channel][frame][0]
            corr = return_scan_offset(stripTemp, 1).astype(int)

            stripTemp = fix_scan_phase(np.array(stripTemp), corr, 1)
            val = round(stripTemp.shape[0] * 0.03)
            stripTemp = stripTemp[val - 1 :, 6:138, :, :]
            if len(frameTemp) != 0:
                frameTemp = np.concatenate((frameTemp, stripTemp), axis=1)
            else:
                frameTemp = stripTemp
        if len(imageData) != 0:
            imageData = np.concatenate((imageData, frameTemp), axis=2)
        else:
            imageData = frameTemp
    return imageData.astype(int)
