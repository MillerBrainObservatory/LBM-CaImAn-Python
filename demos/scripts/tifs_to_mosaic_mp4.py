# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 01:25:20 2023

@author: otero
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage
import tifffile


fps = 18.82 * 2
rolling_average_frames = 5
use_until_frame_n = 600  # -1 for entire recording
rows = 2
columns = 3
gaps_columns = 25
gaps_rows = 25
intensity_percentiles = [25, 99.8]
input_filenames = [
    "/Data/analysis_2pRAM/VolumetricTest/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00001_Reshaped_plane0.tif",
    "/Data/analysis_2pRAM/VolumetricTest/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00001_Reshaped_plane1.tif",
    "/Data/analysis_2pRAM/VolumetricTest/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00001_Reshaped_plane2.tif",
    "/Data/analysis_2pRAM/VolumetricTest/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00001_Reshaped_plane3.tif",
    "/Data/analysis_2pRAM/VolumetricTest/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00001_Reshaped_plane4.tif",
    "/Data/analysis_2pRAM/VolumetricTest/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00001_Reshaped_plane5.tif",
]
# input_filenames = [  r'C:\Users\otero\Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00001_Reshaped_Plane01.tif',
#                      r'C:\Users\otero\Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00001_Reshaped_Plane03.tif',
#                      r'C:\Users\otero\Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00001_Reshaped_Plane05.tif',
#                      r'C:\Users\otero\Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00001_Reshaped_Plane07.tif',
#                      r'C:\Users\otero\Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00001_Reshaped_Plane09.tif',
#                      r'C:\Users\otero\Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00001_Reshaped_Plane11.tif']
# output_filename =     'C:/Users/otero/Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00001_Reshaped_Planes_01_03_05_07_09_11_avg5frames_play2x.mp4'

# input_filenames = [r'C:\Users\otero\Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00006_Reshaped_Plane01.tif',
#                   r'C:\Users\otero\Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00006_Reshaped_Plane03.tif',
#                   r'C:\Users\otero\Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00006_Reshaped_Plane05.tif',
#                   r'C:\Users\otero\Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00006_Reshaped_Plane07.tif',
#                   r'C:\Users\otero\Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00006_Reshaped_Plane09.tif',
#                   r'C:\Users\otero\Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00006_Reshaped_Plane11.tif']
# output_filename =  'C:/Users/otero/Desktop/MosaicMax15/Max15_pl15at0umdeep_0p6by0p6mm_9p6Hz_1umppix_100pct_stim_00001_00006_Reshaped_Planes_01_03_05_07_09_11_avg5frames_play2x.mp4'


# input_filenames = [r'C:\Users\otero\Desktop/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00001_Reshaped_Plane01.tif',
#                   r'C:\Users\otero\Desktop/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00001_Reshaped_Plane03.tif',
#                   r'C:\Users\otero\Desktop/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00001_Reshaped_Plane05.tif',
#                  r'C:\Users\otero\Desktop/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00001_Reshaped_Plane07.tif']
output_filename = r"/Data/analysis_2pRAM/VolumetricTest/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00001_Reshaped_planes1to6_movie.mp4"


tif = tifffile.imread(input_filenames[0])
tif = tif[:use_until_frame_n]
tifs_shape = list(tif.shape)
tifs_shape[0] = tifs_shape[0] - rolling_average_frames + 1

canvas = np.zeros(
    (
        tifs_shape[0],
        (tifs_shape[1] + gaps_rows) * rows - gaps_rows,
        (tifs_shape[2] + gaps_columns) * columns - gaps_columns,
    ),
    dtype=np.uint8,
)

tifs = []
for tif_i in range(len(input_filenames)):
    print("Working on " + input_filenames[tif_i])

    if tif_i != 0:  # No need to re-load the first tif
        tif = tifffile.imread(input_filenames[tif_i])
        tif = tif[:use_until_frame_n]

    # Apply rolling average by convolving
    tif = scipy.signal.convolve(
        tif, np.ones(([rolling_average_frames, 1, 1])), mode="valid"
    )

    # Normalize to [0,1]
    tif -= np.min(tif)
    tif = tif / np.max(tif)

    # Normalize to percentile-dynamic-range
    pct_low, pct_high = np.percentile(tif, intensity_percentiles)
    tif = skimage.exposure.rescale_intensity(tif, in_range=(pct_low, pct_high))

    # Rescale and transform to uint8
    tif = np.round(tif * 255)
    tif = tif.astype(np.uint8)

    # Check
    assert list(tif.shape) == tifs_shape

    # Place it on canvas
    x_start = tif_i % columns * (tif.shape[2] + gaps_columns)
    y_start = tif_i // columns * (tif.shape[1] + gaps_rows)
    x_end = x_start + tif.shape[2]
    y_end = y_start + tif.shape[1]
    canvas[:, y_start:y_end, x_start:x_end] = tif
    plt.imshow(np.mean(canvas, axis=0))
    plt.show()

plt.imshow(np.mean(canvas, axis=0))

size = (canvas.shape[2], canvas.shape[1])
out = cv2.VideoWriter(
    output_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, size, False
)
for f in range(tifs_shape[0]):
    out.write(canvas[f])
out.release()
