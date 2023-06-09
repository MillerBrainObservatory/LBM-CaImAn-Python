# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 19:02:03 2023

@author: otero
"""

# python /FreiwaldSync/MarmoScope/Analysis/SantiVersions/reshaping_and_aligning_planes_multifile_volume.py >dev/pts/2 2>dev/pts/2
# free -s 5 >/dev/pts/2
import copy
import cv2
import datetime
import glob
import h5py
import json
import logging
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 900
import numpy as np
import os
import scipy
import scipy.signal
import skimage
import sys
import tifffile
import time


#%% USER-DEFINED PARAMETERS

make_template_seams_and_plane_alignment = True
reconstruct_all_files = True

# Template parameters 
list_files_for_template = [0]
if not reconstruct_all_files:
    reconstruct_until_this_ifile = 10

# Directories
#raw_data_dirs = [r'F:\Santiago\Data\Coconut\20230107d\Dir/1/'] # Must be a list with 1 or more dirs
raw_data_dirs = [r'/Data/analysis_2pRAM/Coconut/20230107d/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW/'] # Must be a list with 1 or more dirs
fname_must_contain = ''
fname_must_NOT_contain = 'nothing in particular'

# Parameters output volume
save_as_volume_or_planes = 'planes'
if save_as_volume_or_planes == 'volume':
    nonan_volume = True
    lateral_align_planes = False
    
make_mroi_overlaps_identical = True
add_1000_for_nonegative_volume = True
save_output = True # should be true unless testing...

# Parameters MROIs seams
seams_overlap = 'calculate' # Should be either 'calculate', an integer, or a list of length n_planes
if seams_overlap == 'calculate':
    n_ignored_pixels_sides = 5
    min_seam_overlap = 1
    max_seam_overlap = 20 # Used if seams_overlap_setting = 'calculate'
    alignment_plot_checks = False

# Logging
json_logging = True

# Video and mean-frame png
save_mp4 = True
save_meanf_png = True
if save_mp4 or save_meanf_png:
    rows = 6
    columns = 5
    gaps_columns = 5
    gaps_rows = 5
    intensity_percentiles = [15,99.5]
    if save_meanf_png:
        meanf_png_only_first_file = True
    if save_mp4:
        video_only_first_file = True
        video_play_speed = 2
        rolling_average_frames = 10 
        video_duration_secs = 10

#%% Get input and output paths

now = datetime.datetime.now()
date_string = now.strftime("%Y%m%dd_%H%M%St")
json_filename = f"{raw_data_dirs[0]}log_json_{date_string}.json"
json_formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}')

json_logger = logging.getLogger(__name__)
json_logger.setLevel(logging.DEBUG)
json_handler = logging.FileHandler(json_filename)
print_handler = logging.StreamHandler(sys.stdout)
json_handler.setFormatter(json_formatter)
json_logger.addHandler(json_handler)
json_logger.addHandler(print_handler)

path_all_files = []
for i_dir in raw_data_dirs:
    tmp_paths = sorted(glob.glob(i_dir + '/**/*.tif', recursive = True))
    for this_tmp_path in tmp_paths:
        if fname_must_contain in this_tmp_path and fname_must_NOT_contain not in this_tmp_path:
            path_all_files.append(this_tmp_path)
            
if json_logging: json_logger.info(json.dumps({'path_all_files': path_all_files}))

n_template_files = len(list_files_for_template)
path_template_files = [path_all_files[file_idx] for file_idx in list_files_for_template]

del i_dir, raw_data_dirs, fname_must_contain, fname_must_NOT_contain
    

#%%
pipeline_steps = []
if make_template_seams_and_plane_alignment:
    pipeline_steps.append('make_template')
if reconstruct_all_files:
    pipeline_steps.append('reconstruct_all')

    
for current_pipeline_step in pipeline_steps:
    if current_pipeline_step == 'make_template':
        path_input_files = path_template_files
    elif current_pipeline_step == 'reconstruct_all':
        path_input_files = path_all_files
        
    #%% Iterate over files
    n_files = len(path_input_files)
    if reconstruct_all_files:
        list_files_for_reconstruction = range(n_files)
    else:
        list_files_for_reconstruction = range(reconstruct_until_this_ifile)
        
    for i_file in list_files_for_reconstruction:
        tic = time.time()
        path_input_file = path_input_files[i_file]
            
        if i_file == 0:
            first_file_of_recording = True
        else:
            first_file_of_recording = False
            
        print('---------------------------------------------------')
        if json_logging: json_logger.debug(json.dumps({'debug_message':'Started working on: ' + str(path_input_file)}))
    
        #%% Determine if it is a single-plane, Max15, or Max30 recording
        
        if first_file_of_recording:
            if 'SP' in path_input_file:
                n_planes = 1
                chans_order = np.array([0])
            elif 'Max15' in path_input_file:
                n_planes = 15
                chans_order = np.array([ 1,  3,  4,  5,  6,  7,  2,  8,  9, 10, 11, 12, 13, 14, 15]) - 1
            elif 'Max30' in path_input_file:
                n_planes = 30
                chans_order = np.array([ 1,  5,  6,  7,  8,  9,  2, 10, 11, 12, 13, 14, 15, 16, 17,
                                            3, 18, 19, 20, 21, 22, 23,  4, 24, 25, 26, 27, 28, 29, 30]) - 1
            else:
                n_planes = int(input('Check filename... Number of planes?'))
            if json_logging: json_logger.debug(json.dumps({'debug_message':'Number of planes: ' + str(n_planes)}))
            
            #%% Get MROI info from tif metadata
            
            with tifffile.TiffFile(path_input_file) as tif:
                metadata = {}
                for tag in tif.pages[0].tags.values():
                    tag_name, tag_value = tag.name, tag.value
                    metadata[tag_name] = tag_value
            tif.close() # Should be closed automatically when opening using 'with'
            del tif
                    
            mrois_si_raw = json.loads(metadata["Artist"])['RoiGroups']['imagingRoiGroup']['rois']
            if type(mrois_si_raw) != dict:
                mrois_si = []
                for roi in mrois_si_raw:
                    if type(roi['scanfields']) != list:
                        scanfield = roi['scanfields']
                    else: 
                        scanfield = roi['scanfields'][np.where(np.array(roi['zs'])==0)[0][0]]
                    roi_dict = {}
                    roi_dict['center'] = np.array(scanfield['centerXY'])
                    roi_dict['sizeXY'] = np.array(scanfield['sizeXY'])
                    roi_dict['pixXY'] = np.array(scanfield['pixelResolutionXY'])
                    mrois_si.append(roi_dict)
                del roi
            else:
                scanfield = mrois_si_raw['scanfields']
                roi_dict = {}
                roi_dict['center'] = np.array(scanfield['centerXY'])
                roi_dict['sizeXY'] = np.array(scanfield['sizeXY'])
                roi_dict['pixXY'] = np.array(scanfield['pixelResolutionXY'])
                mrois_si = [roi_dict]
            
            #Sort MROIs so they go from left-to-right (but keep the un-sorted because that matches how they were acquired and saved in the long-tif-strip)
            mrois_centers_si = np.array([mroi_si['center'] for mroi_si in mrois_si])
            x_sorted = np.argsort(mrois_centers_si[:,0])
            mrois_si_sorted_x = [mrois_si[i] for i in x_sorted]
            mrois_centers_si_sorted_x = [mrois_centers_si[i] for i in x_sorted]
            
            del mrois_si_raw, roi_dict, scanfield
        
        #%% Load, reshape (so time and planes are 2 independent dimensions) and re-order (planes, fix Jeff's order)
        
        if json_logging: json_logger.debug(json.dumps({'debug_message':'Loading (expect warning if multi-file recording) and reshaping... '}))

        tiff_file = tifffile.imread(path_input_file)
        if n_planes > 1: 
            tiff_file = np.reshape(tiff_file, ( int(tiff_file.shape[0]/n_planes), n_planes,tiff_file.shape[1],tiff_file.shape[2]), order='A') # warnings are expected if the recording is split into many files or incomplete
        else:
            tiff_file = np.expand_dims(tiff_file,1)
        tiff_file = np.swapaxes(tiff_file, 1, 3)
        tiff_file = tiff_file[:,:,:,chans_order]
        
        #%% Separate tif into MROIs
        if first_file_of_recording:
            n_mrois = len(mrois_si)
            tif_pixels_Y = tiff_file.shape[2]
            mrois_pixels_Y = np.array([mroi_si['pixXY'][1] for mroi_si in mrois_si])
            each_flyback_pixels_Y = (tif_pixels_Y - mrois_pixels_Y.sum())//(n_mrois - 1)
        
        if current_pipeline_step == 'reconstruct_all': 
            planes_mrois = np.empty((n_planes,n_mrois), dtype=np.ndarray)
            if json_logging: json_logger.debug(json.dumps({'debug_message':'Separating tif into individual MROIs'}))
            for i_plane in range(n_planes):
                y_start = 0
                for i_mroi in range(n_mrois): # We go over the order in which they were acquired
                    planes_mrois[i_plane, i_mroi] = tiff_file[:, :, y_start:y_start+mrois_pixels_Y[i_mroi], i_plane]
                    y_start += mrois_pixels_Y[i_mroi] + each_flyback_pixels_Y
            
            for i_plane in range(n_planes):
                planes_mrois[i_plane,:] = planes_mrois[i_plane,:][x_sorted]

        else: 
            tiff_file_meanf = np.mean(tiff_file, axis = 0)
    
            if i_file == 0:
                templates_planes_mrois_meanf = np.empty((n_template_files, n_planes,n_mrois), dtype=np.ndarray)
            for i_plane in range(n_planes):
                y_start = 0
                for i_mroi in range(n_mrois): # We go over the order in which they were acquired
                    templates_planes_mrois_meanf[i_file, i_plane, i_mroi] = tiff_file_meanf[:, y_start:y_start+mrois_pixels_Y[i_mroi], i_plane]
                    y_start += mrois_pixels_Y[i_mroi] + each_flyback_pixels_Y
                    
            if i_file != n_template_files -1:
                del tiff_file
                continue
            else: # If this is the last templating file, then we need to do a bunch of stuff
                
            
                #%% Get rough location of MROIs in final canvas based on MROI metadata
                
                # Get pixel sizes
                sizes_mrois_pix = np.array([mroi_pix.shape for mroi_pix in templates_planes_mrois_meanf[0,0,:]])
                sizes_mrois_si = np.array([mroi_si['sizeXY'] for mroi_si in mrois_si_sorted_x])
                pixel_sizes = sizes_mrois_si/sizes_mrois_pix
                psize_x, psize_y = np.mean(pixel_sizes[:,0]), np.mean(pixel_sizes[:,1])
                assert np.product(np.isclose(pixel_sizes[:,1]-psize_y, 0)), "Y-pixels resolution not uniform across MROIs"
                assert np.product(np.isclose(pixel_sizes[:,0]-psize_x, 0)), "X-pixels resolution not uniform across MROIs"
                assert np.product(np.isclose(pixel_sizes[:,0]-pixel_sizes[:,1], 0)), "Pixels do not have squared resolution"
                
                # Calculate the pixel ranges (with their SI locations) that would fit all MROIs
                top_left_corners_si = mrois_centers_si_sorted_x - sizes_mrois_si/2
                bottom_right_corners_si = mrois_centers_si_sorted_x + sizes_mrois_si/2
                xmin_si, ymin_si = top_left_corners_si[:,0].min(), top_left_corners_si[:,1].min()
                xmax_si, ymax_si = bottom_right_corners_si[:,0].max(), bottom_right_corners_si[:,1].max()
                reconstructed_xy_ranges_si = [np.arange(xmin_si, xmax_si, psize_x), np.arange(ymin_si, ymax_si, psize_y)]
                
                # Calculate the starting pixel for each MROI when placed in the reconstructed movie
                top_left_corners_pix = np.empty((n_mrois,2), dtype=int)
                for i_xy in range(2):
                    for i_mroi in range(n_mrois):
                        closest_xy_pix = np.argmin(np.abs(reconstructed_xy_ranges_si[i_xy] - top_left_corners_si[i_mroi, i_xy]))
                        top_left_corners_pix[i_mroi,i_xy] = int(closest_xy_pix)
                        closest_xy_si = reconstructed_xy_ranges_si[i_xy][closest_xy_pix]
                        if not np.isclose(closest_xy_si, top_left_corners_si[i_mroi, i_xy]):
                            
                            if json_logging: json_logger.debug(json.dumps({'debug_message':"ROI %d x does not fit perfectly into image, corner is %.4f but closest available is %.4f" %\
                                  (i_mroi, closest_xy_si, top_left_corners_si[i_mroi, i_xy])}))
                #Sometimes an extra pixel is added because of pixel_size rounding
                for i_xy in range(2):
                    if len(reconstructed_xy_ranges_si[i_xy]) == np.sum(sizes_mrois_pix[:,0]) + 1: 
                        reconstructed_xy_ranges_si[i_xy] = reconstructed_xy_ranges_si[i_xy][:-1]
                     
                    
                #%% Calculate optimal overlap for seams
        
                if seams_overlap == 'calculate':
                    # Determine if all the MROIs are adjacent
                    for i_mroi in range(n_mrois-1):
                        if top_left_corners_pix[i_mroi][0] + sizes_mrois_pix[i_mroi][0] != top_left_corners_pix[i_mroi + 1][0]:
                            raise Exception('MROIs number ' + str(i_mroi) + ' and number ' + str(i_mroi+1) + ' (0-based idx) are not contiguous')
                    
                    # Combine meanf from differete template files: 
                    overlaps_scores = np.zeros((n_template_files,n_planes, n_mrois-1, max_seam_overlap-min_seam_overlap))  # We will avoid i_overlaps = 0
                    
                    for i_template in range (n_template_files):
                        for i_plane in range(n_planes):
                            for i_seam in range(n_mrois-1):
                                for i_overlaps in range(min_seam_overlap,max_seam_overlap):
                                    strip_left  = templates_planes_mrois_meanf[i_template, i_plane, i_seam  ][-n_ignored_pixels_sides-i_overlaps:-n_ignored_pixels_sides]
                                    strip_right = templates_planes_mrois_meanf[i_template, i_plane, i_seam+1][n_ignored_pixels_sides:i_overlaps +n_ignored_pixels_sides]
                                    subtract_left_right = abs(strip_left - strip_right)
                                    overlaps_scores[i_template, i_plane, i_seam, i_overlaps-min_seam_overlap] = np.mean(subtract_left_right)
                                    
                    # Plot scores and select optimal overlap
                    del strip_left, strip_right
                    planes_mrois_meanf = np.mean(templates_planes_mrois_meanf, axis = 0)
                    
                    overlaps_scores_mean_files_mean_rois = np.mean(overlaps_scores, axis = (0,2))
                    overlaps_planes = []
                    for i_plane in range(n_planes):
                        overlaps_planes.append(int(np.argmin(overlaps_scores_mean_files_mean_rois[i_plane])+ min_seam_overlap + 2 * n_ignored_pixels_sides))
                    if json_logging: json_logger.info(json.dumps({'overlaps_planes' : overlaps_planes}))
                  
                    #Plot the scores for the different planes and also potential shifts 
                    if alignment_plot_checks:
                        
                        for i_plane in range(n_planes):
                            plt.plot(range(min_seam_overlap,max_seam_overlap), overlaps_scores_mean_files_mean_rois[i_plane])
                        plt.title('Score for all planes')
                        plt.xlabel('Overlap (pixels)')
                        plt.ylabel('Error (a.u.)')
                        plt.show()
                        
                        for i_plane in range(n_planes):
                            for shift in range(-2, 3):
                                i_overlap = overlaps_planes[i_plane] + shift
                                canvas_alignment_check = np.zeros((len(reconstructed_xy_ranges_si[0]) - (n_mrois-1) * i_overlap, 
                                                                   len(reconstructed_xy_ranges_si[1]), 
                                                                   3), dtype = np.float32)
                                x_start = 0
                                for i_mroi in range(n_mrois):
                                    x_start = top_left_corners_pix[i_mroi][0] - i_mroi * i_overlap
                                    x_end = x_start + sizes_mrois_pix[i_mroi][0]
                                    y_start = top_left_corners_pix[i_mroi][1]
                                    y_end = y_start + sizes_mrois_pix[i_mroi][1]
                                    canvas_alignment_check[x_start:x_end,y_start:y_end,i_mroi%2] = planes_mrois_meanf[i_plane,i_mroi] - np.min(planes_mrois_meanf[i_plane,i_mroi])
                                
                                pct_low, pct_high = np.nanpercentile(canvas_alignment_check, [80,99.9]) # Consider that we are using 1/3 of pixels (RGB channels)
                                canvas_alignment_check = skimage.exposure.rescale_intensity(canvas_alignment_check, in_range=(pct_low, pct_high))
                                plt.imshow(np.swapaxes(canvas_alignment_check,0,1))
                                plt.title('Plane: ' + str(i_plane))
                                plt.xlabel('Original overlap: ' + str(overlaps_planes[i_plane]) + ' + Shift: ' + str(shift))
                                plt.show()
                    if make_mroi_overlaps_identical:
                        overlaps_planes = [int(round(np.mean(overlaps_planes)))] * n_planes
                        
                                 
                elif seams_overlap is int:
                    overlaps_planes = [seams_overlap] * n_planes
                elif seams_overlap is list:
                    overlaps_planes = seams_overlap
                else: 
                    raise Exception('seams_overlap should be set to \'calculate\', an integer, or a list of length n_planes')
        
        del tiff_file
        
        #%% Create a volume container
        
        if first_file_of_recording or current_pipeline_step == 'make_template': # For templatingMROIs, we will get here when working on the last file
            n_x = len(reconstructed_xy_ranges_si[0]) - min(overlaps_planes) * (n_mrois-1)
            n_y = len(reconstructed_xy_ranges_si[1])
            n_z = n_planes
        
        if current_pipeline_step == 'make_template':
            n_f = 1
        else:
            n_f = planes_mrois[0,0].shape[0] #The last file probably has a different number of frames, so always check 
         
        if json_logging: json_logger.debug(json.dumps({'debug_message':'Creating volume of shape: ' + str([n_f, n_x, n_y, n_z]) + ' (f,x,y,z)...'}))
        planes = []
        volume = np.empty((n_f, n_x, n_y, n_z), dtype = np.float32)
        volume[:] = np.nan
        #%% Merge MROIs and place them into a volume
        if json_logging: json_logger.debug(json.dumps({'debug_message':'Merging MROIs and placing them into the volume'}))
        
       
        interplane_shifts = np.zeros((n_planes,2), dtype=int)
        accumulated_shifts = np.zeros((n_planes,2), dtype=int)
        for i_plane in range(n_planes):
            overlap_seams_this_plane = overlaps_planes[i_plane]
            plane_width = len(reconstructed_xy_ranges_si[0]) - overlap_seams_this_plane * (n_mrois-1)
            plane_canvas = np.zeros((n_f, plane_width, n_y), dtype = np.float32)
            for i_mroi in range(n_mrois):
                # The first and last MROIs require different handling
                if i_mroi == 0:
                    x_start_canvas = 0 # This always works because the MROIs were sorted
                    x_end_canvas = x_start_canvas + sizes_mrois_pix[i_mroi][0] - int(np.trunc(overlap_seams_this_plane /2))
                    x_start_mroi = x_start_canvas 
                    x_end_mroi = x_end_canvas
                elif i_mroi != n_mrois-1:
                    x_start_canvas = copy.deepcopy(x_end_canvas)
                    x_end_canvas = x_start_canvas + sizes_mrois_pix[i_mroi][0] - overlap_seams_this_plane
                    x_mroi_width = sizes_mrois_pix[i_mroi][0] - overlap_seams_this_plane
                    x_start_mroi = int(np.ceil(overlap_seams_this_plane /2))
                    x_end_mroi = x_start_mroi + x_mroi_width
                else:
                    x_start_canvas = copy.deepcopy(x_end_canvas)
                    x_end_canvas = plane_width
                    x_start_mroi = int(np.ceil(overlap_seams_this_plane /2))
                    x_end_mroi = sizes_mrois_pix[i_mroi][0]
                
                y_start_canvas = top_left_corners_pix[i_mroi][1]
                y_end_canvas = y_start_canvas + sizes_mrois_pix[i_mroi][1]
                
                if current_pipeline_step == 'reconstruct_all':
                    plane_canvas[:, x_start_canvas:x_end_canvas, y_start_canvas:y_end_canvas] = planes_mrois[i_plane, i_mroi][:,x_start_mroi:x_end_mroi]
                else:
                    plane_canvas[:, x_start_canvas:x_end_canvas, y_start_canvas:y_end_canvas] = planes_mrois_meanf[i_plane, i_mroi][x_start_mroi:x_end_mroi]
            #The plane may need padding due to different overlap in the mroi seams
            pad_x_needed = n_x - plane_width
            pad_left = round(np.trunc(pad_x_needed/2))
            pad_right= round(np.ceil (pad_x_needed/2))
            plane_canvas = np.pad(plane_canvas, ((0,0), (pad_left, pad_right), (0,0)), 'constant', constant_values=np.nan)
            
            if current_pipeline_step == 'make_template':
                # Calculate lateral offsets across planes and align planes
                # For the first file of each recording, we will calculate the lateral-shift vectors across planes
                # For that, we will do cross-correlation between mean-frame images from 2 adjacent planes
                # https://stackoverflow.com/questions/24768222/how-to-detect-a-shift-between-images
                plane_canvas_meanf = np.mean(plane_canvas, axis = 0)

                

                if i_plane == 0:
                    previous_plane_canvas_meanf = copy.deepcopy(plane_canvas_meanf)
                else:
                    # Calculate shifts
                    plane_canvas_meanf = np.mean(plane_canvas, axis = 0)
                    im1_copy = copy.deepcopy(previous_plane_canvas_meanf)
                    im2_copy = copy.deepcopy(plane_canvas_meanf)
                    # Removing nans
                    nonan_mask = np.stack((~np.isnan(im1_copy), ~np.isnan(im2_copy)), axis = 0)
                    nonan_mask = np.all(nonan_mask, axis = 0)
                    coord_nonan_pixels = np.where(nonan_mask)
                    min_x, max_x = np.min(coord_nonan_pixels[0]) , np.max(coord_nonan_pixels[0])
                    min_y, max_y = np.min(coord_nonan_pixels[1]) , np.max(coord_nonan_pixels[1])
                    im1_nonan = im1_copy[min_x:max_x+1,min_y:max_y+1]
                    im2_nonan = im2_copy[min_x:max_x+1,min_y:max_y+1]
    
                    im1_nonan -= np.min(im1_nonan)
                    im2_nonan -= np.min(im2_nonan)
                       
                    cross_corr_img = scipy.signal.fftconvolve(im1_nonan, im2_nonan[::-1,::-1], mode='same')
                    
                    # Calculate vector 
                    corr_img_peak_x,  corr_img_peak_y  = np.unravel_index(np.argmax(cross_corr_img), cross_corr_img.shape)
                    self_corr_peak_x, self_corr_peak_y = [dim / 2 for dim in cross_corr_img.shape]
                    interplane_shift = [corr_img_peak_x - self_corr_peak_x, corr_img_peak_y - self_corr_peak_y]
                    print(interplane_shift)
                    interplane_shifts[i_plane] = copy.deepcopy(interplane_shift)
                    accumulated_shifts[i_plane] = np.sum(interplane_shifts, axis = 0, dtype = int)
                    previous_plane_canvas_meanf = copy.deepcopy(plane_canvas_meanf)
   
            elif current_pipeline_step == 'reconstruct_all':
                ## This works but it is supposed to be slower...
                # volume[:,:,:,i_plane] = scipy.ndimage.interpolation.shift(  
                #     volume[:,:,:,i_plane], # If not float, it cannot place NaNs
                #     (0, accumulated_shifts[i_plane][0], accumulated_shifts[i_plane][1]), 
                #     order = 0, 
                #     cval=np.nan)
                start_x_plane = start_x_volume = start_y_plane = start_y_volume = 0
                end_x_plane = end_x_volume = n_x
                end_y_plane = end_y_volume = n_y
                nan_x_start = nan_y_start = nan_x_end = nan_y_end = 0
                
                if lateral_align_planes:
                    shift_x, shift_y = accumulated_shifts[i_plane]
                else:
                    shift_x, shift_y = 0, 0
                if shift_x < 0:
                    start_x_plane = -shift_x
                    end_x_volume = shift_x
                    nan_x_end = -shift_x
                elif shift_x > 0:
                    end_x_plane = shift_x
                    start_x_volume = -shift_x
                    nan_x_start = shift_x
                if shift_y < 0:
                    start_y_plane = shift_y
                    end_y_volume = -shift_y
                    nan_y_end = -shift_y
                elif shift_y > 0:
                    end_y_plane = -shift_y
                    start_y_volume = shift_y
                    nan_y_start = shift_y
                
                plane_canvas[:,start_x_volume:end_x_volume, start_y_volume:end_y_volume] = \
                plane_canvas[:,start_x_plane: end_x_plane,  start_y_plane: end_y_plane]
                
                if nan_x_start != 0:
                    plane_canvas[:,:nan_x_start,:] = np.nan
                elif nan_x_end != 0:
                    plane_canvas[:,-nan_x_end:,:] = np.nan
                if nan_y_start != 0:
                    plane_canvas[:,:,:nan_y_start] = np.nan
                elif nan_y_end != 0:
                    plane_canvas[:,:,-nan_y_end:] = np.nan
                # planes.append(plane_canvas)
                volume[:,:,:,i_plane] = plane_canvas
        
        a = 2/0
        if json_logging: json_logger.debug(json.dumps({'debug_message':'Plane Canvas shifted and appended'}))
        
        if current_pipeline_step == 'make_template':
            if json_logging: json_logger.info(json.dumps({'accumulated_shifts': accumulated_shifts.tolist()}))
            continue
        else:
            del planes_mrois, plane_canvas
            #volume = np.stack(planes[:], axis = -1)
            #del planes
        if json_logging: json_logger.debug(json.dumps({'debug_message':'Volume created successfully'}))
            
        
    
        #%% Select X,Y pixels that do not have nans for any plane
        if nonan_volume:
            if json_logging: json_logger.debug(json.dumps({'debug_message':'Trimming volume to remove NaNs...'}))
            volume_meanp = np.mean(volume, axis = 3)
            non_nan = ~np.isnan(volume_meanp[0])    
            coord_nonan_pixels = np.where(non_nan)
            min_x, max_x = np.min(coord_nonan_pixels[0]) , np.max(coord_nonan_pixels[0])+1
            min_y, max_y = np.min(coord_nonan_pixels[1]) , np.max(coord_nonan_pixels[1])+1
            volume = volume[:,min_x:max_x,min_y:max_y] 
            if current_pipeline_step == 'reconstruct_all':
                volume = volume.astype(np.int16) # If not using meanfs, the volume should only contain raw int16 values
            if json_logging: json_logger.debug(json.dumps({'debug_message':'Shape of trimmed nonan volume: ' + str(volume.shape)}))

    
        #%% Make volume non-negative
        if add_1000_for_nonegative_volume:
            volume += 1000
    
        #%% Saving outputs
        
        # tifffile.imwrite((path_output_file[:-3] + '.tif'), volume)
        if current_pipeline_step == 'reconstruct_all':
            if save_output:
                if json_logging: json_logger.debug(json.dumps({'debug_message':'Saving output file...'}))
                if save_as_volume_or_planes == 'volume':
                    path_output_file = path_input_file[:-4] + '_Aligned.h5'
                    save_dir = os.path.dirname(path_output_file)
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    h5file = h5py.File(path_output_file, 'w') 
                    h5file.create_dataset('data', data=volume)
                    h5file.attrs.create('metadata', str(metadata)) # You can use json to load it as a dictionary
                    h5file.close()
                    del h5file
                elif save_as_volume_or_planes == 'planes':
                    for i_plane in range(n_planes):
                        save_dir = os.path.dirname(path_output_file) + '/plane'f'{i_plane:02d}/'
                        if not os.path.isdir(save_dir):
                            os.makedirs(save_dir)
                        output_filename = os.path.basename(path_input_file[:-4] + '_plane'f'{i_plane:02d}_Aligned.h5')
                        path_output_file = save_dir + output_filename
                        h5file = h5py.File(path_output_file, 'w') 
                        h5file.create_dataset('data', data=volume[:,:,:,i_plane])
                        h5file.attrs.create('metadata', str(metadata)) # You can use json to load it as a dictionary
                        h5file.close()
                        del h5file
                    
                if json_logging: json_logger.debug(json.dumps({'debug_message':'File saved: ' + path_output_file}))
                
                
         
            #%% Save mean frame png
            if save_meanf_png:
                if not meanf_png_only_first_file or first_file_of_recording:
                    if json_logging: json_logger.debug(json.dumps({'debug_message':'Saving png...'}))
                    canvas_png = np.zeros(((volume.shape[1] + gaps_rows) * rows - gaps_rows,
                                           (volume.shape[2] + gaps_columns) * columns - gaps_columns),
                                          dtype = np.uint8)
                    volume_meanf = np.nanmean(volume, axis = 0)
                    for i_plane in range(n_planes):
                        plane_for_png = copy.deepcopy(volume_meanf[:,:,i_plane])
                        #Normalize to [0,1]
                        plane_for_png -= np.nanmin(plane_for_png)
                        plane_for_png = plane_for_png / np.nanmax(plane_for_png)
                        
                        #Apply percentile-dynamic-range
                        pct_low, pct_high = np.nanpercentile(plane_for_png, intensity_percentiles)
                        plane_for_png = skimage.exposure.rescale_intensity(plane_for_png, in_range=(pct_low, pct_high))
        
                        #Rescale and transform to uint8
                        plane_for_png = np.round(plane_for_png * 255)
                        plane_for_png = plane_for_png.astype(np.uint8)
                        #Place it on canvas
                        x_start = i_plane % columns * (plane_for_png.shape[1]+gaps_columns)
                        y_start = i_plane // columns * (plane_for_png.shape[0]+gaps_rows)
                        x_end = x_start + plane_for_png.shape[1]
                        y_end = y_start + plane_for_png.shape[0]
                        canvas_png[y_start:y_end, x_start:x_end] = plane_for_png
                    fig = plt.figure(dpi=1200)
                    plt.imshow(canvas_png, cmap='gray')
                    title_png_figure = os.path.dirname(path_output_file) + '\n' + os.path.basename(path_output_file) #One line for directory, another for filename
                    plt.title(title_png_figure,fontsize=4)
                    plt.xticks(fontsize=4); plt.yticks(fontsize=4)
                    fig.tight_layout()
                    plt.show()
                    output_filename_meanf_png = path_output_file[:-3] + '.png'
                    fig.savefig(output_filename_meanf_png,bbox_inches='tight')
                    del canvas_png, volume_meanf
            
            #%% Save movie
            if save_mp4:
                if json_logging: json_logger.debug(json.dumps({'debug_message':'Saving mp4'}))
                metadata_software = metadata["Software"].split()
                for i_line in range(len(metadata_software)):
                    this_line = metadata_software[i_line]
                    if "SI.hRoiManager.scanFrameRate" in this_line:
                        #The next line is the '=' and the next after that is the frame-rate
                        frame_rate = float(metadata_software[i_line+2])
                fps = frame_rate * video_play_speed
                if video_duration_secs != 0:
                    use_until_frame_n = round(fps * video_duration_secs) # -1 for entire recording
                else:
                    use_until_frame_n = -1
                canvas_video = np.zeros((volume[:use_until_frame_n].shape[0] - rolling_average_frames + 1, 
                                   (volume.shape[1] + gaps_rows) * rows - gaps_rows,
                                   (volume.shape[2] + gaps_columns) * columns - gaps_columns),
                                   dtype = np.uint8)
                if not video_only_first_file or first_file_of_recording:
                    for i_plane in range(n_planes):
                        plane_for_video = copy.deepcopy(volume[:,:,:,i_plane])
                        plane_for_video = plane_for_video[:use_until_frame_n]
                        #Apply rolling average by convolving
                        plane_for_video = scipy.signal.convolve(plane_for_video, np.ones(([rolling_average_frames,1,1])), mode='valid')
                        #Apply percentile-dynamic-range
                        pct_low, pct_high = np.nanpercentile(plane_for_video, intensity_percentiles)
                        plane_for_video = skimage.exposure.rescale_intensity(plane_for_video, in_range=(pct_low, pct_high))
                        #Normalize to [0,1]
                        plane_for_video -= np.nanmin(plane_for_video)
                        plane_for_video = plane_for_video / np.nanmax(plane_for_video)
                        #Rescale and transform to uint8
                        plane_for_video = np.round(plane_for_video * 255)
                        plane_for_video = plane_for_video.astype(np.uint8)
                        #Place it on canvas
                        x_start = i_plane % columns * (plane_for_video.shape[2]+gaps_columns)
                        y_start = i_plane // columns * (plane_for_video.shape[1]+gaps_rows)
                        x_end = x_start + plane_for_video.shape[2]
                        y_end = y_start + plane_for_video.shape[1]
                        canvas_video[:, y_start:y_end, x_start:x_end] = plane_for_video
                    size_frame_video = (canvas_video.shape[2],canvas_video.shape[1])
                    output_filename_video = path_output_file[:-3] + '_RollingAvg' + str(rolling_average_frames) + 'Frames_Speed' + str(video_play_speed) + 'x.mp4'
                    out = cv2.VideoWriter(output_filename_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size_frame_video, False)
                    for f in range(canvas_video.shape[0]):
                        out.write(canvas_video[f])
                    out.release()
                del canvas_video
                
        #%% This file is done
            
        toc = time.time()
        if json_logging: json_logger.debug(json.dumps({'debug_message':'File processed and outputs saved. Time elapsed: ' + str(toc-tic)}))
        del volume
    
    
    
    # import caiman
    # import caiman.source_extraction.cnmf as caiman_cnmf
    # if current_pipeline_step in [1,2]:
    #     for i_file in range(len(path_output_files)):

    #         file_for_mc = path_output_files[i_file]
            
    #         # Define parameters
    #         if i_file == 0:
    #             n_frames = h5py.File(file_for_mc,'r')['data'].shape[0]
    #             mc_n_frames_per_thread = int(np.ceil(n_frames / mc_n_cpu_threads))
                
    #             opts_dict = {'fnames': file_for_mc,
    #             'strides': mc_strides,    # start a new patch for pw-rigid motion correction every x pixels
    #             'overlaps': mc_overlaps,   # overlap between pathes (size of patch is strides+overlaps)
    #             'max_shifts': mc_max_shifts,   # maximum allowed rigid shifts (in pixels)
    #             'max_deviation_rigid': mc_max_deviation_rigid,  # maximum shifts deviation allowed for patch with respect to rigid shifts
    #             'pw_rigid': mc_pw_rigid,   # flag for performing non-rigid motion correction
    #             'is3D': True,
    #             'num_frames_split': mc_n_frames_per_thread,
    #             'min_mov': 0, # Set to 0 to prevent having different baselines across time chunks
    #             'nonneg_movie': False} # Set to False to prevent having different baselines across time chunks
    #             opts = caiman_cnmf.params.CNMFParams(params_dict=opts_dict)
            
    #             c, dview, n_processes = caiman.cluster.setup_cluster(
    #                 backend='ipyparallel', n_processes=mc_n_cpu_threads, single_thread=False)
                
    #             print('Waiting 60 secs for ipyparallel cluster initialization')
    #             time.sleep(60)
    #         #Create a MC object 
    #         mc_object = caiman.motion_correction.MotionCorrect(file_for_mc, dview=dview, **opts.get_group('motion'))
            
    #         #Now run MC... warnings are expected
    #         if i_file == 0:
    #             mc_object.motion_correct(save_movie=True)
    #             Yr, dims, T = caiman.load_memmap(mc_object.mmap_file[0])
    #             mc_volume = np.reshape(Yr.T, [T] + list(dims), order='F')
    #             mc_template = np.mean(mc_volume, axis = 0)
    #             del mc_volume
    #         else:
    #             mc_object.motion_correct(save_movie=True, template = mc_template)
    #         time.sleep(10)
    #         # caiman.stop_server(dview=dview)
    #         # time.sleep(10)
            
    #         # mc_template_original = copy.deepcopy(mc_template)
            
    #         import caiman
    #         import copy
    #         import numpy as np
    #         import matplotlib.pyplot as plt
            
    #         path_dir = '/home/freiwald/Data/analysis_2pRAM/Coconut/20230107d/Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW/'
    #         mmap_fname = 'Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00026_Aligned_els__d1_341_d2_363_d3_30_order_F_frames_1882.mmap'
    #         h5_fname = 'Max30_500umdeep_1p2by1p2mm_3umppix_18p82Hz_250mW_00001_00016_Aligned.h5'
            
    #         path_mmfile = path_dir + mmap_fname
    #         path_h5file = path_dir + h5_fname
            
    #         Yr, dims, T = caiman.load_memmap(path_mmfile)
    #         mc_volume = copy.deepcopy(np.reshape(Yr.T, [T] + list(dims), order='F'))
    #         mc_template_26 = np.mean(mc_volume, axis = 0)
    #         np.mean(mc_volume)
            
    #         mc_template_22 = (mc_template_06 + mc_template_16) / 2
    #         for p in range(30):
    #             plt.imshow(mc_template_06[:,:,p])
    #             plt.title(str(p) + '_06')
    #             plt.show()
    #             plt.imshow(mc_template_16[:,:,p])
    #             plt.title(str(p) + '_16')
    #             plt.show()
    #             plt.imshow(mc_template_26[:,:,p])
    #             plt.title(str(p) + '_26')
    #             plt.show()
                
    #         # file_reshaped = h5py.File(path_h5file,'r')['data']
    #         # file_reshaped[0,:,:,0]
    #         # np.mean(file_reshaped)
                
    #         # plt.imshow(mc_template_original[:,:,2])
    #         # plt.imshow(mc_template[:,:,20])
            
    #         # plt.imshow(mc_template_original[:,:,20])
    #         # plt.imshow(np.mean(n_frames[:,:,:,6], axis = 0))
            
        
        
        
        
        
        
        
        
        
        
    
            