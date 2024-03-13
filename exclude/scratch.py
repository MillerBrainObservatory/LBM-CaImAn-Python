#%% Libraries and Params

import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import tifffile

plt.rcParams['figure.dpi'] = 900

params = dict()
chan_order = np.array([ 1,  5,  6,  7,  8,  9,  2, 10, 11, 12, 13, 14, 15, 16, 17,
                                            3, 18, 19, 20, 21, 22, 23,  4, 24, 25, 26, 27, 28, 29, 30]) - 1
# Parameters output planes/volumes
params['save_output'] = True
if params['save_output']:
    params['save_as_volume_or_planes'] = 'planes' # 'planes' will save individual planes in subfolders -- 'volume' will save a whole 4D hdf5 volume
    if params['save_as_volume_or_planes'] == 'planes':
        params['concatenate_all_h5_to_tif'] = False # If True, it will take all the time-chunked h5 files, concatenate, and save them as a single .tif

params['make_nonan_volume'] = True # Whether to trim the edges so the output does not have nans. Also affects output as planes if lateral_aligned_planes==True (to compensate for X-Y shifts of MAxiMuM) or params['identical_mroi_overlaps_across_planes']==False (if seams from different planes are merged differently, then some planes will end up being larger than others)
params['lateral_align_planes'] = False # Calculates and compensates the X-Y of MAxiMuM
params['add_1000_for_nonegative_volume'] = True

# Parameters MROIs seams
params['seams_overlap'] = 'calculate' # Should be either 'calculate', an integer, or a list of integers with length=n_planes
if params['seams_overlap'] == 'calculate':
    params['n_ignored_pixels_sides'] = 5 # Useful if there is a delay or incorrect phase for when the EOM turns the laser on/off at the start/end of a resonant-scanner line
    params['min_seam_overlap'] = 5
    params['max_seam_overlap'] = 20 # Used if params['seams_overlap']_setting = 'calculate'
    params['alignment_plot_checks'] = False

#%%
# Look for files used to:
# 1) Make a template to do seam-overlap handling and X-Y shift alignment
# 2) Reconstruct MROI volumes
path_input_file = "/v-data4/foconnell/data/lbm/raw"
files = sorted(glob.glob(path_input_file + '/**/*.tif', recursive=True))
file = files[0]
n_planes = 30
rows, columns = 6, 5

#%% Load Metadata
with tifffile.TiffFile(file) as tif:
    metadata = {}
    for tag in tif.pages[0].tags.values():
        tag_name, tag_value = tag.name, tag.value
        metadata[tag_name] = tag_value

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
else:
    scanfield = mrois_si_raw['scanfields']
    roi_dict = {}
    roi_dict['center'] = np.array(scanfield['centerXY'])
    roi_dict['sizeXY'] = np.array(scanfield['sizeXY'])
    roi_dict['pixXY'] = np.array(scanfield['pixelResolutionXY'])
    mrois_si = [roi_dict]

mrois_centers_si = np.array([mroi_si['center'] for mroi_si in mrois_si])
x_sorted = np.argsort(mrois_centers_si[:, 0])
mrois_si_sorted_x = [mrois_si[i] for i in x_sorted]
mrois_centers_si_sorted_x = [mrois_centers_si[i] for i in x_sorted]

#%% Load Tiff
tiff_file = tifffile.imread(file)
tiff_file = np.reshape(tiff_file, (int(tiff_file.shape[0]/n_planes), n_planes, tiff_file.shape[1],tiff_file.shape[2]), order='A') # warnings are expected if the recording is split into many files or incomplete

tiff_file = np.swapaxes(tiff_file, 1, 3)
tiff_file = tiff_file[..., chan_order]

#%% Load Zarr
tiff_file_zarr = tifffile.imread(file, aszarr=True)
