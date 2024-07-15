import sys
from pathlib import Path
import os
from pathlib import Path
import numpy as np
import cv2

sys.path.append('../../util/')  # TODO: Take this out when we upload to pypi
sys.path.append('exclude/')  # TODO: Take this out when we upload to pypi
sys.path.append('')  # TODO: Take this out when we upload to pypi

import matplotlib.pyplot as plt

try:
    cv2.setNumThreads(0)
except():
    pass
import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.source_extraction.cnmf import cnmf, params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from caiman.utils.visualization import view_quilt

import scanreader as sr

#%% {markdown} User input: data_path
data_path = Path("/mnt/c/Users/RBO/Documents/data/high_res/")
raw = [x for x in data_path.glob(f'*.tif*')][0]

ext = "tif"
reader = sr.read_scan(str(raw), join_contiguous=True, lbm=True, x_cut=(6, 6), y_cut=(17, 0))

if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='multiprocessing', n_processes=None, single_thread=False)

plane = reader[:, :, :, 0, 1:400].squeeze()
plane = np.transpose(plane, (2, 0, 1))

# %%

pix_res = 1
mx = 10 / pix_res

max_shifts = (mx, mx)  # maximum allowed rigid shift in pixels
max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = True  # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)

# %%
# Set parameters for rigid motion correction
options_rigid = {
    'max_shifts': max_shifts,       # max shift in pixels
    'strides': (48, 48),        # create a new patch every x pixels for pw-rigid correction
    'overlaps': (24, 24),       # overlap between patches (size of patch strides+overlaps)
    'max_deviation_rigid': 3,   # maximum deviation allowed for patch with respect to rigid shifts
    'pw_rigid': False,          # flag for performing rigid or piecewise rigid motion correction
    'shifts_opencv': True,      # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    'border_nan': 'copy',       # replicate values along the boundary (if True, fill in with NaN)
    'nonneg_movie': False
}

mc_rigid = MotionCorrect(plane, dview=dview, **options_rigid)
# mc = MotionCorrect(movie, dview=dview, **options_rigid)
mc_rigid.motion_correct(save_movie=True)

# %%
mc_file = Path().home() / 'Documents' / 'data' / 'high_res' / 'raw_p1_tp.tif'

# M1 = mc_rigid.apply_shifts_movie(mc_file)
mc_rigid
# %%
# load motion corrected movie
m_rig = cm.load(mc_rigid.mmap_file)
bord_px_rig = np.ceil(np.max(mc_rigid.shifts_rig)).astype(int)
##%% visualize templates
plt.figure(figsize=(20, 10))
plt.imshow(mc_rigid.total_template_rig, cmap='gray');

# %% md

m_rig.resize(1, 1, 0.2).play(
    q_max=99.5, fr=30, magnification=2, bord_px=0 * bord_px_rig)  # press q to exit

##%% plot rigid shifts

plt.close()
plt.figure(figsize=(20, 10))
plt.plot(mc_rigid.shifts_rig)
plt.legend(['x shifts', 'y shifts'])
plt.xlabel('frames')
plt.ylabel('pixels');
# %%
# correct for rigid motion correction and save the file (in memory mapped form)
mc.motion_correct(save_movie=True)
mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)
mc.motion_correct(save_movie=True, template=mc.total_template_rig)
m_els = cm.load(mc.fname_tot_els)
m_els.resize(1, 1, 0.2).play(
    q_max=99.5, fr=30, magnification=2, bord_px=bord_px_rig)
# %% md
# visualize non-rigid shifts for the entire FOV
#
# TODO: Interactively visualize rigid+non-rigid shifts independantly
# %%
plt.close()
plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(mc.x_shifts_els)
plt.ylabel('x shifts (pixels)')
plt.subplot(2, 1, 2)
plt.plot(mc.y_shifts_els)
plt.ylabel('y_shifts (pixels)')
plt.xlabel('frames')
##%% compute borders to exclude
bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                 np.max(np.abs(mc.y_shifts_els)))).astype(int)
# %% md
# ## Motion Corretion: Optical Flow
# %%
##%% plot the results of Residual Optical Flow
fls = [cm.paths.fname_derived_presuffix(mc.fname_tot_els[0], 'metrics', swapsuffix='npz'),
       cm.paths.fname_derived_presuffix(mc.fname_tot_rig[0], 'metrics', swapsuffix='npz'),
       cm.paths.fname_derived_presuffix(mc.fname[0], 'metrics', swapsuffix='npz'),
       ]

plt.figure(figsize=(20, 10))
for cnt, fl, metr in zip(range(len(fls)), fls, ['pw_rigid', 'rigid', 'raw']):
    with np.load(fl) as ld:
        print(ld.keys())
        print(fl)
        print(str(np.mean(ld['norms'])) + '+/-' + str(np.std(ld['norms'])) +
              ' ; ' + str(ld['smoothness']) + ' ; ' + str(ld['smoothness_corr']))

        plt.subplot(len(fls), 3, 1 + 3 * cnt)
        plt.ylabel(metr)
        print(f"Loading data with base {fl[:-12]}")
        try:
            mean_img = np.mean(
                cm.load(fl[:-12] + '.mmap'), 0)[12:-12, 12:-12]
        except:
            try:
                mean_img = np.mean(
                    cm.load(fl[:-12] + '.tif'), 0)[12:-12, 12:-12]
            except:
                mean_img = np.mean(
                    cm.load(fl[:-12] + 'hdf5'), 0)[12:-12, 12:-12]

        lq, hq = np.nanpercentile(mean_img, [.5, 99.5])
        plt.imshow(mean_img, vmin=lq, vmax=hq)
        plt.title('Mean')
        plt.subplot(len(fls), 3, 3 * cnt + 2)
        plt.imshow(ld['img_corr'], vmin=0, vmax=.35)
        plt.title('Corr image')
        plt.subplot(len(fls), 3, 3 * cnt + 3)
        flows = ld['flows']
        plt.imshow(np.mean(
            np.sqrt(flows[:, :, :, 0] ** 2 + flows[:, :, :, 1] ** 2), 0), vmin=0, vmax=0.3)
        plt.colorbar()
        plt.title('Mean optical flow');
    # %% md
# # Cleanup
# 
# Make sure our parallel cluster is shut down.
# %%
if 'dview' in locals():
    cm.stop_server(dview=dview)
elif 'cluster' in locals():
    cm.stop_server(dview=cluster)