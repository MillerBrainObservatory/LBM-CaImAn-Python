# import cv2
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
import os
import logging
import h5py
import caiman as cm
from caiman.motion_correction import MotionCorrect
# from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
# from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
# from caiman.utils.visualization import plot_contours, inspect_correlation_pnr
import cv2

# limit OpenMP to a single thread
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# VECLIB_MAXIMUM_THREADS=1 # Need this on OSX
dview = None

logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    filename=Path().home() / 'data' / "motion_correction.log",
                    level=logging.DEBUG
                    )
# initialize the logger
logger = logging.getLogger(__name__)

plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['grid.color'] = 'grey'
plt.rcParams['legend.edgecolor'] = 'white'
plt.rcParams['legend.facecolor'] = 'black'

sns.set(rc={'axes.facecolor': 'black', 'grid.color': 'grey'})


def setup_cluster(n_processes=16):
    global dview
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    _, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=n_processes, single_thread=False)
    return dview, n_processes


def teardown_cluster(local_dview):
    cm.stop_server(dview=local_dview)


def motion_correct(fnames, pixel_resolution, savedir, local_dview, plot):
    ext_name = Path(fnames).stem
    logger.info('Starting motion correction: Rigid')
    start_time = time.time()
    max_shift = int(20 / pixel_resolution)
    fnames = str(fnames)
    rigid_options = {
        'fnames': fnames,  # This will be turned to a list if it is given as a string
        'pw_rigid': False,  # For the first iteration, we want rigid motion correction
        'max_shifts': (max_shift, max_shift),
        'upsample_factor_grid': 20,
    }
    # default init_batch is 200
    opts = params.CNMFParams(params_dict=rigid_options)

    mc = MotionCorrect(fnames, var_name_hdf5='dataset_name', dview=local_dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)

    logger.info('Rigid motion correction finished. Saving memmory mapped file')
    # memmap the original file
    try:
        rigid_file = cm.save_memmap(
            [mc.mmap_file],
            base_name=f'{savedir}/rigid_{ext_name}',
            order='C',
            dview=local_dview
        )
        logger.info(f"Saved rigid motion corrected file: {rigid_file}")
    except Exception as e1:
        logger.error(f"Failed to save rigid motion corrected file: {e1}")
        pass

    if plot:
        logger.info('Plotting shifts')
        try:
            fig, ax = plt.subplots(2, 1, figsize=(40, 20))  # Adjust the figsize to get the suitable plot size
            # Top plot for template
            ax[0].imshow(mc.total_template_rig)
            ax[0].set_title('Template', fontdict={'fontsize': '16', 'fontweight': 'bold'})

            ax[1].plot(mc.shifts_rig)
            ax[1].legend(['x shifts', 'y shifts'], fontsize='12', loc='upper right')
            ax[1].set_xlabel('Frames', fontsize='14', fontweight='bold')
            ax[1].set_ylabel('Pixels', fontsize='14', fontweight='bold')
            ax[1].set_title('Shifts', fontdict={'fontsize': '16', 'fontweight': 'bold'})

            # Save the plot
            plt.tight_layout()
            plt.savefig(savedir / f'rigid_motion_correction_{ext_name}.png', dpi=300)
            logger.info(f"Saved rigid motion correction plot: {f'rigid_motion_correction_{Path(fnames).stem}.png'}")
        except Exception as e3:
            logger.error(f"Failed to save rigid motion correction plot: {e3}")

    # Non-rigid motion correction
    logger.info('Starting motion correction: Non-rigid')
    mc.pw_rigid = True
    mc.motion_correct_pwrigid(save_movie=True, template=mc.total_template_rig)
    m_els = cm.load(mc.fname_tot_els)
    logger.info('Non-rigid motion correction finished. Saving memmory mapped file')
    try:
        cm.save_memmap(
            [m_els],
            base_name=f'{savedir}/non_rigid_{ext_name}',
            order='C',
            dview=local_dview
        )
    except Exception as e2:
        cm.save_memmap([m_els],
                       base_name=f'{savedir}/non_rigid_{ext_name}_secondtry',
                       order='C',
                       dview=local_dview)
        logger.error(f"Failed to save non-rigid motion corrected file: {e2}")
        pass

    plt.close()
    plt.figure(figsize=(40, 20))
    plt.subplot(2, 1, 1)
    plt.plot(mc.x_shifts_els)
    plt.ylabel('x shifts (pixels)')
    plt.subplot(2, 1, 2)
    plt.plot(mc.y_shifts_els)
    plt.ylabel('y_shifts (pixels)')
    plt.xlabel('frames')
    bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(int)
    plt.savefig(savedir / f'non_rigid_motion_correction_{ext_name}.png', dpi=300)

    end = time.time()
    print(f"Mc for {ext_name} took {end - start_time} seconds.")
    return local_dview


def correlation_pnr(filename):
    from caiman.utils.visualization import inspect_correlation_pnr

    # load memory mappable file
    Yr, dims, T = cm.load_memmap(filename)
    images = Yr.T.reshape((T,) + dims, order='F')

    # compute some summary images (correlation and peak to noise)
    cn_filter, pnr = cm.summary_images.correlation_pnr(images[::5],
                                                       gSig=3,
                                                       swap_dim=False)
    # inspect the summary images and set the parameters
    inspect_correlation_pnr(cn_filter, pnr)


def load_memmap_file(fname_new):
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order="F")
    print("number of images: ", images.shape[0])
    return images


def cnmf(filename):
    pass


if __name__ == "__main__":
    plane_path = Path().home() / "data" / "lbm" / "planes"
    file_names = [f for f in plane_path.glob('*.mmap') if 'els' in f.stem]
    plane_file = file_names[0]
    file_mem = cm.load(str(plane_file))
    shift_path = Path().home() / "data" / "lbm" / "figs" / "raw_shifts"
    shift_file = [f for f in shift_path.glob('*.mmap') if 'non' in f.stem]
    shift_mem = cm.load(str(shift_file[0]))
    correlation_pnr(str(plane_file))
    plt.show()
    plt.imshow(file_mem)
    print('done')
