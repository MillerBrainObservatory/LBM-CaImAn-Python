# %%
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import scanreader
import util


def plot_frame(frame_arr, frame=1, plane=21, savepath=None):
    if len(frame_arr.shape) == 2:
        plt.imshow(frame_arr)
    else:
        plt.imshow(frame_arr[:, :, plane, frame])
    if savepath:
        plt.savefig(str(savepath))
        print('figure saved')


def extract_scanreader(fpath):
    return scanreader.read_scan(fpath, join_contiguous=True)


def save_dict_to_pickle(data_dict, file_path):
    """
    Saves a dictionary with arrays and nested dictionaries to a pickle file.

    Parameters:
    data_dict (dict): The dictionary to save. Can contain various data types including arrays and other dictionaries.
    file_path (str): The path to the file where the dictionary will be saved.

    Returns:
    None
    """
    with open(file_path, "wb") as file:
        pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    filepath = Path("/data2/fpo/data")
    files = [x for x in filepath.glob("*.tif")]
    # data, metadata = extract(filepath)

    data_corr = util.galvo_corrections.compute_raster_phase()

    scan = extract_scanreader(str(files[0]))
    data = scan[0]
    data_plane_21 = data[:,:,20,1000:]
    mean_img = np.mean(data_plane_21, axis=2)
    plot_frame(mean_img, savepath='/data2/fpo/data/meanimg_frame1000_end.png')



    x = 2
