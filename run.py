# %%
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

import scanreader


def plot_frame(frame_arr, frame=1, plane=21, savepath=None):
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


filepath = Path("/data2/fpo/data")
files = [x for x in filepath.glob("*.tif")]
# data, metadata = extract(filepath)

scan = extract_scanreader(str(files[0]))
data = scan[0]
plot_frame(data, 2, 21, '/data2/fpo/data/test2.png')

x = 2
