import time

import numpy as np
import tifffile

import util


def extract(fpath, metad=None):
    files = [x for x in fpath.glob("*.tif")]
    metad = util.get_metadata_from_tiff(files[0]) if metad is None else metad

    start = time.time()
    tiff_file = tifffile.imread(files[0])
    if len(tiff_file.shape) == 3:
        if metad["num_planes"] * metad["num_frames_total"] == tiff_file.shape[0]:
            num_frames = metad["num_frames_total"]
        elif metad["num_planes"] * metad["num_frames_file"] == tiff_file.shape[0]:
            num_frames = metad["num_frames_file"]
        else:
            raise ValueError("Number of frames in tiff file does not match metadata")

        tiff_file = np.reshape(
            tiff_file,
            (
                num_frames,
                metad["num_planes"],
                tiff_file.shape[1],
                tiff_file.shape[2],
            ),
        )
    stop = time.time()
    print(f"Time to read tiff file: {stop - start:.1f} seconds")

    metad["file_size"] = (tiff_file.size * 2) / 1e9
    metad["data_shape"] = tiff_file.shape
    metad["data_type"] = tiff_file.dtype
    tiff_file = np.swapaxes(tiff_file, 1, 3)
    return tiff_file, metad


__all__ = ["util"]

if __name__ == "__main__":
    from pathlib import Path

    filepath = Path("/data2/fpo/data")
    data, metadata = extract(filepath)

    # print all files / dirs in filepath
    file = [x for x in filepath.glob("*.tif")]
    mdata = util.get_metadata_from_tiff(file[0])
