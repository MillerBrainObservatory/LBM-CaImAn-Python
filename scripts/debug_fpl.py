from pathlib import Path
import dask.array as da
import numpy as np
import tifffile
import fastplotlib as fpl
import lbm_mc as mc
import lbm_caiman_python as lcp
from tqdm.notebook import tqdm

from lbm_mc.caiman_extensions.cnmf import cnmf_cache


def main():
    parent_path = r"D:\DATA\2025-01-16_GCaMP8s_tdtomato_mk302\mk302_2umpx_17p07hz_224pxby448px_2mroi_200mw_50to550umdeep_GCaMP8s_CavAB_00001"
    raw_path = Path(parent_path) / 'zplanes'
    batch_path = Path(parent_path) / 'registered'
    results_path = batch_path / 'registered.pickle'
    df = mc.load_batch(results_path)

    files = lcp.get_files(raw_path, ".tif", 1)
    #files = sorted(files, key=lambda p: int(Path(p).stem.split("_")[-1]))
    data = tifffile.memmap(files[0])
    mc.set_parent_raw_data_path(raw_path)
    raw = fpl.ImageWidget(data, histogram_widget=False)
    raw.show()


if __name__ == "__main__":
    main()
    fpl.loop.run()
