from pathlib import Path
import dask.array as da
import numpy as np
import tifffile
import fastplotlib as fpl
import matplotlib.pyplot as plt
import lbm_mc as mc
import lbm_caiman_python as lcp
from tqdm.notebook import tqdm

from lbm_mc.caiman_extensions.cnmf import cnmf_cache


def main():
    parent_path = Path(r"D:\W2_DATA\kbarber\2025-01-30\mk303\green")
    assembled_path = parent_path.joinpath("assembled")
    batch_path = parent_path.joinpath("lbm_caiman_python", "runs")
    batch_path.mkdir(parents=True, exist_ok=True)

    mc.set_parent_raw_data_path(assembled_path)
    df = mc.create_batch(batch_path / 'results.pickle', remove_existing=True)

    files = lcp.get_files(assembled_path, str_contains='tif')
    metadata = lcp.get_metadata(files[0])

    df.caiman.add_item(
        algo='mcorr',
        input_movie_path=files[6],
        params=lcp.params_from_metadata(metadata),
        item_name=Path(files[6]).stem,  # filename of the movie, but can be anything
    )
    process = df.iloc[-1].caiman.run("local")
    process.wait()
    df = df.caiman.reload_from_disk()
    item = df.iloc[-1]
    tb = item.outputs["traceback"]
    if tb is not None:
        print(tb)


if __name__ == "__main__":
    main()
