import pandas as pd

import lbm_caiman_python
from pathlib import Path
import lbm_mc as mc
from caiman.utils.visualization import get_contours as caiman_get_contours
from lbm_caiman_python.visualize import export_contours_with_params


def main():
    assembled_path = r"D:\W2_DATA\kbarber\2025-01-30\mk303\green\assembled"
    batch_path = r"D:\W2_DATA\kbarber\2025-01-30\mk303\green\lbm_caiman_python\results\eval.pickle"
    mc_path = r"D:\W2_DATA\kbarber\2025-01-30\mk303\green\lbm_caiman_python\results\results.pickle"
    mc.set_parent_raw_data_path(assembled_path)
    df = mc.load_batch(batch_path)
    df_mc = mc.load_batch(mc_path)
    params = df_mc.iloc[6].params
    params['main']['K'] = 1
    df.caiman.add_item(
        input_movie_path=df_mc.iloc[6],
        params=params,
        item_name="plane_7",
        algo="cnmf",
    )
    df.iloc[-1].caiman.run('local')
    df = df.caiman.reload_from_disk()
    out = df.iloc[-1].outputs["traceback"]
    if not out:
        print('done')
        return False
    else:
        print(out)
        return True


if __name__ == "__main__":
    res = main()
    print(res)
