from pathlib import Path
import lbm_mc as mc
import lbm_caiman_python as lcp
from lbm_mc.caiman_extensions.cnmf import cnmf_cache


def main():
    parent_path = Path(r"D:\W2_DATA\kbarber\2025-01-30\mk303\green")
    assembled_path = parent_path.joinpath("assembled")
    batch_path = parent_path.joinpath("lbm_caiman_python", "runs")
    batch_path.mkdir(parents=True, exist_ok=True)

    mc.set_parent_raw_data_path(assembled_path)
    df = mc.load_batch(batch_path / 'results.pickle')

    df.caiman.add_item(
        algo='cnmf',
        input_movie_path=df.iloc[-1],
        params=df.iloc[-1].params,
        item_name=df.iloc[-1].item_name,  # filename of the movie, but can be anything
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
