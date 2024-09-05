from pathlib import Path
from lbm_caiman_python import read_scan

if __name__ == "__main__":

    path = Path().home() / 'caiman_data' / 'animal_01' / 'session_01'
    savedir = path / 'pre_processed'
    reader = read_scan(path, trim_roi_x=(5, 5), trim_roi_y=(17, 0))

    x = 5
