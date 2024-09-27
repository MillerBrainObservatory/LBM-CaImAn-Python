from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from pathlib import Path
import zarr
import mesmerize_core as mc
from mesmerize_viz import *

parent_dir = Path().home() / 'caiman_data' / 'animal_01' / 'session_01'
results_path = parent_dir / 'motion_corrected'

mc.set_parent_raw_data_path(parent_dir.parent)

try:
    df = mc.load_batch(parent_dir / 'batch.pickle')
except (IsADirectoryError, FileNotFoundError):
    df = mc.create_batch(parent_dir / 'batch.pickle')
df = df.caiman.reload_from_disk()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    mcorr_container = df.mcorr.viz(start_index=1)
    mcorr_container.show()
    exit(app.exec_())
# cnmf_container = df.cnmf.viz(start_index=2)