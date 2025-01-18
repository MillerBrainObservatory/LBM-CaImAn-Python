import sys

from qtpy.QtWidgets import QApplication
from lbm_caiman_python.gui.widgets import LBMMainWindow


def run_gui(path):
    app = QApplication(sys.argv)
    main_window = LBMMainWindow(path)
    print('--')
    main_window.show()
    app.exec()
    # fpl.loop.run()
