import webbrowser

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QMainWindow, QFileDialog
from qtpy import QtGui, QtCore
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow

from mbo_utilities import get_files, imread

try:
    from imgui_bundle import imgui, icons_fontawesome_6 as fa
except ImportError:
    raise ImportError("Please install imgui via `conda install -c conda-forge imgui-bundle`")


def get_base_iw():
    """Temp until I figure out how to start with an empty canvas"""
    rand = np.random.randn(100, 100, 100)
    iw = fpl.ImageWidget(rand, histogram_widget=False)
    return iw


def get_iw(path):
    files = get_files(path, "plane", 1)
    arr = imread(files)
    iw = fpl.ImageWidget(arr, histogram_widget=False)
    return iw


class LBMMainWindow(QMainWindow):

    @property
    def image_widget(self):
        return self._image_widget

    def __init__(self):
        super(LBMMainWindow, self).__init__()

        print('Setting up main window')
        self.setGeometry(50, 50, 1500, 800)
        self.setWindowTitle("LBM-CaImAn-Python Pipeline")

        app_icon = QtGui.QIcon()
        icon_path = str(Path().home() / ".lbm" / "icons" / "icon_caiman_python.svg")
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(64, 64))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,50,100); "
                             "color:white;}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")
        self.styleInactive = ("QPushButton {Text-align: left; "
                              "background-color: rgb(50,50,50); "
                              "color:gray;}")

        print('Setting up image widget')
        self._image_widget = get_base_iw()
        gui = PreviewTracesWidget(size=50)
        self._image_widget.figure.add_gui(gui)
        qwidget = self._image_widget.show()
        self.setCentralWidget(qwidget)
        self.resize(1200, 800)


class PreviewTracesWidget(EdgeWindow):
    def __init__(self, figure, size, location, title, image_widget):
        super().__init__(figure=figure, size=size, location=location, title=title)
        self._image_widget = image_widget

        # whether or not a dimension is in play mode
        self._playing: dict[str, bool] = {"t": False, "z": False}

        self.tfig = fpl.Figure()

        self.raw_trace = self.tfig[0, 0].add_line(np.zeros(self._image_widget.data[0].shape[0]))
        self._image_widget.managed_graphics[0].add_event_handler("click")
        self.tfig.show()

    def pixel_clicked(self, ev):
        col, row = ev.pick_info["index"]
        if self._image_widget.ndim == 4:
            self.raw_trace.data[:, 1] = self._image_widget.data[0][:, self._image_widget.current_index["z"], row, col]
        elif self._image_widget.ndim == 3:
            self.raw_trace.data[:, 1] = self._image_widget.data[0][:, row, col]
        else:
            raise ValueError("ImageWidget has an unexpected number of dimensions. Expected 3 or 4.")
        self.tfig[0, 0].auto_scale(maintain_aspect=False)

    def update(self):

        imgui.push_font(self._fa_icons)
        if imgui.button(label=fa.ICON_FA_FOLDER_OPEN):
            print("Opening file dialog")
            dlg_kwargs = {
                "parent": self.parent,
                "caption": "Open folder with z-planes",
            }
            name = QFileDialog.getExistingDirectory(**dlg_kwargs)
            print(name)
            self.parent.update_widget(name)

        imgui.pop_font()
        if imgui.is_item_hovered(0):
            imgui.set_tooltip("Open a file dialog to load data")
