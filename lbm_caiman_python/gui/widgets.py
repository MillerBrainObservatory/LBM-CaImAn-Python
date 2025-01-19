import webbrowser

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QMainWindow, QFileDialog
from qtpy import QtGui, QtCore
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow

from lbm_caiman_python.lcp_io import get_files_ext, stack_from_files

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
    files = get_files_ext(path, "plane", 1)
    zstack = stack_from_files(files)
    iw = fpl.ImageWidget(zstack, histogram_widget=False)
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
        gui = MenuWidget(self, size=50)
        self._image_widget.figure.add_gui(gui)
        qwidget = self._image_widget.show()
        self.setCentralWidget(qwidget)
        self.resize(1200, 800)

    def update_widget(self, path):
        print('Updating image widget')

        # get a new MenuWidget instance
        new_gui = MenuWidget(self, size=50)

        # get a new ImageWidget instance
        image_widget = get_iw(path)

        # add the new ImageWidget to the main window
        image_widget.figure.add_gui(new_gui)

        # start the render loop
        qwidget = image_widget.show()

        self._image_widget.close()

        self.setCentralWidget(qwidget)

        # delete the old ImageWidget
        self._image_widget = image_widget


class MenuWidget(EdgeWindow):
    def __init__(self, parent, size):
        flags = imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize
        super().__init__(
            figure=parent.image_widget.figure,
            size=size,
            location="top",
            title="Toolbar",
            window_flags=flags,
        )
        self.parent = parent

    def update(self):

        if imgui.button("Documentation"):
            webbrowser.open(
                "https://millerbrainobservatory.github.io/LBM-CaImAn-Python/"
            )

        imgui.same_line()

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

        imgui.same_line()
