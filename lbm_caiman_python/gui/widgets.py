import webbrowser

from pathlib import Path
from qtpy.QtWidgets import QMainWindow, QFileDialog
from qtpy import QtGui, QtCore
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow

from lbm_caiman_python.lcp_io import get_files_ext, stack_from_files

try:
    from imgui_bundle import imgui, icons_fontawesome_6 as fa
except ImportError:
    raise ImportError("Please install imgui via `conda install -c conda-forge imgui-bundle`")


def load_dialog_folder(iw):
    dlg_kwargs = {
        "parent": None,
        "caption": "Open folder with z-planes",
    }
    name = QFileDialog.getExistingDirectory(**dlg_kwargs)
    iw.fname = name
    load_folder(iw)


def load_folder(iw):
    print(iw.fname)
    save_folder = Path(iw.fname)
    plane_folders_ext = get_files_ext(save_folder, "plane", 2)
    plane_folders = list(save_folder.rglob("*plane*"))
    if plane_folders:
        stack_from_files(plane_folders_ext)
        print("Found planeX folders in folder")
    else:
        print("No processed planeX folders in folder")
        return
    iw.data = stack_from_files(plane_folders_ext)


def get_iw(path, recursive=False):
    if recursive:
        files = get_files_ext(path, "plane", 2)
    files = get_files_ext(path, "plane", 2)
    zstack = stack_from_files(files)
    iw = fpl.ImageWidget(zstack, histogram_widget=False)
    return iw


class LBMMainWindow(QMainWindow):
    def __init__(self, path):
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
        self.image_widget = get_iw(path)
        self.menu_widget = MenuWidget(self.image_widget, size=50)

        gui = MenuWidget(self.image_widget, size=50)
        self.image_widget.figure.add_gui(gui)
        qwidget = self.image_widget.show()
        self.setCentralWidget(qwidget)
        self.resize(self.image_widget.data[0].shape[-2], self.image_widget.data[0].shape[-1])


class PreviewDataWidget(EdgeWindow):
    def __init__(self, image_widget, size):
        flags = imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize
        super().__init__(figure=image_widget.figure, size=size, location="right", title="Preview Data",
                         window_flags=flags)
        self.image_widget = image_widget
        self.sigma = 0
        self.mean_window_size = 0

    def update(self):
        if imgui.button("Open Pipeline Documentation"):
            webbrowser.open(
                "https://millerbrainobservatory.github.io/LBM-CaImAn-Python/"
            )

        imgui.new_line()
        imgui.separator()

        something_changed = False

        # slider for gaussian filter sigma value
        changed, value = imgui.slider_int(label="sigma", v=self.sigma, v_min=0, v_max=40)
        if changed:
            self.sigma = value
            something_changed = True

        # int entries for gaussian filter order
        changed, value = imgui.slider_int(f"Mean window", v=self.mean_window_size, v_min=0, v_max=20)
        if changed:
            self.mean_window_size = value
            something_changed = True

        # calculate stats and display in a text widget
        imgui.new_line()
        imgui.separator()
        imgui.text("Statistics")
        if imgui.button("Calculate Noise"):
            self.calculate_noise()
            # display loading bar
            imgui.text("Calculating noise...")
        imgui.new_line()
        imgui.separator()

        if something_changed:
            self.process_image()

    def process_image(self):

        self.image_widget.figure[0, 0].add_text(f"Window Size: {self.mean_window_size}")
        self.image_widget.figure[0, 0].add_text(f"Window Size: {self.mean_window_size}")
        # processed = gaussian_filter(self.image_widget.data, sigma=self.sigma, order=(self.order_y, self.order_x))
        pass  # self.image_widget.window_funcs = {"t": (np.mean, self.mean_window_size)}

    def calculate_noise(self):
        pass


class MenuWidget(EdgeWindow):
    def __init__(self, image_widget, size):
        flags = imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize
        super().__init__(
            figure=image_widget.figure,
            size=size,
            location="top",
            title="Toolbar",
            window_flags=flags,
        )
        self.image_widget = image_widget

    def update(self):

        if imgui.button("Documentation"):
            webbrowser.open(
                "https://millerbrainobservatory.github.io/LBM-CaImAn-Python/"
            )

        imgui.same_line()

        imgui.push_font(self._fa_icons)
        if imgui.button(label=fa.ICON_FA_FOLDER_OPEN):
            print("Opening file dialog")
            load_dialog_folder(self.image_widget)

        imgui.pop_font()
        if imgui.is_item_hovered(0):
            imgui.set_tooltip("Open a file dialog to load data")

        imgui.same_line()
