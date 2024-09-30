import os
from pathlib import Path

import dask.array as da
import numpy as np
import zarr
from dask_image import imread
from magicgui import magicgui
from qtpy import QtCore
from qtpy.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import napari
from napari.qt import get_stylesheet
from napari.settings import get_settings
from superqt import QRangeSlider

import scanreader
from tifffile import tifffile


# class MyDialog(QDialog):
#     def __init__(self, parent=None) -> None:
#         super().__init__(parent)
#         self.first_input = QSpinBox()
#         self.second_input = QSpinBox()
#         layout = QGridLayout()
#         # layout.addWidget(QLabel('Trim ROI Edges'), 0, 0)
#         layout.addWidget(QLabel('Left Edge:'), 0, 0)
#         layout.addWidget(self.first_input, 0, 1)
#         layout.addWidget(QLabel('Right Edge:'), 1, 0)
#         layout.addWidget(self.second_input, 1, 1)
#         layout.addWidget(self.btn, 2, 0, 1, 2)
#         self.setLayout(layout)
#         self.first_input.events.clicked.connect(self.run)
#


def get_slice(reader, y, x, z, t):
    return reader[:, y, x, z, t]


import sys
from PyQt5.QtCore import (Qt, pyqtSignal)
from PyQt5.QtWidgets import (QWidget, QLCDNumber, QSlider,
                             QVBoxLayout, QApplication)
from PyQt5.QtCore import pyqtSlot


class Example(QWidget):
    def __init__(self, reader_path, _viewer):
        super().__init__()
        self.reader = scanreader.read_scan(reader_path, join_contiguous=True)
        self._left_trim = 0
        self._right_trim = 0
        if _viewer.layers is not None:
            _viewer.add_image(self.reader[:, :, :, 0, 2])
        self.initUI()

    @property
    def left_trim(self):
        return self._left_trim

    @left_trim.setter
    def left_trim(self, value):
        self._left_trim = value

    @property
    def right_trim(self):
        return self._right_trim

    @right_trim.setter
    def right_trim(self, value):
        self._right_trim = value

    def trim(self):
        new_slice_x = [slice(s.start + self.left_trim, s.stop - self.right_trim) for s in self.reader.fields[0].output_xslices]
        return [i for s in new_slice_x for i in range(s.start, s.stop)]

    @pyqtSlot(int)
    def on_right_trim_valueChanged(self, value):
        self.right_trim = value
        self.update_image(value)

    # @pyqtSlot(int)
    def on_left_trim_valueChanged(self, value):
        self.left_trim = value
        self.update_image(value)

    def update_image(self):
        trim_x = self.trim()
        arr = self.reader[:, :, trim_x, 0, 2]
        viewer.layers[0].data = arr

    def initUI(self):
        self.lcd = QLCDNumber(self)
        self.t_left_box = MySpinBox()
        self.t_right_box = MySpinBox()
        layout = QGridLayout()
        layout.addWidget(QLabel('Trim ROI Edges'), 0, 0, 0, 0)
        layout.addWidget(QLabel('Left Edge:'), 1, 0)
        layout.addWidget(self.t_left_box, 1, 1)
        layout.addWidget(QLabel('Right Edge:'), 2, 0)
        layout.addWidget(self.t_right_box, 2, 1)

        # vbox = QVBoxLayout()
        # vbox.addWidget(self.lcd)
        # vbox.addWidget(self.t_right_box)
        # vbox.addWidget(self.sld)

        self.setLayout(layout)
        self.t_left_box.valueChanged.connect(self.on_left_trim_valueChanged)
        self.t_right_box.valueChanged.connect(self.on_right_trim_valueChanged)

class MySpinBox(QSpinBox):
    valueHasChanged = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valueChanged.connect(self.valueHasChanged)

    def setValue(self, value, emit=False):
        if not emit:
            self.valueChanged.disconnect(self.valueHasChanged)
        super().setValue(value)
        if not emit:
            self.valueChanged.connect(self.valueHasChanged)


from qtpy.QtWidgets import QVBoxLayout, QWidget
from magicgui.widgets import FileEdit, Label, CheckBox, PushButton

class FileWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.name = 'Raw File Widget'
        self._viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.Label1 = Label(value='select raw file for reader')
        self.layout().addWidget(self.Label1.native)
        self.target_file = FileEdit(mode='r')
        self.reader=scanreader.read_scan(str(self.target_file), join_contiguous=True)
        self.layout().addWidget(self.target_file.native)
        self.checkbox = CheckBox(text='split channels')
        self.layout().addWidget(self.checkbox.native)
        self.load_button = PushButton(text='Load')
        self.load_button.clicked.connect(self.load_file)
        self.layout().addWidget(self.load_button.native)
        self.layout().addWidget(self.save_button.native)

    def load_file(self):
        print('start')
        store = tifffile.imread(self.target_file.value, aszarr=True)
        cache = zarr.LRUStoreCache(store, max_size=2 ** 30)
        zobj = zarr.open(cache, mode='r')
        print(zobj.attrs.keys())

        d = da.from_zarr(zobj)
        c = np.argmin(d.shape)
        if len(d.shape) == 2:
            d_list = [d, d[::2, ::2], d[::4, ::4], d[::8, ::8], d[::16, ::16]]
            self._viewer.add_image(d_list, blending='additive', contrast_limits=[0, 255])
        elif (len(d.shape) > 2) & self.checkbox.value:
            d = np.moveaxis(d, c, -1)
            d_list = [d, d[::2, ::2, :], d[::4, ::4, :], d[::8, ::8, :], d[::16, ::16, :]]
            self._viewer.add_image(d_list, rgb=False, contrast_limits=[0, 255], channel_axis=c,
                                   name=[f'ch{x}' for x in range(d.shape[-1])],
                                   blending='additive',)
        elif (len(d.shape) > 2) & (self.checkbox.value is False):
            d = np.moveaxis(d, c, -1)
            d_list = [d, d[::2, ::2, :], d[::4, ::4, :], d[::8, ::8, :], d[::16, ::16, :]]
            if d.shape[-1] == 3:
                self._viewer.add_image(d_list, rgb=True, contrast_limits=[0, 255])
            else:
                self._viewer.add_image(d_list, rgb=False, contrast_limits=[0, 255], channel_axis=-1,
                                       name=[f'ch{x}' for x in range(d.shape[-1])],
                                       blending='additive')


parent = Path('/home/rbo/caiman_data')
raw_tiff_name = parent / 'high_res.tif'
reader = scanreader.read_scan(str(raw_tiff_name), join_contiguous=True)

viewer = napari.Viewer()
viewer.add_image(reader[:, :, :, 0, 2])

# widget = Example(str(raw_tiff_name),viewer)
viewer.window.add_dock_widget(FileWidget, area='right')
napari.run()
# x=2
