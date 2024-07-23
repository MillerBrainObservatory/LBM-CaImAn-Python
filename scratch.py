import sys
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QLCDNumber, QVBoxLayout, QApplication, QSpinBox
from pathlib import Path
from skimage.io import imread
import napari


def trim(scan, amounts_x):
    new_slice_x = [slice(s.start + amounts_x[0], s.stop - amounts_x[1]) for s in scan.fields[0].output_xslices]
    return [i for s in new_slice_x for i in range(s.start, s.stop)]


class Example(QWidget):
    def __init__(self, reader, viewer):
        super().__init__()
        self.reader = reader
        self.viewer = viewer
        self.initUI()

    @pyqtSlot(int)
    def on_sld_valueChanged(self, value):
        self.lcd.display(value)
        self.update_image(value, value)  # Update image based on slider value

    def update_image(self, tleft=0, tright=0):
        print('Calling update image')
        trim_x = range(tleft, self.reader.shape[2] - tright)
        arr = self.reader[:, :, trim_x, 0, 2]
        self.viewer.layers[0].data = arr
        self.viewer.update()

    def initUI(self):
        self.lcd = QLCDNumber(self)
        self.sld = MySpinBox()

        vbox = QVBoxLayout()
        vbox.addWidget(self.lcd)
        vbox.addWidget(self.sld)

        self.setLayout(vbox)
        self.sld.valueChanged.connect(self.on_sld_valueChanged)

        self.setWindowTitle('Signal & slot')


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


parent = Path('/home/rbo/caiman_data')
raw_tiff_name = parent / 'high_res.tif'

reader = imread(raw_tiff_name)
viewer = napari.Viewer()
viewer.add_image(reader)

app = QApplication(sys.argv)
widget = Example(reader, viewer)
viewer.window.add_dock_widget(widget, area='right')
napari.run()
app.exec_()
