# test_example = false

from PyQt6 import QtWidgets, QtCore
import fastplotlib as fpl


def qt_fpl(movie, text=""):

    def update_frame(ix):
        iw.figure[0, 0]["image_widget_managed"].data = movie[ix]

    iw = fpl.ImageWidget(movie, figure_kwargs={"size": (700, 560)})
    iw.figure[0, 0].add_text(text, name="text")

    # create a QMainWindow
    main_window = QtWidgets.QMainWindow()

    # Create a QSlider for updating frames
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setMaximum(movie.shape[0] - 1)
    slider.setMinimum(0)
    slider.valueChanged.connect(update_frame)

    # put slider in a dock
    dock = QtWidgets.QDockWidget()
    dock.setWidget(slider)

    # put the dock in the main window
    main_window.addDockWidget(
        QtCore.Qt.DockWidgetArea.BottomDockWidgetArea,
        dock
    )

    # calling fig.show() is required to start the rendering loop
    qwidget = iw.show()

    main_window.setCentralWidget(qwidget)

    main_window.resize(movie.shape[2], movie.shape[1])

    main_window.show()
    fpl.loop.run()


def run_gui(movie, text=""):
    """
    Run the GUI for the fastplotlib image widget.

    Parameters
    ----------
    movie : np.ndarray
        The movie to display.
    name : str
        The name of the image widget.
    text : str
        The text to display.

    Returns
    -------

    """
    qt_fpl(movie, text=text)
