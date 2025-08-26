import sys

import numpy as np

import genesis as gs
from genesis.sensors.data_handlers import DataHandler
from genesis.utils.misc import tensor_to_array

IS_PYQTGRAPH_AVAILABLE = False
try:
    import pyqtgraph as pg

    IS_PYQTGRAPH_AVAILABLE = True
except ImportError:
    pass


class PyQtGraphPlotter(DataHandler):
    """
    Real-time plot using PyQtGraph for live sensor data visualization.

    Creates a live updating plot window showing sensor data as it arrives.
    Uses PyQtGraph for efficient real-time plotting performance.

    Parameters
    ----------
    labels : list[str]
        Labels for each data channel to plot.
    vis_window_size : tuple[int, int], optional
        The size of the plot window. Defaults to (800, 600).
    rolling_window_size : int, optional
        Number of data points to display in the rolling window. Defaults to 100.
    title : str, optional
        Plot window title. Defaults to "Live Sensor Data".
    """

    def __init__(
        self,
        labels: list[str],
        vis_window_size: tuple[int, int] = (800, 600),
        rolling_window_size: int = 100,
        title: str = "Live Sensor Data",
    ):
        if "pyqtgraph" not in sys.modules:
            gs.raise_exception(
                "PyQtGraphPlotter: pyqtgraph is not installed. Please install it with `pip install pyqtgraph`."
            )

        self.labels = labels
        self.vis_window_size = vis_window_size
        self.rolling_window_size = rolling_window_size
        self.title = title

        self.x_data = []
        self.y_data = [[] for _ in labels]

        self.app = None
        self.widget = None
        self.plot_widget = None
        self.curves = []

    def initialize(self):
        """Initialize PyQtGraph application and plot window."""
        if not pg.QtWidgets.QApplication.instance():
            self.app = pg.QtWidgets.QApplication([])
        else:
            self.app = pg.QtWidgets.QApplication.instance()

        self.widget = pg.GraphicsLayoutWidget(show=True, title=self.title)
        self.widget.resize(*self.vis_window_size)

        self.plot_widget = self.widget.addPlot(title=self.title)
        self.plot_widget.setLabel("left", "Value")
        self.plot_widget.setLabel("bottom", "Time Step")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        self.curves = []
        colors = ["r", "g", "b", "c", "m", "y", "w"]  # https://pyqtgraph.readthedocs.io/en/latest/user_guide/style.html
        for i, label in enumerate(self.labels):
            color = colors[i % len(colors)]
            curve = self.plot_widget.plot(pen=pg.mkPen(color=color, width=2), name=label)
            self.curves.append(curve)

        self.x_data.clear()
        for y_list in self.y_data:
            y_list.clear()

        gs.logger.info("PyQtGraphPlotter: created PyQtGraph window")

    def process(self, data, cur_time):
        """Process new data point and update plot."""
        data = np.atleast_1d(tensor_to_array(data))

        if len(data) != len(self.labels):
            gs.raise_exception(
                f"PyQtGraphPlotter: Data length ({len(data)}) doesn't match number of labels ({len(self.labels)})"
            )

        self.x_data.append(cur_time)
        for i, value in enumerate(data):
            self.y_data[i].append(float(value))

        if len(self.x_data) > self.rolling_window_size:
            self.x_data.pop(0)
            for y_list in self.y_data:
                y_list.pop(0)

        for curve, y_data in zip(self.curves, self.y_data):
            curve.setData(x=self.x_data, y=y_data)

        if self.app:
            self.app.processEvents()

    def cleanup(self):
        """Clean up PyQtGraph resources."""
        if self.widget:
            try:
                self.widget.close()
                gs.logger.debug("PyQtGraphPlotter: closed PyQtGraph window")
            except Exception as e:
                gs.logger.warning(f"PyQtGraphPlotter: Error closing window: {e}")
            finally:
                self.widget = None
                self.plot_widget = None
                self.curves.clear()
