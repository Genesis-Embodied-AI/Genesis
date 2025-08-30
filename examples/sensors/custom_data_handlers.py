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

IS_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt

    IS_MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


class PyQtGraphPlotter(DataHandler):
    """
    Real-time plot using PyQtGraph for live sensor data visualization.

    Parameters
    ----------
    labels : list[str] | dict[str, list[str]] | None
        Labels for data channels. Can be:
        - list[str]: Labels for array-like data channels
        - dict[str, list[str]]: Dict keys become subplot titles, values are channel labels within each subplot
        - None: Auto-detect from first data input (supports both array and dict data)
    vis_window_size : tuple[int, int], optional
        The size of the plot window. Defaults to (800, 600).
    rolling_window_size : int, optional
        Number of data points to display in the rolling window. Defaults to 100.
    title : str, optional
        Plot window title. Defaults to "Live Sensor Data".
    """

    def __init__(
        self,
        labels: list[str] | dict[str, list[str]] | None = None,
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

        # Will be set during first data processing or initialization
        self.is_dict_data = None
        self.subplot_structure = None

        self.x_data = []
        self.y_data = {}  # Will store data organized by subplot and channel

        self.app = None
        self.widget = None
        self.plot_widgets = []  # Multiple plot widgets for subplots
        self.curves = {}  # Organized as {subplot_key: [curves_in_subplot]}

    def initialize(self):
        """Initialize PyQtGraph application and plot window."""
        if not pg.QtWidgets.QApplication.instance():
            self.app = pg.QtWidgets.QApplication([])
        else:
            self.app = pg.QtWidgets.QApplication.instance()

        self.widget = pg.GraphicsLayoutWidget(show=True, title=self.title)
        self.widget.resize(*self.vis_window_size)

        # If labels are provided, set up structure now
        if self.labels is not None:
            self._setup_plot_structure(self.labels)

        gs.logger.info("PyQtGraphPlotter: created PyQtGraph window")

    def _setup_plot_structure(self, labels_or_data):
        """Set up the plot structure based on labels or first data sample."""
        if isinstance(labels_or_data, dict):
            # Dictionary data - create subplots
            self.is_dict_data = True
            if isinstance(list(labels_or_data.values())[0], (list, tuple)):
                # labels_or_data is {subplot_name: [channel_labels]}
                self.subplot_structure = labels_or_data
            else:
                # labels_or_data is actual data {key: array_like_data}
                # Need to determine number of channels from array size
                self.subplot_structure = {}
                for key, values in labels_or_data.items():
                    values = np.atleast_1d(tensor_to_array(values))
                    n_channels = len(values)
                    if n_channels == 1:
                        self.subplot_structure[key] = [key]
                    elif n_channels == 3:
                        # Common case for 3D data (IMU, force, etc.)
                        self.subplot_structure[key] = ["x", "y", "z"]
                    else:
                        # Generic indexing for other sizes
                        self.subplot_structure[key] = [f"{key}_{i}" for i in range(n_channels)]
        elif isinstance(labels_or_data, (list, tuple)):
            # Array data - single plot
            self.is_dict_data = False
            self.subplot_structure = {"main": labels_or_data}
        else:
            gs.raise_exception("PyQtGraphPlotter: Invalid labels or data format")

        # Clear existing plots and create new structure
        self.plot_widgets.clear()
        self.curves.clear()
        self.y_data.clear()

        colors = ["r", "g", "b", "c", "m", "y", "w"]

        # Create plots for each subplot
        for subplot_idx, (subplot_key, channel_labels) in enumerate(self.subplot_structure.items()):
            # Add new row if not the first plot
            if subplot_idx > 0:
                self.widget.nextRow()

            plot_widget = self.widget.addPlot(title=subplot_key if self.is_dict_data else self.title)
            plot_widget.setLabel("left", "Value")
            plot_widget.setLabel("bottom", "Time")
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            if len(channel_labels) > 1:
                plot_widget.addLegend()

            self.plot_widgets.append(plot_widget)

            # Create curves for this subplot - reset color for each subplot
            subplot_curves = []
            self.y_data[subplot_key] = {}

            for color_idx, channel_label in enumerate(channel_labels):
                color = colors[color_idx % len(colors)]
                curve = plot_widget.plot(pen=pg.mkPen(color=color, width=2), name=channel_label)
                subplot_curves.append(curve)
                self.y_data[subplot_key][channel_label] = []

            self.curves[subplot_key] = subplot_curves

        # Initialize time data
        self.x_data.clear()

    def process(self, data, cur_time):
        """Process new data point and update plot."""
        # Auto-detect data structure on first call if not already set up
        if self.subplot_structure is None:
            self._setup_plot_structure(data)

        # Convert data to appropriate format
        if isinstance(data, dict):
            # Dictionary input - expect dict format
            if not self.is_dict_data:
                gs.raise_exception("PyQtGraphPlotter: Expected array data but received dictionary")
            processed_data = {}
            for key, values in data.items():
                if key not in self.subplot_structure:
                    continue  # Skip keys not in our structure
                values = np.atleast_1d(tensor_to_array(values))
                processed_data[key] = values
        else:
            # Array input
            data = np.atleast_1d(tensor_to_array(data))
            if self.is_dict_data:
                gs.raise_exception("PyQtGraphPlotter: Expected dictionary data but received array")
            processed_data = {"main": data}

        # Update time data
        self.x_data.append(cur_time)

        # Update y data for each subplot
        for subplot_key, subplot_data in processed_data.items():
            if subplot_key not in self.y_data:
                gs.logger.debug(f"PyQtGraphPlotter: Skipping unknown subplot key '{subplot_key}'")
                continue

            channel_labels = list(self.subplot_structure[subplot_key])
            if len(subplot_data) != len(channel_labels):
                gs.logger.warning(
                    f"PyQtGraphPlotter: Data length ({len(subplot_data)}) doesn't match "
                    f"number of channels ({len(channel_labels)}) for subplot '{subplot_key}'"
                )
                continue

            for i, channel_label in enumerate(channel_labels):
                if i < len(subplot_data):
                    self.y_data[subplot_key][channel_label].append(float(subplot_data[i]))

        # Safety check: ensure all channels have same length as x_data
        for subplot_key in self.y_data:
            for channel_label in self.y_data[subplot_key]:
                while len(self.y_data[subplot_key][channel_label]) < len(self.x_data):
                    # Pad with last value or 0 if empty
                    last_val = (
                        self.y_data[subplot_key][channel_label][-1] if self.y_data[subplot_key][channel_label] else 0.0
                    )
                    self.y_data[subplot_key][channel_label].append(last_val)

        # Maintain rolling window
        if len(self.x_data) > self.rolling_window_size:
            self.x_data.pop(0)
            for subplot_key in self.y_data:
                for channel_label in self.y_data[subplot_key]:
                    if len(self.y_data[subplot_key][channel_label]) > 0:
                        self.y_data[subplot_key][channel_label].pop(0)

        # Update all curves
        for subplot_key, curves in self.curves.items():
            channel_labels = list(self.subplot_structure[subplot_key])
            for curve, channel_label in zip(curves, channel_labels):
                y_data = self.y_data[subplot_key][channel_label]
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
                self.plot_widgets.clear()
                self.curves.clear()


class MPLPlotter(DataHandler):
    """
    Real-time plot using MatPlotLib for live sensor data visualization.

    Parameters
    ----------
    labels : list[str] | dict[str, list[str]] | None
        Labels for data channels. Can be:
        - list[str]: Labels for array-like data channels
        - dict[str, list[str]]: Dict keys become subplot titles, values are channel labels within each subplot
        - None: Auto-detect from first data input (supports both array and dict data)
    vis_window_size : tuple[int, int], optional
        The size of the plot window in inches. Defaults to (10, 6).
    rolling_window_size : int, optional
        Number of data points to display in the rolling window. Defaults to 100.
    title : str, optional
        Plot window title. Defaults to "Live Sensor Data".
    """

    def __init__(
        self,
        labels: list[str] | dict[str, list[str]] | None = None,
        vis_window_size: tuple[int, int] = (10, 6),
        rolling_window_size: int = 100,
        title: str = "Live Sensor Data",
    ):
        if not IS_MATPLOTLIB_AVAILABLE:
            gs.raise_exception(
                "MPLPlotter: matplotlib is not installed. Please install it with `pip install matplotlib`."
            )

        self.labels = labels
        self.vis_window_size = vis_window_size
        self.rolling_window_size = rolling_window_size
        self.title = title

        # Will be set during first data processing or initialization
        self.is_dict_data = None
        self.subplot_structure = None

        # Data storage
        self.x_data = []
        self.y_data = {}  # Will store data organized by subplot and channel

        # Matplotlib objects
        self.fig = None
        self.axes = []  # Multiple axes for subplots
        self.lines = {}  # Organized as {subplot_key: [lines_in_subplot]}
        self.backgrounds = []  # Backgrounds for blitting

        # Colors for different data series
        self.colors = ["red", "green", "blue", "cyan", "magenta", "yellow", "black", "orange"]

    def initialize(self):
        """Initialize Matplotlib figure and setup blitting."""
        plt.ion()  # Turn on interactive mode

        # If labels are provided, set up structure now
        if self.labels is not None:
            self._setup_plot_structure(self.labels)

        gs.logger.info("MPLPlotter: created Matplotlib window")

    def _setup_plot_structure(self, labels_or_data):
        """Set up the plot structure based on labels or first data sample."""
        if isinstance(labels_or_data, dict):
            # Dictionary data - create subplots
            self.is_dict_data = True
            if isinstance(list(labels_or_data.values())[0], (list, tuple)):
                # labels_or_data is {subplot_name: [channel_labels]}
                self.subplot_structure = labels_or_data
            else:
                # labels_or_data is actual data {key: array_like_data}
                # Need to determine number of channels from array size
                self.subplot_structure = {}
                for key, values in labels_or_data.items():
                    values = np.atleast_1d(tensor_to_array(values))
                    n_channels = len(values)
                    if n_channels == 1:
                        self.subplot_structure[key] = [key]
                    elif n_channels == 3:
                        # Common case for 3D data (IMU, force, etc.)
                        self.subplot_structure[key] = ["x", "y", "z"]
                    else:
                        # Generic indexing for other sizes
                        self.subplot_structure[key] = [f"{key}_{i}" for i in range(n_channels)]
        elif isinstance(labels_or_data, (list, tuple)):
            # Array data - single plot
            self.is_dict_data = False
            self.subplot_structure = {"main": labels_or_data}
        else:
            gs.raise_exception("MPLPlotter: Invalid labels or data format")

        # Create figure and subplots
        n_subplots = len(self.subplot_structure)

        if n_subplots == 1:
            self.fig, ax = plt.subplots(figsize=self.vis_window_size)
            self.axes = [ax]
        else:
            self.fig, axes = plt.subplots(
                n_subplots, 1, figsize=self.vis_window_size, sharex=True, constrained_layout=True
            )
            self.axes = axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]

        self.fig.suptitle(self.title)

        # Clear existing structure
        self.lines.clear()
        self.y_data.clear()
        self.backgrounds.clear()

        # Create lines for each subplot
        for subplot_idx, (subplot_key, channel_labels) in enumerate(self.subplot_structure.items()):
            ax = self.axes[subplot_idx]
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

            # Set subplot title if we have multiple subplots
            if self.is_dict_data and n_subplots > 1:
                ax.set_title(subplot_key)

            # Create lines for this subplot - reset color for each subplot
            subplot_lines = []
            self.y_data[subplot_key] = {}

            for color_idx, channel_label in enumerate(channel_labels):
                color = self.colors[color_idx % len(self.colors)]
                (line,) = ax.plot([], [], color=color, label=channel_label, linewidth=2)
                subplot_lines.append(line)
                self.y_data[subplot_key][channel_label] = []

            self.lines[subplot_key] = subplot_lines

            if len(channel_labels) > 1:
                ax.legend(loc="upper right")

            # Set initial limits
            ax.set_xlim(0, 10)
            ax.set_ylim(-1, 1)

        # Show the plot and capture backgrounds for blitting
        self.fig.show()
        self.fig.canvas.draw()

        for ax in self.axes:
            self.backgrounds.append(self.fig.canvas.copy_from_bbox(ax.bbox))

        # Initialize time data
        self.x_data.clear()

    def process(self, data, cur_time):
        """Process new data point and update plot using blitting."""
        # Auto-detect data structure on first call if not already set up
        if self.subplot_structure is None:
            self._setup_plot_structure(data)

        # Convert data to appropriate format
        if isinstance(data, dict):
            # Dictionary input - expect dict format
            if not self.is_dict_data:
                gs.raise_exception("MPLPlotter: Expected array data but received dictionary")
            processed_data = {}
            for key, values in data.items():
                if key not in self.subplot_structure:
                    continue  # Skip keys not in our structure
                values = np.atleast_1d(tensor_to_array(values))
                processed_data[key] = values
        else:
            # Array input
            data = np.atleast_1d(tensor_to_array(data))
            if self.is_dict_data:
                gs.raise_exception("MPLPlotter: Expected dictionary data but received array")
            processed_data = {"main": data}

        # Update time data
        self.x_data.append(cur_time)

        # Update y data for each subplot
        for subplot_key, subplot_data in processed_data.items():
            if subplot_key not in self.y_data:
                gs.logger.debug(f"MPLPlotter: Skipping unknown subplot key '{subplot_key}'")
                continue

            channel_labels = list(self.subplot_structure[subplot_key])
            if len(subplot_data) != len(channel_labels):
                gs.logger.warning(
                    f"MPLPlotter: Data length ({len(subplot_data)}) doesn't match "
                    f"number of channels ({len(channel_labels)}) for subplot '{subplot_key}'"
                )
                continue

            for i, channel_label in enumerate(channel_labels):
                if i < len(subplot_data):
                    self.y_data[subplot_key][channel_label].append(float(subplot_data[i]))

        # Safety check: ensure all channels have same length as x_data
        for subplot_key in self.y_data:
            for channel_label in self.y_data[subplot_key]:
                while len(self.y_data[subplot_key][channel_label]) < len(self.x_data):
                    # Pad with last value or 0 if empty
                    last_val = (
                        self.y_data[subplot_key][channel_label][-1] if self.y_data[subplot_key][channel_label] else 0.0
                    )
                    self.y_data[subplot_key][channel_label].append(last_val)

        # Maintain rolling window
        if len(self.x_data) > self.rolling_window_size:
            self.x_data.pop(0)
            for subplot_key in self.y_data:
                for channel_label in self.y_data[subplot_key]:
                    if len(self.y_data[subplot_key][channel_label]) > 0:
                        self.y_data[subplot_key][channel_label].pop(0)

        # Update each subplot
        for subplot_idx, (subplot_key, subplot_lines) in enumerate(self.lines.items()):
            ax = self.axes[subplot_idx]

            # Check if axis limits need updating for this subplot
            limits_changed = False
            if self.x_data:
                x_min, x_max = min(self.x_data), max(self.x_data)
                x_range = x_max - x_min
                if x_range == 0:
                    x_range = 1
                new_x_limits = (x_min - x_range * 0.05, x_max + x_range * 0.05)
                if new_x_limits != ax.get_xlim():
                    ax.set_xlim(new_x_limits)
                    limits_changed = True

                # Update y limits based on all data in this subplot
                all_y_values = []
                for channel_label in self.y_data[subplot_key]:
                    all_y_values.extend(self.y_data[subplot_key][channel_label])

                if all_y_values:
                    y_min, y_max = min(all_y_values), max(all_y_values)
                    y_range = y_max - y_min
                    if y_range == 0:
                        y_range = 1
                    new_y_limits = (y_min - y_range * 0.1, y_max + y_range * 0.1)
                    if new_y_limits != ax.get_ylim():
                        ax.set_ylim(new_y_limits)
                        limits_changed = True

            # If limits changed, redraw background for this subplot
            if limits_changed:
                self.fig.canvas.draw()
                self.backgrounds[subplot_idx] = self.fig.canvas.copy_from_bbox(ax.bbox)

            # Restore background and update line data for this subplot
            self.fig.canvas.restore_region(self.backgrounds[subplot_idx])

            channel_labels = list(self.subplot_structure[subplot_key])
            for line, channel_label in zip(subplot_lines, channel_labels):
                y_data = self.y_data[subplot_key][channel_label]
                line.set_data(self.x_data, y_data)
                ax.draw_artist(line)

            # Blit the updated subplot
            self.fig.canvas.blit(ax.bbox)

        self.fig.canvas.flush_events()

    def cleanup(self):
        """Clean up Matplotlib resources."""
        if self.fig:
            try:
                plt.close(self.fig)
                gs.logger.debug("MPLPlotter: closed Matplotlib window")
            except Exception as e:
                gs.logger.warning(f"MPLPlotter: Error closing window: {e}")
            finally:
                self.fig = None
                self.axes.clear()
                self.lines.clear()
                self.backgrounds.clear()
