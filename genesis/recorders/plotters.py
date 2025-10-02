import io
import itertools
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, T

import numpy as np
import torch
from PIL import Image

import genesis as gs
from genesis.utils import has_display, tensor_to_array

from .base_recorder import Recorder, RecorderOptions
from .recorder_manager import RecorderManager, register_recording

IS_PYQTGRAPH_AVAILABLE = False
try:
    import pyqtgraph as pg

    IS_PYQTGRAPH_AVAILABLE = True
except ImportError:
    pass

IS_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib as mpl

    IS_MATPLOTLIB_AVAILABLE = tuple(map(int, mpl.__version__.split("."))) >= (3, 7, 0)
except ImportError:
    pass


COLORS = itertools.cycle(("r", "g", "b", "c", "m", "y"))


def _data_to_array(data: Sequence) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = tensor_to_array(data)
    return np.atleast_1d(data)


class BasePlotterOptions(RecorderOptions):
    """
    Base class for plot visualization.

    Parameters
    ----------
    title: str
        The title of the plot.
    window_size: tuple[int, int]
        The size of the window in pixels.
    save_to_filename: str | None
        If provided, the animation will be saved to a file with the given filename.
    show_window: bool | None
        Whether to show the window. If not provided, it will be set to True if a display is connected, False otherwise.
    """

    title: str = ""
    window_size: tuple[int, int] = (800, 600)
    save_to_filename: str | None = None
    show_window: bool | None = None


class BasePlotter(Recorder):

    def __init__(self, manager: "RecorderManager", options: RecorderOptions, data_func: Callable[[], T]):
        super().__init__(manager, options, data_func)
        self._frames_buffer: list[np.ndarray] = []

    def build(self):
        super().build()
        self.show_window = self._options.show_window if self._options.show_window is not None else has_display()

        self.video_writer = None
        if self._options.save_to_filename:

            def _get_video_frame_buffer(plotter):
                # Make sure that all the data in the pipe has been processed before rendering anything
                if not plotter._frames_buffer:
                    if plotter._data_queue is not None and not plotter._data_queue.empty():
                        while not plotter._frames_buffer:
                            time.sleep(0.1)

                return plotter._frames_buffer.pop(0)

            self.video_writer = self._manager.add_recorder(
                data_func=partial(_get_video_frame_buffer, self),
                rec_options=gs.recorders.VideoFile(
                    filename=self._options.save_to_filename,
                    hz=self._options.hz,
                ),
            )

    def process(self, data, cur_time):
        # Update plot
        self._update_plot()

        # Render frame if necessary
        if self._options.save_to_filename:
            self._frames_buffer.append(self.get_image_array())

    def cleanup(self):
        if self.video_writer is not None:
            self.video_writer.stop()
            self._frames_buffer.clear()
            self.video_writer = None

    def _update_plot(self):
        """
        Update plot.
        """
        raise NotImplementedError(f"[{type(self).__name__}] _update_plot() is not implemented.")

    def get_image_array(self):
        """
        Capture the plot image as a video frame.

        Returns
        -------
        image_array : np.ndarray
            The RGB image as a numpy array.
        """
        raise NotImplementedError(f"[{type(self).__name__}] get_image_array() is not implemented.")


@dataclass
class LinePlotterMixinOptions:
    """
    Mixin class for live line plot visualization of scalar data.

    The recorded data_func should return scalar data (single scalar, a tuple of scalars, or a dict with string keys and
    scalar or tuple of scalars as values).

    Parameters
    ----------
    labels: tuple[str] | dict[str, tuple[str]] | None
        The labels for the plot. The length of the labels should match the length of the data.
        If a dict is provided, the data should also be a dict of tuples of strings that match the length of the data.
        The keys will be used as subplot titles and the values will be used as labels within each subplot.
    x_label: str, optional
        Label for the horizontal axis.
    y_label: str, optional
        Label for the vertical axis.
    history_length: int
        The maximum number of previous data to store.
    """

    labels: tuple[str, ...] | dict[str, tuple[str, ...]] | None = None
    x_label: str = ""
    y_label: str = ""
    history_length: int = 100


class LinePlotHelper:
    """
    Helper class that manages line plot data.

    Use composition pattern.
    """

    def __init__(self, options: LinePlotterMixinOptions, data: dict[str, Sequence] | Sequence):
        self._x_data: list[float] = []
        self._y_data: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._history_length = options.history_length

        # Note that these attributes will be set during first data processing or initialization
        self._is_dict_data: bool | None = None
        self._subplot_structure: dict[str, tuple[str, ...]] = {}

        if isinstance(data, dict):
            self._is_dict_data = True

            if options.labels is not None:
                assert isinstance(
                    options.labels, dict
                ), f"[{type(self).__name__}] Labels must be a dict when data is a dict"
                assert set(options.labels.keys()) == set(
                    data.keys()
                ), f"[{type(self).__name__}] Label keys must match data keys"

                for key in data.keys():
                    data_values = _data_to_array(data[key])
                    label_values = options.labels[key]
                    assert len(label_values) == len(
                        data_values
                    ), f"[{type(self).__name__}] Label count must match data count for key '{key}'"
                    self._subplot_structure[key] = tuple(label_values)
            else:
                self._subplot_structure = {}
                for key, values in data.items():
                    values = _data_to_array(values)
                    self._subplot_structure[key] = tuple(f"{key}_{i}" for i in range(len(values)))
        else:
            self._is_dict_data = False
            data = _data_to_array(data)

            if options.labels is not None:
                if not isinstance(options.labels, Sequence):
                    options.labels = (options.labels,)
                assert len(options.labels) == len(data), f"[{type(self).__name__}] Label count must match data count"
                plot_labels = tuple(options.labels)
            else:
                plot_labels = tuple(f"data_{i}" for i in range(len(data)))

            self._subplot_structure = {"main": plot_labels}

    def clear_data(self):
        self._x_data.clear()
        self._y_data.clear()

    def process(self, data, cur_time):
        """Process new data point and update plot."""

        if self._is_dict_data:
            processed_data = {}
            for key, values in data.items():
                if key not in self._subplot_structure:
                    continue  # skip keys not included in subplot structure
                values = _data_to_array(values)
                processed_data[key] = values
        else:
            data = _data_to_array(data)
            processed_data = {"main": data}

        # Update time data
        self._x_data.append(cur_time)

        # Update y data for each subplot
        for subplot_key, subplot_data in processed_data.items():
            channel_labels = self._subplot_structure[subplot_key]
            if len(subplot_data) != len(channel_labels):
                gs.logger.warning(
                    f"[{type(self).__name__}] Data length ({len(subplot_data)}) doesn't match "
                    f"expected number of channels ({len(channel_labels)}) for subplot '{subplot_key}', skipping..."
                )
                continue

            for i, channel_label in enumerate(channel_labels):
                if i < len(subplot_data):
                    self._y_data[subplot_key][channel_label].append(float(subplot_data[i]))

        # Maintain rolling history window
        if len(self._x_data) > self._history_length:
            self._x_data.pop(0)
            for subplot_key in self._y_data:
                for channel_label in self._y_data[subplot_key]:
                    try:
                        self._y_data[subplot_key][channel_label].pop(0)
                    except IndexError:
                        break  # empty, nothing to do.

    @property
    def x_data(self):
        return self._x_data

    @property
    def y_data(self):
        return self._y_data

    @property
    def is_dict_data(self):
        return self._is_dict_data

    @property
    def subplot_structure(self):
        return self._subplot_structure


class BasePyQtPlotter(BasePlotter):
    """
    Base class for PyQt based plotters.
    """

    def build(self):
        if not IS_PYQTGRAPH_AVAILABLE:
            gs.raise_exception(
                f"{type(self).__name__} pyqtgraph is not installed. Please install it with `pip install pyqtgraph`."
            )

        super().build()

        self.app: pg.QtWidgets.QApplication | None = None
        self.widget: pg.GraphicsLayoutWidget | None = None
        self.plot_widgets: list[pg.PlotWidget] = []
        if not pg.QtWidgets.QApplication.instance():
            self.app = pg.QtWidgets.QApplication([])
        else:
            self.app = pg.QtWidgets.QApplication.instance()

        self.widget = pg.GraphicsLayoutWidget(show=self.show_window, title=self._options.title)
        if self.show_window:
            gs.logger.info(f"[{type(self).__name__}] created PyQtGraph window")
        self.widget.resize(*self._options.window_size)

    def cleanup(self):
        super().cleanup()

        if self.widget:
            try:
                self.widget.close()
                gs.logger.debug(f"[{type(self).__name__}] closed PyQtGraph window")
            except Exception as e:
                gs.logger.warning(f"[{type(self).__name__}] Error closing window: {e}")
            finally:
                self.widget = None
                self.plot_widgets.clear()

    @property
    def run_in_thread(self) -> bool:
        return True

    def get_image_array(self):
        """
        Capture the plot image as a video frame.

        Returns
        -------
        image_array : np.ndarray
            The image as a numpy array in (b,g,r,a) format.
        """
        pixmap = self.widget.grab()
        qimage = pixmap.toImage()

        # pyqtgraph provides imageToArray but it always outputs (b,g,r,a) format
        # https://pyqtgraph.readthedocs.io/en/latest/api_reference/functions.html#pyqtgraph.functions.imageToArray
        return pg.imageToArray(qimage, copy=True, transpose=True)


class PyQtLinePlotterOptions(BasePlotterOptions, LinePlotterMixinOptions):
    """
    Live line plot visualization of data using PyQtGraph.

    The recorded data_func should return scalar data (single scalar, a tuple of scalars, or a dict with string keys and
    scalar or tuple of scalars as values).

    Parameters
    ----------
    title: str
        The title of the plot.
    window_size: tuple[int, int]
        The size of the window in pixels.
    save_to_filename: str | None
        If provided, the animation will be saved to a file with the given filename.
    show_window: bool | None
        Whether to show the window. If not provided, it will be set to True if a display is connected, False otherwise.
    labels: tuple[str] | dict[str, tuple[str]] | None
        The labels for the plot. The length of the labels should match the length of the data.
        If a dict is provided, the data should also be a dict of tuples of strings that match the length of the data.
        The keys will be used as subplot titles and the values will be used as labels within each subplot.
    x_label: str, optional
        Label for the horizontal axis.
    y_label: str, optional
        Label for the vertical axis.
    history_length: int
        The maximum number of previous data to store.
    """

    pass


@register_recording(PyQtLinePlotterOptions)
class PyQtLinePlotter(BasePyQtPlotter):

    def build(self):
        super().build()

        self.line_plot = LinePlotHelper(options=self._options, data=self._data_func())
        self.curves: dict[str, list[pg.PlotCurveItem]] = {}

        # create plots for each subplot
        for subplot_idx, (subplot_key, channel_labels) in enumerate(self.line_plot.subplot_structure.items()):
            # add new row if not the first plot
            if subplot_idx > 0:
                self.widget.nextRow()

            plot_widget = self.widget.addPlot(title=subplot_key if self.line_plot.is_dict_data else self._options.title)
            plot_widget.setLabel("bottom", self._options.x_label)
            plot_widget.setLabel("left", self._options.y_label)
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_widget.addLegend()

            self.plot_widgets.append(plot_widget)

            # create lines for this subplot
            subplot_curves = []

            for color, channel_label in zip(COLORS, channel_labels):
                curve = plot_widget.plot(pen=pg.mkPen(color=color, width=2), name=channel_label)
                subplot_curves.append(curve)

            self.curves[subplot_key] = subplot_curves

    def process(self, data, cur_time):
        self.line_plot.process(data, cur_time)
        super().process(data, cur_time)

    def _update_plot(self):
        # update all curves
        for subplot_key, curves in self.curves.items():
            channel_labels = self.line_plot.subplot_structure[subplot_key]
            for curve, channel_label in zip(curves, channel_labels):
                curve.setData(x=self.line_plot.x_data, y=self.line_plot.y_data[subplot_key][channel_label])

        if self.app:
            self.app.processEvents()

    def cleanup(self):
        super().cleanup()
        self.line_plot.clear_data()
        self.curves.clear()


class BaseMPLPlotter(BasePlotter):
    """
    Base class for matplotlib based plotters.
    """

    def build(self):
        if not IS_MATPLOTLIB_AVAILABLE:
            gs.raise_exception(
                f"{type(self).__name__} matplotlib is not installed. Please install it with `pip install matplotlib>=3.7.0`."
            )

        super().build()

        import matplotlib.pyplot as plt

        self.fig: plt.Figure | None = None
        self._lock = threading.Lock()

        # matplotlib figsize uses inches
        dpi = mpl.rcParams.get("figure.dpi", 100)
        self.figsize = (self._options.window_size[0] / dpi, self._options.window_size[1] / dpi)

    def _show_fig(self):
        if self.show_window:
            self.fig.show()
            gs.logger.info(f"[{type(self).__name__}] created matplotlib window")

    def cleanup(self):
        """Clean up matplotlib resources."""
        super().cleanup()

        # Logger may not be available anymore
        logger_exists = hasattr(gs, "logger")

        if self.fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self.fig)
                if logger_exists:
                    gs.logger.debug(f"[{type(self).__name__}] Closed matplotlib window")
            except Exception as e:
                if logger_exists:
                    gs.logger.warning(f"[{type(self).__name__}] Error closing window: {e}")
            finally:
                self.fig = None

    def get_image_array(self):
        """
        Capture the plot image as a video frame.

        Returns
        -------
        image_array : np.ndarray
            The RGB image as a numpy array.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        self._lock.acquire()
        if isinstance(self.fig.canvas, FigureCanvasAgg):
            # Read internal buffer
            width, height = self.fig.canvas.get_width_height(physical=True)
            rgba_array_flat = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            rgb_array = rgba_array_flat.reshape((height, width, 4))[..., :3]

            # Rescale image if necessary
            if (width, height) != tuple(self._options.window_size):
                img = Image.fromarray(rgb_array)
                img = img.resize(self._options.window_size, resample=Image.BILINEAR)
                rgb_array = np.asarray(img)
            else:
                rgb_array = rgb_array.copy()
        else:
            # Slower but more generic fallback only if necessary
            buffer = io.BytesIO()
            self.fig.canvas.print_figure(buffer, format="png", dpi="figure")
            buffer.seek(0)
            img = Image.open(buffer)
            rgb_array = np.asarray(img.convert("RGB"))
        self._lock.release()

        return rgb_array

    @property
    def run_in_thread(self) -> bool:
        # matplotlib throws NSInternalInconsistencyException when trying to use threading for visualization on macOS
        return not self.show_window or gs.platform != "macOS"


class MPLLinePlotterOptions(BasePlotterOptions, LinePlotterMixinOptions):
    """
    Live line plot visualization of data using matplotlib.

    The recorded data_func should return scalar data (single scalar, a tuple of scalars, or a dict with string keys and
    scalar or tuple of scalars as values).

    Parameters
    ----------
    title: str
        The title of the plot.
    window_size: tuple[int, int]
        The size of the window in pixels.
    save_to_filename: str | None
        If provided, the animation will be saved to a file with the given filename.
    show_window: bool | None
        Whether to show the window. If not provided, it will be set to True if a display is connected, False otherwise.
    labels: tuple[str] | dict[str, tuple[str]] | None
        The labels for the plot. The length of the labels should match the length of the data.
        If a dict is provided, the data should also be a dict of tuples of strings that match the length of the data.
        The keys will be used as subplot titles and the values will be used as labels within each subplot.
    x_label: str, optional
        Label for the horizontal axis.
    y_label: str, optional
        Label for the vertical axis.
    history_length: int
        The maximum number of previous data to store.
    """

    pass


@register_recording(MPLLinePlotterOptions)
class MPLLinePlotter(BaseMPLPlotter):

    def build(self):
        super().build()

        self.line_plot = LinePlotHelper(options=self._options, data=self._data_func())

        import matplotlib.pyplot as plt

        self.axes: list[plt.Axes] = []
        self.lines: dict[str, list[plt.Line2D]] = {}
        self.backgrounds: list[Any] = []

        # Create figure and subplots
        n_subplots = len(self.line_plot.subplot_structure)
        if n_subplots == 1:
            self.fig, ax = plt.subplots(figsize=self.figsize)
            self.axes = [ax]
        else:
            self.fig, axes = plt.subplots(n_subplots, 1, figsize=self.figsize, sharex=True, constrained_layout=True)
            self.axes = axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
        self.fig.suptitle(self._options.title)

        # Create lines for each subplot
        for subplot_idx, (subplot_key, channel_labels) in enumerate(self.line_plot.subplot_structure.items()):
            ax = self.axes[subplot_idx]
            ax.set_xlabel(self._options.x_label)
            ax.set_ylabel(self._options.y_label)
            ax.grid(True, alpha=0.3)

            if self.line_plot.is_dict_data and n_subplots > 1:
                ax.set_title(subplot_key)

            subplot_lines = []

            for color, channel_label in zip(COLORS, channel_labels):
                (line,) = ax.plot([], [], color=color, label=channel_label, linewidth=2)
                subplot_lines.append(line)

            self.lines[subplot_key] = subplot_lines

        # Legend must be outside, otherwise it will not play well with blitting
        self.fig.legend(ncol=sum(map(len, self.lines.values())), loc="outside lower center")
        self.fig.canvas.draw()

        for ax in self.axes:
            self.backgrounds.append(self.fig.canvas.copy_from_bbox(ax.bbox))

        self._show_fig()

    def process(self, data, cur_time):
        self.line_plot.process(data, cur_time)
        super().process(data, cur_time)

    def _update_plot(self):
        self._lock.acquire()

        # Update each subplot
        for subplot_idx, (subplot_key, subplot_lines) in enumerate(self.lines.items()):
            ax = self.axes[subplot_idx]

            # Check if axis limits need updating for this subplot
            limits_changed = False
            if self.line_plot.x_data:
                x_min, x_max = min(self.line_plot.x_data), max(self.line_plot.x_data)
                x_range = x_max - x_min
                if x_range == 0:
                    x_range = 1
                new_x_limits = (x_min - x_range * 0.05, x_max + x_range * 0.05)
                if new_x_limits != ax.get_xlim():
                    ax.set_xlim(new_x_limits)
                    limits_changed = True

                # Update y limits based on all data in this subplot
                all_y_values = []
                for channel_label in self.line_plot.y_data[subplot_key]:
                    all_y_values.extend(self.line_plot.y_data[subplot_key][channel_label])

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

            # Update lines
            channel_labels = self.line_plot.subplot_structure[subplot_key]
            for line, channel_label in zip(subplot_lines, channel_labels):
                y_data = self.line_plot.y_data[subplot_key][channel_label]
                line.set_data(self.line_plot.x_data, y_data)
                ax.draw_artist(line)

            # Blit the updated subplot
            self.fig.canvas.blit(ax.bbox)

        self.fig.canvas.flush_events()
        self._lock.release()

    def cleanup(self):
        super().cleanup()
        self.line_plot.clear_data()
        self.lines.clear()
        self.backgrounds.clear()


class MPLImagePlotterOptions(BasePlotterOptions):
    """
    Live visualization of image data using matplotlib.

    The image data should be an array-like object with shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4).

    Parameters
    ----------
    title: str
        The title of the plot.
    window_size: tuple[int, int]
        The size of the window in pixels.
    save_to_filename: str | None
        If provided, the animation will be saved to a file with the given filename.
    show_window: bool | None
        Whether to show the window. If not provided, it will be set to True if a display is connected, False otherwise.
    """

    pass


@register_recording(MPLImagePlotterOptions)
class MPLImagePlotter(BaseMPLPlotter):
    """
    Live image viewer using matplotlib.

    The image data should be an array-like object with shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4).
    """

    def build(self):
        super().build()

        import matplotlib.pyplot as plt

        self.image_plot = None
        self.background = None

        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.fig.tight_layout(pad=0)
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.image_plot = self.ax.imshow(np.zeros((1, 1)), cmap="plasma", origin="upper", aspect="auto")
        self._show_fig()

    def process(self, data, cur_time):
        """Process new image data and update display."""
        if isinstance(data, torch.Tensor):
            img_data = tensor_to_array(data)
        else:
            img_data = np.asarray(data)

        vmin, vmax = np.min(img_data), np.max(img_data)

        current_vmin, current_vmax = self.image_plot.get_clim()
        if vmin != current_vmin or vmax != current_vmax:
            self.image_plot.set_clim(vmin, vmax)
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        self.fig.canvas.restore_region(self.background)
        self.image_plot.set_data(img_data)
        self.ax.draw_artist(self.image_plot)
        self.fig.canvas.blit(self.ax.bbox)

        self.fig.canvas.flush_events()

    def cleanup(self):
        super().cleanup()

        self.ax = None
        self.image_plot = None
        self.background = None
