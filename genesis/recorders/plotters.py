import io
import itertools
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

import numpy as np
import torch
from PIL import Image

import genesis as gs
from genesis.utils import has_display, tensor_to_array

from .base_recorder import Recorder, RecorderOptions
from .recorder_manager import register_recording

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

    def build(self):
        super().build()
        self.show_window = self._options.show_window if self._options.show_window is not None else has_display()

    def process(self, data, cur_time):
        pass  # allow super() calls

    def cleanup(self):
        pass  # allow super() calls


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


LinePlotterOptionsMixinT = TypeVar("LinePlotterOptionsMixinT", bound=LinePlotterMixinOptions)


class LinePlotterMixin(Generic[LinePlotterOptionsMixinT]):
    """Base class for real-time plotters with shared functionality."""

    def build(self):
        super().build()

        self.x_data: list[float] = []
        self.y_data: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        # Note that these attributes will be set during first data processing or initialization
        self.is_dict_data: bool | None = None
        self.subplot_structure: dict[str, tuple[str, ...]] | None = None

        if self._options.labels is not None:
            self._setup_plot_structure(self._options.labels)
        else:
            self._setup_plot_structure(self._data_func())

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
        self._frames_buffer: list[np.ndarray] = []

    def reset(self, envs_idx=None):
        super().reset(envs_idx)

        # no envs specific resetting supported
        self.x_data.clear()
        self.y_data.clear()

    def cleanup(self):
        """Clean up resources."""
        if self.video_writer is not None:
            self.video_writer.stop()
            self._frames_buffer.clear()
            self.video_writer = None

    def _setup_plot_structure(self, labels_or_data: dict[str, Sequence] | Sequence):
        """Set up the plot structure based on labels or first data sample."""
        if isinstance(labels_or_data, dict):
            self.is_dict_data = True
            next_dict_value = next(iter(labels_or_data.values()))

            if isinstance(next_dict_value, (torch.Tensor, np.ndarray)):
                # data was provided
                self.subplot_structure = {}
                for key, values in labels_or_data.items():
                    values = _data_to_array(values)
                    self.subplot_structure[key] = tuple(f"{key}_{i}" for i in range(len(values)))
            elif isinstance(next_dict_value, Sequence) and isinstance(next_dict_value[0], str):
                # labels were provided
                self.subplot_structure = {k: tuple(v) for k, v in labels_or_data.items()}
            else:
                gs.raise_exception(f"[{type(self).__name__}] Unsupported input argument type: {type(labels_or_data)}")
        else:
            self.is_dict_data = False
            if not isinstance(labels_or_data, Sequence):
                labels_or_data = (labels_or_data,)
            if isinstance(labels_or_data[0], (int, float, np.number)):
                labels_or_data = [f"data_{i}" for i in range(len(labels_or_data))]
            self.subplot_structure = {"main": tuple(labels_or_data)}

    def process(self, data, cur_time):
        """Process new data point and update plot."""

        if self.is_dict_data:
            processed_data = {}
            for key, values in data.items():
                if key not in self.subplot_structure:
                    continue  # skip keys not included in subplot structure
                values = _data_to_array(values)
                processed_data[key] = values
        else:
            data = _data_to_array(data)
            processed_data = {"main": data}

        # Update time data
        self.x_data.append(cur_time)

        # Update y data for each subplot
        for subplot_key, subplot_data in processed_data.items():
            channel_labels = self.subplot_structure[subplot_key]
            if len(subplot_data) != len(channel_labels):
                gs.logger.warning(
                    f"[{type(self).__name__}] Data length ({len(subplot_data)}) doesn't match "
                    f"expected number of channels ({len(channel_labels)}) for subplot '{subplot_key}', skipping..."
                )
                continue

            for i, channel_label in enumerate(channel_labels):
                if i < len(subplot_data):
                    self.y_data[subplot_key][channel_label].append(float(subplot_data[i]))

        # Maintain rolling history window
        if len(self.x_data) > self._options.history_length:
            self.x_data.pop(0)
            for subplot_key in self.y_data:
                for channel_label in self.y_data[subplot_key]:
                    try:
                        self.y_data[subplot_key][channel_label].pop(0)
                    except IndexError:
                        break  # empty, nothing to do.

        # Update plot
        self._update_plot()

        # Render frame if necessary
        if self._options.save_to_filename:
            self._frames_buffer.append(self.get_image_array())

    def _update_plot(self):
        """
        Update plot.
        """
        raise NotImplementedError

    def get_image_array(self):
        """
        Capture the plot image as a video frame.

        Returns
        -------
        image_array : np.ndarray
            The RGB image as a numpy array.
        """
        raise NotImplementedError


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
            The RGB image as a numpy array.
        """
        pixmap = self.widget.grab()
        qimage = pixmap.toImage()

        qimage = qimage.convertToFormat(pg.QtGui.QImage.Format_RGB888)
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())

        return np.array(ptr).reshape((qimage.height(), qimage.width(), 3))


class PyQtLinePlotterOptions(BasePlotterOptions, LinePlotterMixinOptions):
    """
    Live line plot visualization of data using PyQtGraph.

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
class PyQtLinePlotter(LinePlotterMixin[PyQtLinePlotterOptions], BasePyQtPlotter):
    """
    Real-time plot using PyQt for live sensor data visualization.
    """

    def build(self):
        super().build()

        self.curves: dict[str, list[pg.PlotCurveItem]] = {}

        # create plots for each subplot
        for subplot_idx, (subplot_key, channel_labels) in enumerate(self.subplot_structure.items()):
            # add new row if not the first plot
            if subplot_idx > 0:
                self.widget.nextRow()

            plot_widget = self.widget.addPlot(title=subplot_key if self.is_dict_data else self._options.title)
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

    def _update_plot(self):
        # update all curves
        for subplot_key, curves in self.curves.items():
            channel_labels = self.subplot_structure[subplot_key]
            for curve, channel_label in zip(curves, channel_labels):
                y_data = self.y_data[subplot_key][channel_label]
                curve.setData(x=self.x_data, y=y_data)

        if self.app:
            self.app.processEvents()

    def cleanup(self):
        super().cleanup()

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

    @property
    def run_in_thread(self) -> bool:
        return not self.show_window or gs.platform != "macOS"

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


class MPLLinePlotterOptions(BasePlotterOptions, LinePlotterMixinOptions):
    """
    Live line plot visualization of data using matplotlib.

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
class MPLLinePlotter(LinePlotterMixin[MPLLinePlotterOptions], BaseMPLPlotter):
    """
    Real-time plot using matplotlib for live sensor data visualization.

    Inherits common plotting functionality from BasePlotter.
    """

    def build(self):
        super().build()

        import matplotlib.pyplot as plt

        self.axes: list[plt.Axes] = []
        self.lines: dict[str, list[plt.Line2D]] = {}
        self.backgrounds: list[Any] = []

        # Create figure and subplots
        n_subplots = len(self.subplot_structure)
        if n_subplots == 1:
            self.fig, ax = plt.subplots(figsize=self.figsize)
            self.axes = [ax]
        else:
            self.fig, axes = plt.subplots(n_subplots, 1, figsize=self.figsize, sharex=True, constrained_layout=True)
            self.axes = axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
        self.fig.suptitle(self._options.title)

        # Create lines for each subplot
        for subplot_idx, (subplot_key, channel_labels) in enumerate(self.subplot_structure.items()):
            ax = self.axes[subplot_idx]
            ax.set_xlabel(self._options.x_label)
            ax.set_ylabel(self._options.y_label)
            ax.grid(True, alpha=0.3)

            if self.is_dict_data and n_subplots > 1:
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

    def _update_plot(self):
        self._lock.acquire()

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

            # Update lines
            channel_labels = self.subplot_structure[subplot_key]
            for line, channel_label in zip(subplot_lines, channel_labels):
                y_data = self.y_data[subplot_key][channel_label]
                line.set_data(self.x_data, y_data)
                ax.draw_artist(line)

            # Blit the updated subplot
            self.fig.canvas.blit(ax.bbox)

        self.fig.canvas.flush_events()
        self._lock.release()

    def cleanup(self):
        super().cleanup()

        self.lines.clear()
        self.backgrounds.clear()


class MPLImagePlotterOptions(BasePlotterOptions):
    """
    Live visualization of image data using matplotlib.

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
