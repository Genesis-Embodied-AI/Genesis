"""
TEMP FILE !!!
"""

import itertools
from typing import Callable

import numpy as np
import torch

import genesis as gs
from genesis.recorders.base_recorder import Recorder, RecorderOptions
from genesis.recorders.recorder_manager import register_recording
from genesis.utils import tensor_to_array

IS_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt

    IS_MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


COLORS = (
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 1.0, 0.2, 1.0),
    (0.2, 0.6, 1.0, 1.0),
    (1.0, 1.0, 0.2, 1.0),
)


class MPLImageViewerOptions(RecorderOptions):
    """Live visualization of image data using MatPlotLib."""

    pass


@register_recording(MPLImageViewerOptions)
class MPLImageViewer(Recorder):
    """
    Real-time image viewer using MatPlotLib for live image data visualization.

    Uses efficient blitting for smooth real-time updates.
    """

    def build(self):
        if not IS_MATPLOTLIB_AVAILABLE:
            gs.raise_exception(
                "[MPLImageViewer] matplotlib is not installed. Please install it with `pip install matplotlib`."
            )

        super().build()

        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.image_plot = None
        self.colorbar = None
        self.background = None

        plt.ion()
        self.fig, self.ax = plt.subplots(num="Image Viewer")
        self.ax.set_title("Image Data")

        gs.logger.info("[MPLImageViewer] created Matplotlib image window")

    def process(self, data, cur_time):
        """Process new image data and update display."""
        if self.fig is None or self.ax is None:
            return

        if isinstance(data, torch.Tensor):
            img_data = tensor_to_array(data)
        else:
            img_data = np.asarray(data)

        if img_data.ndim == 3 and img_data.shape[0] == 1:
            img_data = img_data[0]  # remove batch dimension
        elif img_data.ndim == 3 and img_data.shape[-1] in [1, 3, 4]:
            if img_data.shape[-1] == 1:
                img_data = img_data.squeeze(-1)

        if self.image_plot is None:
            vmin, vmax = np.min(img_data), np.max(img_data)
            self.image_plot = self.ax.imshow(
                img_data, vmin=vmin, vmax=vmax, cmap="plasma", origin="upper", aspect="auto"
            )
            self.colorbar = self.fig.colorbar(self.image_plot, ax=self.ax)
            self.ax.set_xlabel("Width")
            self.ax.set_ylabel("Height")

            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        else:
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
        """Clean up Matplotlib resources."""
        if self.fig:
            try:
                plt.close(self.fig)
                gs.logger.debug("[MPLImageViewer] closed Matplotlib window")
            except Exception as e:
                gs.logger.warning(f"[MPLImageViewer] Error closing window: {e}")
            finally:
                self.fig = None
                self.ax = None
                self.image_plot = None
                self.colorbar = None
                self.background = None

    @property
    def run_in_thread(self) -> bool:
        return gs.platform != "macOS"


class PointCloudDrawerOptions(RecorderOptions):
    """
    Draw point cloud data using scene.draw_debug_spheres.

    Parameters
    ----------
    sphere_radius: float
        The radius of the debug spheres used to visualize points.
    colors: list[tuple[float, float, float, float]] | None
        Custom colors for different environments. If None, uses default colors.
    max_range: float | None
        Maximum range for filtering points. If None, will try to get from sensor data.
    draw_debug_spheres: Callable | None
        Function to draw debug spheres (typically scene.draw_debug_spheres).
    clear_debug_object: Callable | None
        Function to clear debug objects (typically scene.clear_debug_object).
    """

    sphere_radius: float = 0.03
    colors: list[tuple[float, float, float, float]] | None = None
    max_range: float | None = None
    draw_debug_spheres: Callable | None = None
    clear_debug_object: Callable | None = None


@register_recording(PointCloudDrawerOptions)
class PointCloudDrawer(Recorder):
    """
    Draw point cloud data using scene.draw_debug_spheres.

    Visualizes point cloud data by drawing debug spheres in the scene with different
    colors for multiple environments.
    """

    def build(self):
        super().build()

        if self._options.draw_debug_spheres is None or self._options.clear_debug_object is None:
            gs.raise_exception(
                "[PointCloudDrawer] draw_debug_spheres and clear_debug_object functions must be provided in options"
            )
        self.debug_objects: list[object | None] = []

    def process(self, data, cur_time):
        """Process point cloud data and visualize using debug spheres."""
        if data.ndim == 2:
            data = data.unsqueeze(0)

        if len(self.debug_objects) == 0:
            self.debug_objects = [None] * data.shape[0]

        for env_idx, color in zip(range(data.shape[0]), itertools.cycle(COLORS)):
            points = data[env_idx]

            if self.debug_objects[env_idx] is not None:
                self._options.clear_debug_object(self.debug_objects[env_idx])

            self.debug_objects[env_idx] = self._options.draw_debug_spheres(
                points, radius=self._options.sphere_radius, color=color
            )

    def cleanup(self):
        """Clean up debug objects."""
        if self.debug_objects:
            for node in self.debug_objects:
                if node is not None:
                    self._options.clear_debug_object(node)
        self.debug_objects.clear()

    @property
    def run_in_thread(self) -> bool:
        return True
