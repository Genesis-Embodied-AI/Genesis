from typing import Annotated, Any

from pydantic import BeforeValidator, Field, StrictBool

import genesis as gs
from genesis.typing import NonNegativeInt, PathType, PositiveFloat, PositiveInt, PositiveVec2IType

from .options import Options


IS_PYAV_AVAILABLE = False
try:
    import av

    IS_PYAV_AVAILABLE = True
except ImportError:
    pass


class RecorderOptions(Options):
    """
    Options for recording simulation data by automatically sampling data from a data source, e.g. a sensor.

    Parameters
    ----------
    hz: float, optional
        The frequency at which to sample data, in Hz (samples per second).
        If None, the data will be sampled every step.
    buffer_size: int, optional
        Applicable when run_in_thread is True. The size of the data queue buffer.
        Defaults to 0, which means infinite size.
    buffer_full_wait_time: float, optional
        Applicable when run_in_thread is True. The time to wait for buffer space to become available when the
        buffer is full. Defaults to 0.1 seconds.
    """

    hz: PositiveFloat | None = None
    buffer_size: NonNegativeInt = 0
    buffer_full_wait_time: PositiveFloat = 0.1


class BaseFileWriterOptions(RecorderOptions):
    """
    Base class for file writer options.

    Parameters
    ----------
    filename: str
        The path of the output file.
    save_on_reset: bool, optional
        Whether to save the data on reset. Defaults to False.
        If True, a counter will be added to the filename and incremented on each reset.
    """

    filename: PathType
    save_on_reset: StrictBool = False


class VideoFile(BaseFileWriterOptions):
    """
    Stream video frames to file using PyAV.

    The PyAV writer streams data directly to the file instead of buffering it in memory. Incoming data should either be
    grayscale [H, W] or color [H, W, RGB] where values are uint8 (0, 255).

    Parameters
    ----------
    filename : str
        The path of the output video file ending in ".mp4".
    name : str
        The name of the video. Note that it may be different from filename. If empty, then filename will be used as a
        fallback. Default to "".
    fps : int, optional
        Frames per second for the video. Defaults to the data collection Hz ("real-time").
    codec : str, optional
        The codec to use for the video file. Defaults to "libx264".
    bitrate: float
        The bitrate of the video. This higher the better the quality of the video.
        Defaults to 1.0.
    codec_options: dict[str, str]
        Additional low-level codec options that will be pass to ffmpeg. Empty by default.
    save_on_reset: bool, optional
        Whether to save the data on reset. If True, a counter will be added to the filename and incremented on each
        reset. Defaults to False.
    """

    filename: PathType = Field(pattern=r"(?i).*\.mp4$")
    fps: PositiveInt | None = None
    name: str = ""
    codec: str = "libx264"
    bitrate: float = 1.0
    codec_options: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        if not IS_PYAV_AVAILABLE:
            gs.raise_exception("PyAV is not installed. Please install it with `pip install av`.")
        if self.codec not in av.codecs_available:
            gs.raise_exception(f"[{type(self).__name__}] Codec '{self.codec}' not supported.")


class CSVFile(BaseFileWriterOptions):
    """
    Writes to a .csv file using `csv.writer`.

    Can handle any array-like or dict[str, array-like] output, e.g. from sensors.
    Values must be N-dimensional tensors, arrays or scalars (np.generic, int, float, str)
    If the data or header is a dict, it cannot be further nested. Values are processed in order.

    Parameters
    ----------
    filename : str
        The name of the CSV file to save the data.
    header : tuple[str] | None, optional
        Column headers for the CSV file. It should match the format of the incoming data, where each scalar value has
        an associated header. If the data is a dict, the header should match the total length of the number of values
        after flattening the values.
    save_every_write: bool, optional
        Whether to flush the data to disk as soon as new data is recieved. Defaults to False.
    save_on_reset: bool, optional
        Whether to save the data on scene reset. Defaults to False.
        If True, a counter will be added to the filename and incremented on each reset.
    """

    filename: PathType = Field(pattern=r"(?i).*\.csv$")
    header: tuple[str, ...] | None = Field(default=None, strict=False)
    save_every_write: StrictBool = False


class NPZFile(BaseFileWriterOptions):
    """
    Buffers all data and writes to a .npz file at cleanup.

    Can handle any numeric or array-like or dict[str, array-like] data, e.g. from sensors.

    Parameters
    ----------
    filename : str
        The name of the .npz file to save the data.
    save_on_reset: bool, optional
        Whether to save the data on reset. Defaults to False.
        If True, a counter will be added to the filename and incremented on each reset.
    """

    filename: PathType = Field(pattern=r"(?i).*\.npz$")


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
    window_size: PositiveVec2IType = (800, 600)
    save_to_filename: PathType | None = None
    show_window: StrictBool | None = None


class LinePlotterMixinOptions(Options):
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

    labels: Annotated[
        tuple[str, ...] | dict[str, tuple[str, ...]] | None,
        BeforeValidator(
            lambda v: v if v is None else ({k: tuple(val) for k, val in v.items()} if isinstance(v, dict) else tuple(v))
        ),
    ] = None
    x_label: str = ""
    y_label: str = ""
    history_length: PositiveInt = 100


class PyQtLinePlot(BasePlotterOptions, LinePlotterMixinOptions):
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


class MPLLinePlot(BasePlotterOptions, LinePlotterMixinOptions):
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


class MPLImagePlot(BasePlotterOptions):
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
