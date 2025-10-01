import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pydantic import Field

import genesis as gs
from genesis.utils import tensor_to_array

from .base_recorder import Recorder, RecorderOptions
from .recorder_manager import register_recording

IS_PYAV_AVAILABLE = False
try:
    import av

    IS_PYAV_AVAILABLE = True
except ImportError:
    pass


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

    filename: str
    save_on_reset: bool = False


class BaseFileWriter(Recorder):
    """
    Base class for file writers.

    Handles filename counter when save_on_reset is True.
    """

    def build(self):
        super().build()
        self.counter = 0

        os.makedirs(os.path.abspath(os.path.dirname(self._options.filename)), exist_ok=True)
        self._initialize_writer()

    def reset(self, envs_idx=None):
        super().reset(envs_idx)

        # no envs specific saving supported
        if self._options.save_on_reset:
            self.cleanup()
            self.counter += 1
            self._initialize_writer()

    def _get_filename(self):
        if self._options.save_on_reset:
            path, ext = os.path.splitext(self._options.filename)
            return f"{path}_{self.counter}{ext}"
        return self._options.filename

    def _initialize_writer(self):
        pass


class VideoFileWriterOptions(BaseFileWriterOptions):
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

    fps: int | None = None
    name: str = ""
    codec: str = "libx264"
    bitrate: float = 1.0
    codec_options: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, context):
        super().model_post_init(context)

        if self.codec not in av.codecs_available:
            gs.raise_exception(f"[{type(self).__name__}] Codec '{self._options.codec}' not supported.")

        if not self.filename.endswith(".mp4"):
            gs.raise_exception(f"[{type(self).__name__}] Video filename must have '.mp4' extension.")


@register_recording(VideoFileWriterOptions)
class VideoFileWriter(BaseFileWriter):
    video_container: "av.container.OutputContainer | None"
    video_stream: "av.video.stream.VideoStream | None"
    video_frame: "av.video.frame.VideoFrame | None"
    video_buffer: "np.ndarray | None"

    def build(self):
        if not IS_PYAV_AVAILABLE:
            gs.raise_exception("PyAV is not installed. Please install it with `pip install av`.")

        self.video_container = None
        self.video_stream = None
        self.video_frame = None
        self.video_buffer = None

        self.fps = int(
            round(
                1.0 / (self._steps_per_sample * self._manager._step_dt)
                if self._options.fps is None
                else self._options.fps
            )
        )

        super().build()

    def _initialize_writer(self):
        video_path = self._get_filename()
        video_name = self._options.name or Path(video_path).stem

        # Create ffmpeg video container
        self.video_container = av.open(video_path, mode="w")
        self.video_container.metadata["title"] = video_name

    def _initialize_data(self, data):
        assert isinstance(data, (np.ndarray, torch.Tensor))
        is_color = data.ndim == 3 and data.shape[-1] == 3
        if isinstance(data, np.ndarray):
            is_dtype_int = np.issubdtype(data.dtype, np.integer)
        else:
            is_dtype_int = not torch.is_floating_point(data)
        if data.ndim != 2 + is_color or not is_dtype_int:
            gs.raise_exception(f"[{type(self).__name__}] Data must be either grayscale [H, W] or color [H, W, RGB]")
        height, width, *_ = data.shape

        # Create ffmpeg video stream
        self.video_stream = self.video_container.add_stream(self._options.codec, rate=self.fps)
        assert isinstance(self.video_stream, av.video.stream.VideoStream)
        self.video_stream.width, self.video_stream.height = (width, height)
        self.video_stream.pix_fmt = "yuv420p"
        self.video_stream.bit_rate = int(self._options.bitrate * (8 * 1024**2))
        self.video_stream.codec_context.options = self._options.codec_options

        # Create frame storage once for efficiency
        if is_color:
            self.video_frame = av.VideoFrame(width, height, "rgb24")
            frame_plane = self.video_frame.planes[0]
            self.video_buffer = np.asarray(memoryview(frame_plane)).reshape((-1, frame_plane.line_size // 3, 3))
        else:
            self.video_frame = av.VideoFrame(width, height, "gray8")
            frame_plane = self.video_frame.planes[0]
            self.video_buffer = np.asarray(memoryview(frame_plane)).reshape((-1, frame_plane.line_size))

    def process(self, data, cur_time):
        if self.video_buffer is None:
            self._initialize_data(data)

        if isinstance(data, torch.Tensor):
            data = tensor_to_array(data)
        data = data.astype(np.uint8)

        # Write frame
        self.video_buffer[: data.shape[0], : data.shape[1]] = data
        for packet in self.video_stream.encode(self.video_frame):
            self.video_container.mux(packet)

    def cleanup(self):
        if self.video_container is not None:
            # Finalize video recording.
            # Note that 'video_stream' may be None if 'process' what never called.
            if self.video_stream is not None:
                for packet in self.video_stream.encode(None):
                    self.video_container.mux(packet)
            self.video_container.close()

            gs.logger.info(f'Video saved to "~<{self._options.filename}>~".')

            self.video_container = None
            self.video_stream = None
            self.video_frame = None
            self.video_buffer = None

    @property
    def run_in_thread(self) -> bool:
        return False


class CSVFileWriterOptions(BaseFileWriterOptions):
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

    header: tuple[str, ...] | None = None
    save_every_write: bool = False

    def model_post_init(self, context):
        super().model_post_init(context)
        if not self.filename.lower().endswith(".csv"):
            gs.raise_exception(f"[{type(self).__name__}] CSV output must be a .csv file")


@register_recording(CSVFileWriterOptions)
class CSVFileWriter(BaseFileWriter):
    def _initialize_writer(self):
        self.wrote_data = False
        self.file_handle = open(self._get_filename(), "w", encoding="utf-8", newline="")
        self.csv_writer = csv.writer(self.file_handle)

    def _sanitize_to_list(self, value):
        if isinstance(value, (torch.Tensor, np.ndarray)):
            return value.reshape((-1,)).tolist()
        elif isinstance(value, (int, float, bool)):
            return [value]
        elif isinstance(value, (list, tuple)):
            return value
        else:
            gs.raise_exception(f"[{type(self).__name__}] Unsupported data type: {type(value)}")

    def process(self, data, cur_time):
        row_data = [cur_time]
        if isinstance(data, dict):
            for value in data.values():
                row_data.extend(self._sanitize_to_list(value))
        else:
            row_data.extend(self._sanitize_to_list(data))

        if not self.wrote_data:  # write header
            header = ["timestamp"]
            if self._options.header:
                header.extend(self._options.header)
            else:
                if isinstance(data, dict):
                    for key, val in data.items():
                        if hasattr(val, "__len__"):
                            header.extend([f"{key}_{i}" for i in range(len(val))])
                        else:
                            header.append(key)
                else:
                    header.extend([f"data_{i}" for i in range(1, len(row_data))])
            if len(header) != len(row_data):
                gs.raise_exception(f"[{type(self).__name__}] header length does not match data length.")
            self.csv_writer.writerow(header)

        self.wrote_data = True
        self.csv_writer.writerow(row_data)
        if self._options.save_every_write:
            self.file_handle.flush()

    def cleanup(self):
        if self.file_handle:
            if self.wrote_data:
                self.file_handle.close()
                gs.logger.info(f'[CSVFileWriter] Saved to ~<"{self._get_filename()}">~.')
            else:
                self.file_handle.close()
                os.remove(self._get_filename())  # delete empty file

    @property
    def run_in_thread(self) -> bool:
        return True


class NPZFileWriterOptions(BaseFileWriterOptions):
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

    def model_post_init(self, context):
        super().model_post_init(context)
        if not self.filename.lower().endswith(".npz"):
            gs.raise_exception(f"[{type(self).__name__}] NPZ output must be an .npz file")


@register_recording(NPZFileWriterOptions)
class NPZFileWriter(BaseFileWriter):
    def build(self):
        self.all_data: dict[str, list] = defaultdict(list)

        super().build()

    def process(self, data, cur_time):
        self.all_data["timestamp"].append(cur_time)
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    value = tensor_to_array(value)
                assert isinstance(value, (int, float, bool, list, tuple, np.ndarray))
                self.all_data[key].append(value)
        else:
            self.all_data["data"].append(tensor_to_array(data))

    def cleanup(self):
        filename = self._get_filename()
        if self.all_data["timestamp"]:  # at least one data point was collected
            try:
                np.savez_compressed(filename, **self.all_data)
            except ValueError as error:
                gs.logger.warning(f"NPZFileWriter: saving as dtype=object due to ValueError: {error}")
                np.savez_compressed(filename, **{k: np.array(v, dtype=object) for k, v in self.all_data.items()})
            gs.logger.info(f'[NPZFileWriter] Saved data with keys {list(self.all_data.keys())} to ~<"{filename}">~.')
            self.all_data.clear()

    @property
    def run_in_thread(self) -> bool:
        return True
