import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from genesis.options.recorders import (
    VideoFile as VideoFileWriterOptions,
    CSVFile as CSVFileWriterOptions,
    NPZFile as NPZFileWriterOptions,
)
from genesis.utils import tensor_to_array

from .base_recorder import Recorder
from .recorder_manager import register_recording

try:
    import av
except ImportError:
    pass


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


@register_recording(VideoFileWriterOptions)
class VideoFileWriter(BaseFileWriter):
    video_container: "av.container.OutputContainer | None"
    video_stream: "av.video.stream.VideoStream | None"
    video_frame: "av.video.frame.VideoFrame | None"
    video_buffer: "np.ndarray | None"

    def build(self):
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
