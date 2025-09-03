import csv
import os
from collections import defaultdict
from collections.abc import Iterable

import cv2
import numpy as np
import torch

import genesis as gs
from genesis.utils import tensor_to_array
from genesis.utils.tools import animate

from .base_recorder import Recorder, RecorderOptions
from .recorder_manager import register_recording


class BaseFileWriterOptions(RecorderOptions):
    """Base class for file writer options."""

    filename: str


class VideoFileWriterOptions(BaseFileWriterOptions):
    """
    Buffer all frames and writes video at cleanup using moviepy.

    Parameters
    ----------
    filename : str
        The path of the output video file ending in ".mp4".
    fps : float, optional
        Frames per second for the video. Defaults to the data collection Hz ("real-time").
    """

    fps: float | None = None

    def validate(self):
        super().validate()
        if not self.filename.lower().endswith(".mp4"):
            gs.raise_exception(f"[{type(self).__name__}] Video output should be an .mp4 file")


class BaseVideoFileWriter(Recorder):
    def initialize(self):
        self.fps = 1.0 / (self._steps_per_sample * self._manager._step_dt)


@register_recording(VideoFileWriterOptions)
class VideoFileWriter(BaseVideoFileWriter):

    def initialize(self):
        super().initialize()
        self.frames: list[np.ndarray] = []

    def process(self, data, cur_time):
        self.frames.append(data)

    def cleanup(self):
        if self.frames:
            animate(self.frames, filename=self._options.filename, fps=self.fps)
            self.frames.clear()
        else:
            gs.logger.warning(f"[{type(self).__name__}] No frames to write to video file.")

    @property
    def run_in_thread(self) -> bool:
        return True


class Cv2VideoFileWriterOptions(VideoFileWriterOptions):
    """
    Stream video frames to file using cv2.VideoWriter.

    The cv2 writer streams data directly to the file instead of buffering it in memory.

    Parameters
    ----------
    filename : str
        The path of the output video file ending in ".mp4" or ".avi".
    fps : float, optional
        Frames per second for the video. Defaults to the data collection Hz ("real-time").
    fourcc : str, optional
        The codec to use for the video file. Defaults to "avc1" for .mp4 files.
        Supported encodings: .mp4 -> "avc1", "mp4v", "h264"; .avi -> "XVID", "MJPG"
    """

    fourcc: str = "avc1"

    def validate(self):
        super().validate()
        if self.filename.endswith(".mp4"):
            if self.fourcc not in ["avc1", "mp4v", "h264"]:
                gs.raise_exception(
                    f"[{type(self).__name__}] '{self.fourcc}' is not a supported video codec for .mp4, "
                    "expected 'avc1', 'mp4v', or 'h264'"
                )
        elif self.filename.endswith(".avi"):
            if self.fourcc not in ["XVID", "MJPG"]:
                gs.raise_exception(
                    f"[{type(self).__name__}] '{self.fourcc}' is not a supported video codec for .avi, "
                    "expected 'XVID' or 'MJPG'"
                )
        else:
            gs.raise_exception(f"[{type(self).__name__}] Video filename must end with .mp4 or .avi.")


@register_recording(Cv2VideoFileWriterOptions)
class Cv2VideoFileWriter(BaseVideoFileWriter):

    def initialize(self):
        super().initialize()
        self.video_writer: cv2.VideoWriter | None = None

        # lazy init these values when processing the first frame
        self.width, self.height = 0, 0
        self.is_color: bool | None = None

        os.makedirs(os.path.abspath(os.path.dirname(self._options.filename)), exist_ok=True)

    def _lazy_initialize(self, data):
        self.is_color = len(data[0].shape) == 3 and data[0].shape[-1] == 3
        if not self.is_color and len(data[0].shape) != 2:
            gs.raise_exception(f"[{type(self).__name__}] Data must be either grayscale [H, W] or color [H, W, RGB]")
        self.height, self.width = data[0].shape[:2]

        video_writer = cv2.VideoWriter(
            self._options.filename,
            cv2.VideoWriter_fourcc(*self._options.fourcc),
            self.fps,
            (self.width, self.height),
            self.is_color,
        )
        if not video_writer.isOpened():
            gs.raise_exception(f'[{type(self).__name__}] Failed to open video writer for "{self._options.filename}"')

    def process(self, data, cur_time):
        if self.is_color is None:
            self._lazy_initialize(data)

        if self.is_color:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        self.video_writer.write(data)

    def cleanup(self):
        if self.video_writer:
            self.video_writer.release()
            gs.logger.info(f'Video saved to "~<{self._options.filename}>~".')
            self.video_writer = None

    @property
    def run_in_thread(self) -> bool:
        return True


class CSVFileWriterOptions(BaseFileWriterOptions):
    """
    Writes to a .csv file using `csv.writer`.

    Can handle any array-like or dict[str, array-like] output, e.g. from sensors.

    Parameters
    ----------
    filename : str
        The name of the CSV file to save the data.
    header : list[str] | None, optional
        Column headers for the CSV file.
    flush_interval : int | None, optional
        Determines how often the data is saved to disk. None = only at cleanup, 0 = every write, >0 = every N writes.
    """

    header: list[str] | None = None
    flush_interval: int | None = None

    def validate(self):
        super().validate()
        if not self.filename.lower().endswith(".csv"):
            gs.raise_exception(f"[{type(self).__name__}] CSV output must be a .csv file")
        if self.flush_interval is not None and self.flush_interval < 0:
            gs.raise_exception(f"[{type(self).__name__}] flush_interval must be None or >= 0")


@register_recording(CSVFileWriterOptions)
class CSVFileWriter(Recorder):

    def initialize(self):
        self.file_handle = None
        self.csv_writer = None
        self.wrote_header = False

        if self._options.flush_interval is not None:
            self.flush_counter = 0

        os.makedirs(os.path.abspath(os.path.dirname(self._options.filename)), exist_ok=True)
        self.file_handle = open(self._options.filename, "w", encoding="utf-8", newline="")
        self.csv_writer = csv.writer(self.file_handle)

    def process(self, data, cur_time):
        row_data = [cur_time]
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, torch.Tensor):
                    value = value.tolist()
                row_data.append(value)
        else:
            row_data.extend(data.tolist())

        if not self.wrote_header:
            header = {"timestamp"}
            if self._options.header:
                header.update(self._options.header)
            else:
                if isinstance(data, dict):
                    for key in data.keys():
                        header.add(key)
                else:
                    header.update(["data" + i for i in (len(data) if isinstance(data, Iterable) else 1)])
            if len(header) != len(row_data):
                gs.raise_exception(f"[{type(self).__name__}] header length does not match data length.")
            self.csv_writer.writerow(header)
            self.wrote_header = True

        self.csv_writer.writerow(row_data)
        if self._options.flush_interval is not None:
            self.flush_counter += 1
            if self.flush_counter >= self._options.flush_interval:
                self.file_handle.flush()
                self.flush_counter = 0

    def cleanup(self):
        if self.file_handle:
            gs.logger.info(f'Saving CSV file to ~<"{self._options.filename}">~...')
            self.file_handle.close()
            gs.logger.info("File saved.")

    @property
    def run_in_thread(self) -> bool:
        return True


class NPZFileWriterOptions(BaseFileWriterOptions):
    """
    Buffers all data and writes to a .npz file at cleanup.

    Can handle any array-like or dict[str, array-like] output, e.g. from sensors.

    Parameters
    ----------
    filename : str
        The name of the .npz file to save the data.
    """

    def validate(self):
        super().validate()
        if not self.filename.lower().endswith(".npz"):
            gs.raise_exception(f"[{type(self).__name__}] NPZ output must be an .npz file")


@register_recording(NPZFileWriterOptions)
class NPZFileWriter(Recorder):

    def initialize(self):
        self.all_data: dict[str, list] = defaultdict(list)

    def process(self, data, cur_time):
        self.all_data["timestamp"].append(cur_time)
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    value = tensor_to_array(value)
                self.all_data[key].append(value)
        else:
            self.all_data["data"].append(tensor_to_array(data))

    def cleanup(self):
        gs.logger.info(f'Saving data to "~<{self._options.filename}>~"...')
        saved_data = self.all_data
        try:
            np.savez_compressed(self._options.filename, **saved_data)
        except ValueError as error:
            gs.logger.warning(f"NPZFileWriter: saving as dtype=object due to ValueError: {error}")
            saved_data = {k: np.array(v, dtype=object) for k, v in self.all_data.items()}
            np.savez_compressed(self._options.filename, **saved_data)
        gs.logger.info(f"NPZ data saved with keys {list(saved_data.keys())}.")

    @property
    def run_in_thread(self) -> bool:
        return True
