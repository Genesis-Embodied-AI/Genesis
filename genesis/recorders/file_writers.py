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
    def initialize(self):
        self.counter = 0

    def reset(self, envs_idx=None):
        # envs_idx is ignored
        if self._options.save_on_reset:
            self.cleanup()
            self.counter += 1

    def _get_filename(self):
        if self._options.save_on_reset:
            path, ext = os.path.splitext(self._options.filename)
            return f"{path}_{self.counter}{ext}"
        return self._options.filename


class BaseVideoFileWriter(BaseFileWriter):
    def initialize(self):
        super().initialize()
        self.fps = 1.0 / (self._steps_per_sample * self._manager._step_dt)


class VideoFileWriterOptions(BaseFileWriterOptions):
    """
    Buffer all frames and writes video at cleanup using moviepy.

    Parameters
    ----------
    filename : str
        The path of the output video file ending in ".mp4".
    fps : float, optional
        Frames per second for the video. Defaults to the data collection Hz ("real-time").
    save_on_reset: bool, optional
        Whether to save the data on reset. Defaults to False.
        If True, a counter will be added to the filename and incremented on each reset.
    """

    fps: float | None = None

    def validate(self):
        super().validate()
        if not self.filename.lower().endswith(".mp4"):
            gs.raise_exception(f"[{type(self).__name__}] Video output should be an .mp4 file")


@register_recording(VideoFileWriterOptions)
class VideoFileWriter(BaseVideoFileWriter):

    def initialize(self):
        super().initialize()
        self.frames: list[np.ndarray] = []
        self.counter = 0

    def process(self, data, cur_time):
        self.frames.append(data)

    def cleanup(self):
        if self.frames:
            animate(self.frames, filename=self._get_filename(), fps=self.fps)
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
    save_on_reset: bool, optional
        Whether to save the data on reset. Defaults to False.
        If True, a counter will be added to the filename and incremented on each reset.
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

    def _initialize_data(self, data):
        self.is_color = len(data[0].shape) == 3 and data[0].shape[-1] == 3
        if not self.is_color and len(data[0].shape) != 2:
            gs.raise_exception(f"[{type(self).__name__}] Data must be either grayscale [H, W] or color [H, W, RGB]")
        self.height, self.width = data[0].shape[:2]

    def _initialize_writer(self):
        filename = self._get_filename()
        video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*self._options.fourcc),
            self.fps,
            (self.width, self.height),
            self.is_color,
        )
        if not video_writer.isOpened():
            gs.raise_exception(f'[{type(self).__name__}] Failed to open video writer for "{filename}"')

    def process(self, data, cur_time):
        if self.is_color is None:
            self._initialize_data(data)
        if self.video_writer is None:
            self._initialize_writer()

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
    save_on_reset: bool, optional
        Whether to save the data on reset. Defaults to False.
        If True, a counter will be added to the filename and incremented on each reset.
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
class CSVFileWriter(BaseFileWriter):

    def initialize(self):
        super().initialize()
        if self._options.flush_interval is not None:
            self.flush_counter = 0

        os.makedirs(os.path.abspath(os.path.dirname(self._options.filename)), exist_ok=True)
        self._initialize_writer()

    def _initialize_writer(self):
        self.wrote_header = False
        self.wrote_data = False
        self.file_handle = open(self._get_filename(), "w", encoding="utf-8", newline="")
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
        self.wrote_data = True
        if self._options.flush_interval is not None:
            self.flush_counter += 1
            if self.flush_counter >= self._options.flush_interval:
                self.file_handle.flush()
                self.flush_counter = 0

    def cleanup(self):
        if self.file_handle:
            if self.wrote_data:
                gs.logger.info(f'Saving CSV file to ~<"{self._get_filename()}">~...')
                self.file_handle.close()
                gs.logger.info("File saved.")
            else:
                self.file_handle.close()
                os.remove(self._get_filename())  # delete empty file

    def reset(self, envs_idx=None):
        # envs_idx is ignored
        super().reset(envs_idx)
        if self._options.save_on_reset:
            self._initialize_writer()

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
    save_on_reset: bool, optional
        Whether to save the data on reset. Defaults to False.
        If True, a counter will be added to the filename and incremented on each reset.
    """

    def validate(self):
        super().validate()
        if not self.filename.lower().endswith(".npz"):
            gs.raise_exception(f"[{type(self).__name__}] NPZ output must be an .npz file")


@register_recording(NPZFileWriterOptions)
class NPZFileWriter(BaseFileWriter):

    def initialize(self):
        super().initialize()
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
        filename = self._get_filename()
        saved_data = self.all_data
        if any(len(v) > 0 for v in saved_data.values()):
            gs.logger.info(f'Saving data to "~<{filename}>~"...')
            try:
                np.savez_compressed(filename, **saved_data)
            except ValueError as error:
                gs.logger.warning(f"NPZFileWriter: saving as dtype=object due to ValueError: {error}")
                saved_data = {k: np.array(v, dtype=object) for k, v in self.all_data.items()}
                np.savez_compressed(filename, **saved_data)
            gs.logger.info(f"NPZ data saved with keys {list(saved_data.keys())}.")
            self.all_data.clear()

    @property
    def run_in_thread(self) -> bool:
        return True
