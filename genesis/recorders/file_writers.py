import csv
import os
from collections import defaultdict

import cv2
import numpy as np
import torch

import genesis as gs
from genesis.utils import tensor_to_array

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
    """
    Base class for file writers.

    Handles filename counter when save_on_reset is True.
    """

    def build(self):
        super().build()
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


class VideoFileWriterOptions(BaseFileWriterOptions):
    """
    Stream video frames to file using cv2.VideoWriter.

    The cv2 writer streams data directly to the file instead of buffering it in memory.
    Incoming data should either be grayscale [H, W] or color [H, W, RGB] where values are uint8 (0, 255).

    Parameters
    ----------
    filename : str
        The path of the output video file ending in ".mp4" or ".avi".
    fps : float, optional
        Frames per second for the video. Defaults to the data collection Hz ("real-time").
    fourcc : str, optional
        The codec to use for the video file. Defaults to "h264" for .mp4 files.
        Supported encodings: .mp4 -> "avc1", "mp4v", "h264"; .avi -> "XVID", "MJPG"
    save_on_reset: bool, optional
        Whether to save the data on reset. Defaults to False.
        If True, a counter will be added to the filename and incremented on each reset.
    """

    FOURCC_MAP: dict[str, tuple[str, ...]] = {
        ".mp4": ("h264", "avc1", "mp4v"),
        ".avi": ("XVID", "MJPG"),
    }

    fps: float | None = None
    fourcc: str = "h264"

    def validate(self):
        super().validate()
        is_valid_ext = False
        for ext, codecs in self.FOURCC_MAP.items():
            if self.filename.endswith(ext):
                if self.fourcc in codecs:
                    is_valid_ext = True
                    break
                else:
                    gs.raise_exception(
                        f"[{type(self).__name__}] '{self.fourcc}' is not a supported video codec for {ext}, "
                        f"expected one of {codecs}"
                    )
        if not is_valid_ext:
            gs.raise_exception(
                f"[{type(self).__name__}] Video filename must end with one of {tuple(self.FOURCC_MAP.keys())}."
            )


@register_recording(VideoFileWriterOptions)
class VideoFileWriter(BaseFileWriter):

    def build(self):
        super().build()
        self.video_writer: cv2.VideoWriter | None = None
        self.fps = (
            1.0 / (self._steps_per_sample * self._manager._step_dt) if self._options.fps is None else self._options.fps
        )

        # lazy init these values when processing the first frame
        self.width, self.height = 0, 0
        self.is_color: bool | None = None

        os.makedirs(os.path.abspath(os.path.dirname(self._options.filename)), exist_ok=True)

    def _initialize_data(self, data):
        assert isinstance(data, (np.ndarray, torch.Tensor))
        self.is_color = data.ndim == 3 and data.shape[-1] == 3
        if not self.is_color and data[0].ndim != 2:
            gs.raise_exception(f"[{type(self).__name__}] Data must be either grayscale [H, W] or color [H, W, RGB]")
        self.height, self.width, *_ = data.shape

    def _initialize_writer(self):
        filename = self._get_filename()
        self.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*self._options.fourcc),
            self.fps,
            (self.width, self.height),
            self.is_color,
        )
        if not self.video_writer.isOpened():
            gs.raise_exception(f'[{type(self).__name__}] Failed to open video writer for "{filename}"')

    def process(self, data, cur_time):
        if self.is_color is None:
            self._initialize_data(data)
        if self.video_writer is None:
            self._initialize_writer()

        if isinstance(data, torch.Tensor):
            data = tensor_to_array(data)
        data = data.astype(np.uint8)

        if self.is_color:
            data = np.flip(data, axis=-1)  # convert from RGB to BGR

        self.video_writer.write(data)

    def cleanup(self):
        if self.video_writer is not None:
            self.video_writer.release()
            gs.logger.info(f'Video saved to "~<{self._options.filename}>~".')
            self.video_writer = None

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

    def validate(self):
        super().validate()
        if not self.filename.lower().endswith(".csv"):
            gs.raise_exception(f"[{type(self).__name__}] CSV output must be a .csv file")


@register_recording(CSVFileWriterOptions)
class CSVFileWriter(BaseFileWriter):

    def build(self):
        super().build()

        os.makedirs(os.path.abspath(os.path.dirname(self._options.filename)), exist_ok=True)
        self._initialize_writer()

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

    Can handle any numeric or array-like or dict[str, array-like] data, e.g. from sensors.

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

    def build(self):
        super().build()
        self.all_data: dict[str, list] = defaultdict(list)

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
