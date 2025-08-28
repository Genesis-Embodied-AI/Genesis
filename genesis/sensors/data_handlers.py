import csv
import os
from collections import defaultdict
from collections.abc import Iterable
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from pydantic_core import core_schema

import genesis as gs
from genesis.utils import tensor_to_array
from genesis.utils.tools import animate


class DataHandler:
    """Base class for datastream output handlers"""

    def initialize(self):
        raise NotImplementedError()

    def process(self, data, cur_time):
        raise NotImplementedError()

    def cleanup(self):
        raise NotImplementedError()

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.is_instance_schema(cls)


class VideoFileWriter(DataHandler):
    """
    Buffer all frames and writes video at cleanup.

    Parameters
    ----------
    filename : str
        The name of the video file to save the frames.
    fps : float, optional
        Frames per second for the video. Defaults to 60 fps (may not be real-time depending on the data collection Hz).
    streams_idx : array-like, optional
        If frames of multiple videos are incoming, this specifies which ones to save.
    """

    def __init__(self, filename: str, fps=60, streams_idx=None):
        assert filename.endswith(".mp4"), "Video output must be an .mp4 file"
        self.filename = filename
        self.fps = fps
        self.frames: list[np.ndarray] = []
        self.streams_idx = streams_idx

    def initialize(self):
        self.frames.clear()

    def process(self, data, cur_time):
        """Expects incoming data to be np.ndarray or list[np.ndarray] if multiple videos are incoming."""
        if self.streams_idx is None:
            self.frames.append(data)
        else:
            self.frames.append(data[self.streams_idx])

    def cleanup(self):
        if self.frames:
            if self.streams_idx is None:
                animate(self.frames, filename=self.filename, fps=self.fps)
            else:
                filename_no_ext, ext = os.path.splitext(self.filename)
                for i in range(len(self.streams_idx)):
                    animate(self.frames, filename=f"{filename_no_ext}_{i}{ext}", fps=self.fps)
            self.frames.clear()
        else:
            gs.logger.warning("VideoFileWriter: No frames to write to video file.")


class VideoFileStreamer(DataHandler):
    """
    Stream video frames to file using cv2.VideoWriter.

    Parameters
    ----------
    filename : str
        The name of the video file to save the frames.
    fps : int
        Frames per second for the video. Defaults to 60 fps (may not be real-time depending on the data collection Hz).
    fourcc : str
        The codec to use for the video file. Defaults to "avc1" for .mp4 files.
    streams_idx : array-like, optional
        If frames of multiple videos are incoming, this specifies which ones to save.
    """

    def __init__(
        self,
        filename: str,
        fps=60,
        fourcc="avc1",
        streams_idx=None,
    ):
        if filename.endswith(".mp4"):
            assert fourcc in [
                "avc1",
                "mp4v",
                "h264",
            ], "Unsupported video codec for .mp4, use 'avc1', 'mp4v', or 'h264'"
        elif filename.endswith(".avi"):
            assert fourcc in ["XVID", "MJPG"], "Unsupported video codec for .avi, use 'XVID' or 'MJPG'"
        else:
            raise ValueError("Video filename must end with .mp4 or .avi")
        self.video_writers: list[cv2.VideoWriter] = []
        self.fps = fps
        self.fourcc = fourcc

        if streams_idx is None:
            self.streams_idx = [0]
            self.filenames = [filename]
        else:
            self.streams_idx = streams_idx
            self.filenames = []
            for i in streams_idx:
                filename_no_ext, ext = os.path.splitext(filename)
                filename = f"{filename_no_ext}_{i}{ext}"
                self.filenames.append(filename)

        # lazy init these values when processing the first frame
        self.width, self.height = 0, 0
        self.is_color: bool | None = None

    def initialize(self):
        os.makedirs(os.path.abspath(os.path.dirname(self.filenames[0])), exist_ok=True)

    def _lazy_initialize(self, data):
        self.is_color = len(data[0].shape) == 3 and data[0].shape[-1] == 3
        assert self.is_color or len(data[0].shape) == 2, "Data must be either grayscale [H, W] or color [H, W, RGB]"
        self.height, self.width = data[0].shape[:2]
        for i in self.streams_idx:
            video_writer = cv2.VideoWriter(
                self.filenames[i],
                cv2.VideoWriter_fourcc(*self.fourcc),
                self.fps,
                (self.width, self.height),
                self.is_color,
            )
            if not video_writer.isOpened():
                gs.raise_exception(f"Failed to open video writer for {self.filenames[i]}")

            self.video_writers.append(video_writer)

    def process(self, data, cur_time):
        if len(self.streams_idx) == 1:
            data = [data]

        if self.is_color is None:
            self._lazy_initialize(data)

        for i, frame in enumerate(data):
            if self.is_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writers[i].write(frame)

    def cleanup(self):
        for i, video_writer in enumerate(self.video_writers):
            video_writer.release()
            gs.logger.info(f'Video saved to "~<{self.filenames[i]}>~".')
        self.video_writers.clear()


class CSVFileWriter(DataHandler):
    """
    Writes to a .csv file using `csv.writer`. Can handle any array-like or dictionary output from sensors.

    Parameters
    ----------
    filename : str
        The name of the CSV file to save the data.
    flush_interval : int | None
        Determines how often the data is saved to disk. None = only at cleanup, 0 = every write, >0 = every N writes.
    """

    def __init__(self, filename: str, header: list[str] | None = None, flush_interval: Optional[int] = None):
        assert filename.endswith(".csv"), "CSV output must be a CSV file"
        self.filename = filename
        self.file_handle = None
        self.csv_writer = None

        self.header = header
        self.wrote_header = False

        self.flush_interval = flush_interval
        if flush_interval is not None:
            self.flush_counter = 0

    def initialize(self):
        os.makedirs(os.path.abspath(os.path.dirname(self.filename)), exist_ok=True)
        self.file_handle = open(self.filename, "w", encoding="utf-8", newline="")
        self.csv_writer = csv.writer(self.file_handle)

    def process(self, data, cur_time):
        row_data = [cur_time]
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, torch.Tensor):
                    value = tensor_to_array(value).tolist()
                row_data.append(value)
        else:
            row_data.extend(tensor_to_array(data).tolist())

        if not self.wrote_header:
            header = {"timestamp"}
            if self.header:
                header.update(self.header)
            else:
                if isinstance(data, dict):
                    for key in data.keys():
                        header.add(key)
                else:
                    header.update(["data" + i for i in (len(data) if isinstance(data, Iterable) else 1)])
            assert len(header) == len(row_data), "CSVFileWriter: header length must match data length"
            self.csv_writer.writerow(header)
            self.wrote_header = True

        self.csv_writer.writerow(row_data)
        if self.flush_interval is not None:
            self.flush_counter += 1
            if self.flush_counter >= self.flush_interval:
                self.file_handle.flush()
                self.flush_counter = 0

    def cleanup(self):
        if self.file_handle:
            gs.logger.info(f'Saving CSV file to ~<"{self.filename}">~...')
            self.file_handle.close()
            gs.logger.info("File saved.")


class NPZFileWriter(DataHandler):
    """
    Buffers all data and writes to a .npz file at cleanup. Can handle any array-like or dictionary output from sensors.

    Parameters
    ----------
    filename : str
        The name of the .npz file to save the data.
    """

    def __init__(self, filename: str):
        assert filename.endswith(".npz"), "NPZ output must be an .npz file"
        self.filename = filename
        self.all_data: dict[str, list] = defaultdict(list)

    def initialize(self):
        self.all_data.clear()

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
        gs.logger.info(f'Saving data to "~<{self.filename}>~"...')
        saved_data = {k: np.array(v) for k, v in self.all_data.items()}
        try:
            np.savez_compressed(self.filename, **saved_data)
        except ValueError as error:
            gs.logger.warning(f"NPZFileWriter: saving as dtype=object due to ValueError: {error}")
            saved_data = {k: np.array(v, dtype=object) for k, v in self.all_data.items()}
            np.savez_compressed(self.filename, **saved_data)
        gs.logger.info(f"NPZ data saved with keys {list(saved_data.keys())}.")


class CallbackHandler(DataHandler):
    """
    A simple data handler that calls the provided callback function.

    Parameters
    ----------
    callback : callable
        A function that takes the data as input and processes it.
    """

    def __init__(self, callback: Callable):
        self.callback = callback

    def initialize(self):
        pass

    def process(self, data, cur_time):
        self.callback(data, cur_time)

    def cleanup(self):
        pass
