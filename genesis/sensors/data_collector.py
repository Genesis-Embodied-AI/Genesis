import genesis as gs
import threading
import queue
import cv2
import csv
import os
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
from genesis.utils.misc import tensor_to_array
from genesis.utils.tools import animate


class DataOutType(Enum):
    CSV = "csv"
    VIDEO = "video_write"
    VIDEO_STREAM = "video_stream"
    CALLBACK = "callback"

    @staticmethod
    def get_output_handler(config):
        if config.out_type == DataOutType.CSV:
            return CsvFileOutput(config, **config.handler_kwargs)
        elif config.out_type == DataOutType.VIDEO:
            return VideoFileOutput(config, **config.handler_kwargs)
        elif config.out_type == DataOutType.VIDEO_STREAM:
            return VideoStreamToFileOutput(config, **config.handler_kwargs)
        elif config.out_type == DataOutType.CALLBACK:
            return CallbackOutput(config, **config.handler_kwargs)
        else:
            raise ValueError(f"Unknown data handler type: {config.out_type}")


@dataclass
class DataStreamConfig:
    out_type: DataOutType
    hz: Optional[float] = None  # simulation Hz (steps per second at sim dt)
    buffer_size: int = 0  # size of the data queue buffer, 0 is infinite
    buffer_full_wait_time: float = 0.1  # time to wait for buffer space to become available
    _steps_per_sample: int = 1  # how often to sample data, default is every step
    handler_kwargs: Optional[dict] = None  # additional kwargs for the handler


class DataCollector:
    def __init__(
        self,
        sensor,
        configs: DataStreamConfig | List[DataStreamConfig],
    ):
        if not isinstance(configs, (list, tuple)):
            configs = [configs]

        self._sensor = sensor
        self.configs = configs
        self.data_queues = []
        self.processor_threads = []
        self.is_recording = False
        self.is_paused = False

        self.sim_dt = sensor._sim.dt

        self.output_handlers = []

    def start_recording(self):
        """Start data recording."""
        if self.is_recording:
            self.is_paused = False
            return

        for idx, config in enumerate(self.configs):
            data_queue = queue.Queue(maxsize=config.buffer_size)
            self.data_queues.append(data_queue)

            if not config.hz:
                config.hz = 1.0 / self.sim_dt  # default to real-time speed
            steps_per_sample_float = 1.0 / (config.hz * self.sim_dt)
            steps_per_sample = max(1, round(steps_per_sample_float))
            if steps_per_sample_float != steps_per_sample:
                gs.logger.warning(
                    f"Data collection hz={config.hz} is not a valid integer multiple step size of simulation dt. "
                    f"Using hz={1.0 / steps_per_sample / self.sim_dt} instead."
                )
            config._steps_per_sample = steps_per_sample

            thread = threading.Thread(target=self.make_process_callback(idx))
            thread.start()
            self.processor_threads.append(thread)

            handler = DataOutType.get_output_handler(config)
            handler.initialize()
            self.output_handlers.append(handler)

        self.is_recording = True
        self.is_paused = False

    def pause_recording(self):
        """Pause data recording. Resume with `start_recording()`."""
        self.is_paused = True

    def stop_recording(self):
        """Stop and complete data recording."""
        self.is_recording = False
        for i, handler in enumerate(self.output_handlers):
            handler.cleanup()
            self.processor_threads[i].join()

    def destroy(self):
        if self.is_active:
            self.stop_recording()

    def step(self, cur_step_global: int):
        """
        If recording, process data every `steps_per_sample` steps.
        """
        if not self.is_recording or self.is_paused:
            return
        for idx, config in enumerate(self.configs):
            if cur_step_global % config._steps_per_sample == 0:
                self.read_sensor(idx)

    def read_sensor(self, handler_idx=0):
        """
        Read and process data from the sensor.
        Usually called by step(), but can also be called manually to process data immediately.
        """
        data = self._sensor.read()
        data_queue = self.data_queues[handler_idx]
        try:
            data_queue.put(data, block=True, timeout=self.configs[handler_idx].buffer_full_wait_time)
        except queue.Full:
            gs.logger.warning("Data queue is full, dropping oldest data sample.")
            data_queue.get_nowait()
            data_queue.put_nowait(data)

    def make_process_callback(self, handler_idx):
        data_queue = self.data_queues[handler_idx]
        def _process_data():
            """Background thread that processes and outputs data"""
            while self.is_recording or not data_queue.empty():
                try:
                    data = data_queue.get(timeout=1.0)
                    for handler in self.output_handlers:
                        handler.process(data)
                    data_queue.task_done()
                except queue.Empty:
                    continue
        return _process_data

    @property
    def is_active(self):
        """Check if the data streamer is active"""
        return self.is_recording and not self.is_paused


class DataOutput:
    """Base class for datastream output handlers"""

    def __init__(self, config: DataStreamConfig):
        self.config = config

    def initialize(self):
        raise NotImplementedError()

    def process(self, data):
        raise NotImplementedError()

    def cleanup(self):
        raise NotImplementedError()


class VideoFileOutput(DataOutput):
    """
    Buffer all frames and writes video at cleanup.

    Parameters
    ----------
    config : StreamingConfig
        The recording configuration.
    filename : str
        The name of the video file to save the frames.
    fps : float, optional
        Frames per second for the video. If None, it will be set to real-time speed based on hz.
    streams_idx : array-like, optional
        If frames of multiple videos are incoming, this specifies which ones to save.
    """

    def __init__(self, config: DataStreamConfig, filename: str, fps=None, streams_idx=None):
        super().__init__(config)
        assert filename.endswith(".mp4"), "Video output must be an .mp4 file"
        self.filename = filename
        self.fps = fps or config.hz
        self.frames = []
        self.streams_idx = streams_idx

    def initialize(self):
        self.frames.clear()

    def process(self, data):
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


class VideoStreamToFileOutput(DataOutput):
    """
    Stream video frames to file.

    Parameters
    ----------
    config : StreamingConfig
        The recording configuration.
    filename : str
        The name of the video file to save the frames.
    fps : int
        Frames per second for the video.
    streams_idx : array-like, optional
        If frames of multiple videos are incoming, this specifies which ones to save.
    """

    def __init__(
        self,
        config: DataStreamConfig,
        filename: str,
        shape,
        fps=None,
        fourcc="avc1",
        is_color=True,
        streams_idx=None,
    ):
        super().__init__(config)
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
        self.config = config
        self.video_writers = []
        self.width, self.height = shape[:2]
        self.fps = fps or config.hz
        self.fourcc = fourcc
        self.is_color = is_color
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

    def initialize(self):
        os.makedirs(os.path.abspath(os.path.dirname(self.filenames[0])), exist_ok=True)
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

    def process(self, data):
        if len(self.streams_idx) == 1:
            data = [data]
        for i, frame in enumerate(data):
            assert isinstance(frame, np.ndarray), "Video image data must be a numpy array"
            assert (
                frame.shape[0] == self.height and frame.shape[1] == self.width
            ), f"Video frame shape {frame.shape} does not match expected shape ({self.height}, {self.width})"
            if self.is_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writers[i].write(frame)

    def cleanup(self):
        for i, video_writer in enumerate(self.video_writers):
            video_writer.release()
            gs.logger.info(f"Video saved to ~<{self.filenames[i]}>~.")
        self.video_writers.clear()


class CsvFileOutput(DataOutput):
    """
    CSV output handler for saving data to a CSV file.

    Parameters
    ----------
    config : StreamingConfig
        The recording configuration.
    filename : str
        The name of the CSV file to save the data.
    flush_interval : int | None
        Determines how often the data is saved to disk. None = only at cleanup, 0 = every write, >0 = every N writes.
    """

    def __init__(
        self, config: DataStreamConfig, filename: str, flush_interval: Optional[int] = None
    ):
        super().__init__(config)
        assert filename.endswith(".csv"), "CSV output must be a CSV file"
        self.filename = filename
        self.file_handle = None
        self.csv_writer = None
        self.flush_interval = flush_interval
        if flush_interval is not None:
            self.flush_counter = 0

    def initialize(self):
        os.makedirs(os.path.abspath(os.path.dirname(self.filename)), exist_ok=True)
        self.file_handle = open(self.filename, "w", encoding="utf-8", newline="")
        self.csv_writer = csv.writer(self.file_handle)

    def process(self, data):
        row_data = np.atleast_1d(tensor_to_array(data))
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


class CallbackOutput(DataOutput):
    """
    A simple data handler that calls the provided callback function.
    """

    def __init__(self, config: DataStreamConfig, callback):
        super().__init__(config)
        assert callable(callback), "Callback must be a callable function"
        self.callback = callback

    def initialize(self):
        pass

    def process(self, data):
        self.callback(data)

    def cleanup(self):
        pass
