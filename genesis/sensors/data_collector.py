import genesis as gs
import threading
import queue
from typing import Optional
from dataclasses import dataclass
from .data_handlers import DataHandler


@dataclass
class DataRecordingOptions:
    """
    Options for recording data from a sensor.

    Parameters
    ----------
    handler: DataHandler
        The handler that will process the recorded data.
    hz: float, optional
        The frequency at which to sample data, in Hz (samples per second).
        If None, the sensor will be sampled every simulation step.
    buffer_size: int
        The size of the data queue buffer. Defaults to 0, which means infinite size.
    buffer_full_wait_time: float
        The time to wait for buffer space to become available when the buffer is full. Defaults to 0.1 seconds.
    """

    handler: DataHandler
    hz: Optional[float] = None
    buffer_size: int = 0
    buffer_full_wait_time: float = 0.1

    _steps_per_sample: int = 1  # how often to sample data, calculated based on hz


class DataCollector:
    def __init__(
        self,
        sensor,
        options: DataRecordingOptions | list[DataRecordingOptions],
    ):
        if not isinstance(options, (list, tuple)):
            options = [options]

        self._sensor = sensor
        self.options_list = options
        self.data_queues: list[queue.Queue] = []
        self.processor_threads: list[threading.Thread] = []
        self.output_handlers: list[DataHandler] = []
        self.is_recording = False
        self.is_paused = False
        self.sim_dt = sensor._sim.dt

    def start_recording(self):
        """Start data recording."""
        if self.is_recording:
            self.is_paused = False
            return

        self.is_recording = True
        self.is_paused = False

        for idx, options in enumerate(self.options_list):
            data_queue = queue.Queue(maxsize=options.buffer_size)
            self.data_queues.append(data_queue)

            if options.hz:
                steps_per_sample_float = 1.0 / (options.hz * self.sim_dt)
                steps_per_sample = max(1, round(steps_per_sample_float))
                if steps_per_sample_float != steps_per_sample:
                    gs.logger.warning(
                        f"Data collection hz={options.hz} is not a valid integer multiple step size of simulation dt. "
                        f"Using hz={1.0 / steps_per_sample / self.sim_dt} instead."
                    )
                options._steps_per_sample = steps_per_sample

            options.handler.initialize()
            self.output_handlers.append(options.handler)

            thread = threading.Thread(target=self.make_process_callback(idx))
            thread.start()
            self.processor_threads.append(thread)

    def pause_recording(self):
        """Pause data recording. Resume with `start_recording()`."""
        self.is_paused = True

    def stop_recording(self):
        """Stop and complete data recording."""
        self.is_recording = False
        for i, handler in enumerate(self.output_handlers):
            handler.cleanup()
            self.processor_threads[i].join()

    def step(self, cur_step_global: int):
        """
        If recording, process data every `steps_per_sample` steps.
        """
        if not self.is_recording or self.is_paused:
            return
        for idx, options in enumerate(self.options_list):
            if cur_step_global % options._steps_per_sample == 0:
                self.read_sensor(idx)

    def read_sensor(self, handler_idx=0):
        """
        Read and process data from the sensor.
        Usually called by step(), but can also be called manually to process data immediately.
        """
        data = self._sensor.read()
        data_queue = self.data_queues[handler_idx]
        try:
            data_queue.put(data, block=True, timeout=self.options_list[handler_idx].buffer_full_wait_time)
        except queue.Full:
            gs.logger.warning("Data queue is full, dropping oldest data sample.")
            data_queue.get_nowait()
            data_queue.put_nowait(data)

    def make_process_callback(self, handler_idx):
        def _process_data():
            """Background thread that processes and outputs data"""
            while self.is_recording or not self.data_queues[handler_idx].empty():
                try:
                    data = self.data_queues[handler_idx].get(timeout=1.0)
                    self.output_handlers[handler_idx].process(data)
                    self.data_queues[handler_idx].task_done()
                except queue.Empty:
                    continue

        return _process_data

    @property
    def is_active(self):
        """Check if the data streamer is active"""
        return self.is_recording and not self.is_paused
