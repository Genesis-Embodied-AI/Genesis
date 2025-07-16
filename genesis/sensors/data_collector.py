import genesis as gs
import threading
import queue
from dataclasses import dataclass
from .data_handlers import DataHandler
from .base_sensor import Sensor


@dataclass
class RecordingOptions:
    """
    Options for recording data from a sensor.

    Parameters
    ----------
    handler: DataHandler
        The handler that will process the recorded data.
    sensor_idx: int, optional
        The index of the sensor in the SensorDataCollector.sensors list.
    hz: float, optional
        The frequency at which to sample data, in Hz (samples per second).
        If None, the sensor will be sampled every step.
    buffer_size: int
        The size of the data queue buffer. Defaults to 0, which means infinite size.
    buffer_full_wait_time: float
        The time to wait for buffer space to become available when the buffer is full. Defaults to 0.1 seconds.
    """

    handler: DataHandler
    sensor_idx: int = -1
    hz: float | None = None
    buffer_size: int = 0
    buffer_full_wait_time: float = 0.1

    _steps_per_sample: int = 1  # how often to sample data, calculated based on hz if given


class SensorDataRecorder:
    """
    Utility class to automatically collect data from sensors and process it using specified handlers.

    Parameters
    ----------
    sensor: gs.sensors.Sensor

    """

    def __init__(
        self,
        sensors: list[Sensor] = [],
        rec_options: list[RecordingOptions] = [],
        step_dt: float | None = None,
    ):
        self._sensors = sensors
        self._rec_options = rec_options
        for opt in self._rec_options:
            if opt.sensor_idx < 0 or opt.sensor_idx >= len(self._sensors):
                gs.raise_exception(
                    f"Invalid sensor_idx {opt.sensor_idx} for RecordingOptions. "
                    f"Must be in range [0, {len(self._sensors) - 1}]."
                )
        self.step_dt = step_dt

        self._data_queues: list[queue.Queue] = []
        self._processor_threads: list[threading.Thread] = []
        self._is_recording = False
        self._is_paused = False
        self._step = 0

    def add_sensor(self, sensor: Sensor, options: RecordingOptions | list[RecordingOptions]):
        """
        Add a sensor to the data collector with specified recording options.
        """
        if not isinstance(options, list):
            options = [options]
        for opt in options:
            opt.sensor_idx = len(self._sensors)

        self._sensors.append(sensor)
        self._rec_options.extend(options)

    def start_recording(self):
        """Start data recording."""
        self._is_paused = False
        if self._is_recording:
            # Resuming recording after pause, skip initialization
            return

        self._is_recording = True
        self._step = 0

        for idx, options in enumerate(self._rec_options):
            data_queue = queue.Queue(maxsize=options.buffer_size)
            self._data_queues.append(data_queue)

            if options.hz:
                if self.step_dt:
                    steps_per_sample_float = 1.0 / (options.hz * self.step_dt)
                    steps_per_sample = max(1, round(steps_per_sample_float))
                    if steps_per_sample_float != steps_per_sample:
                        gs.logger.warning(
                            f"Data collection hz={options.hz} is not an integer multiple of step size of step dt. "
                            f"Using hz={1.0 / steps_per_sample / self.step_dt} instead."
                        )
                    options._steps_per_sample = steps_per_sample
                else:
                    gs.logger.warning(
                        "Data collection hz is set, but step_dt is not provided. "
                        "Using default steps_per_sample=1 (sample every step)."
                    )
                    options._steps_per_sample = 1

            options.handler.initialize()

            thread = threading.Thread(target=self.make_process_callback(idx))
            thread.start()
            self._processor_threads.append(thread)

    def pause_recording(self):
        """Pause data recording. Resume with `start_recording()`."""
        if not self._is_recording:
            gs.logger.warning("Ignoring pause_recording(): data recording is not active.")
            return
        self._is_paused = True

    def stop_recording(self):
        """Stop and complete data recording."""
        if not self._is_recording:
            gs.logger.warning("Ignoring stop_recording(): data recording is not active.")
            return
        self._is_recording = False
        for i, opt in enumerate(self._rec_options):
            opt.handler.cleanup()
            self._processor_threads[i].join()

    def step(self):
        """
        Increment the step count and process sensor data if recording is active.
        Each sensor is read every n steps based on its associated RecordingOptions.
        """
        self._step += 1
        if not self._is_recording or self._is_paused:
            return
        for opt in self._rec_options:
            if self._step % opt._steps_per_sample == 0:
                self.read_sensor(opt.sensor_idx)

    def read_sensor(self, rec_idx=0):
        """
        Read and process data from the sensor.
        Usually called by step(), but can also be called manually to process data immediately.
        """
        options = self._rec_options[rec_idx]
        data = self._sensors[options.sensor_idx].read()
        data_queue = self._data_queues[rec_idx]
        try:
            data_queue.put(data, block=True, timeout=options.buffer_full_wait_time)
        except queue.Full:
            gs.logger.warning("Data queue is full, dropping oldest data sample.")
            data_queue.get_nowait()
            data_queue.put_nowait(data)

    def make_process_callback(self, rec_idx):
        def _process_data():
            """Background thread that processes and outputs data"""
            while self._is_recording or not self._data_queues[rec_idx].empty():
                try:
                    data = self._data_queues[rec_idx].get(timeout=1.0)
                    self._rec_options[rec_idx].handler.process(data)
                    self._data_queues[rec_idx].task_done()
                except queue.Empty:
                    continue

        return _process_data

    @property
    def is_active(self):
        """Check if in active recording (is_recording and not is_paused)"""
        return self._is_recording and not self._is_paused

    @property
    def is_recording(self):
        return self._is_recording

    @property
    def is_paused(self):
        return self._is_paused

    @property
    def sensors(self):
        return self._sensors
