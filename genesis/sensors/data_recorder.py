import queue
import threading

import genesis as gs
from genesis.options.recording import RecordingOptions

from .base_sensor import Sensor
from .data_handlers import DataHandler


class SensorDataRecorder:
    """
    Utility class to automatically collect data from sensors and process it using specified handlers.
    Each sensor added to this recorder will be sampled depending on the specified recording options,
    and processed in a background thread.

    Parameters
    ----------
    sensors: gs.sensors.Sensor
        The sensors to record data from.
    rec_options: list[RecordingOptions], optional
        The recording options for each sensor, specifying how to handle the recorded data.
    step_dt: float | None, optional
        The time step for the simulation.
        If provided, it is used to calculate the steps per sample for each sensor based on the hz in RecordingOptions.
        If None, hz will be ignored and the sensor will be sampled every step.
    """

    def __init__(
        self,
        step_dt: float | None = None,
    ):
        self._sensors: list[Sensor] = []
        self._rec_options: list[RecordingOptions] = []
        self.step_dt = step_dt

        self._data_queues: list[queue.Queue] = []
        self._processor_threads: list[threading.Thread] = []
        self._is_recording = False
        self._is_paused = False
        self._step = 0

    def add_sensor(
        self, sensor: Sensor, options: DataHandler | list[DataHandler] | RecordingOptions | list[RecordingOptions]
    ):
        """
        Add a sensor to the data collector with specified recording options.

        Parameters
        ----------
        sensor: Sensor
            The sensor to record.
        options: DataHandler | list[DataHandler] | RecordingOptions | list[RecordingOptions]
            How the sensor data should be recorded.
            Use RecordingOptions to specify

            The handler(s) that will process the recorded data. Can be a single handler or a list of handlers.
            If RecordingOptions is provided, it should contain the handler and optionally other parameters like hz.
        """
        try:
            if isinstance(options, DataHandler):
                opt = RecordingOptions(handler=options)
                opt._sensor_idx = len(self._sensors)
                _options = [opt]
            elif isinstance(options, RecordingOptions):
                _options = [options]
            else:
                _options = []
                for opt in options:  # may raise TypeError if not iterable
                    if isinstance(opt, DataHandler):
                        opt = RecordingOptions(handler=opt)
                    elif not isinstance(opt, RecordingOptions):
                        raise TypeError()

                    opt._sensor_idx = len(self._sensors)
                    _options.append(opt)
        except TypeError as e:
            gs.raise_exception_from(
                "SensorDataRecorder.add_sensor(...) options must be a DataHandler or RecordingOptions.", e
            )

        self._sensors.append(sensor)
        self._rec_options.extend(_options)

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
                self.read_sensor(opt._sensor_idx)

    def read_sensor(self, rec_idx=0):
        """
        Read and process data from the sensor.
        Usually called by step(), but can also be called manually to process data immediately.
        """
        options = self._rec_options[rec_idx]
        data = self._sensors[options._sensor_idx].read()
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
