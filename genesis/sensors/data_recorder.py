import queue
import threading

import genesis as gs
from genesis.options.recording import RecordingOptions
from genesis.sensors.data_handlers import DataHandler


class SensorDataRecorder:
    """
    Utility class to automatically collect data from a single sensor and process it using specified handlers.
    Each handler can have different sampling frequencies and buffer sizes.

    Parameters
    ----------
    sensor: Sensor
        The sensor to record data from.
    step_dt: float
        The simulation time step.
    """

    def __init__(self, sensor, step_dt: float):
        self._sensor = sensor
        self._step_dt = step_dt

        self._handlers: list[DataHandler] = []
        self._options: list[RecordingOptions] = []
        self._data_queues: list[queue.Queue] = []
        self._processor_threads: list[threading.Thread] = []
        self._is_recording = False
        self._is_paused = False

    def add_handler(self, handler: DataHandler, rec_options: RecordingOptions | None = None):
        """
        Add a data handler to process the sensor data and start recording.

        Parameters
        ----------
        rec_options: RecordingOptions
            The recording options which includes the data handler and other recording parameters.
        """
        self._handlers.append(handler)

        if rec_options is None:
            rec_options = RecordingOptions()
        elif rec_options.hz:
            steps_per_sample_float = 1.0 / (rec_options.hz * self._step_dt)
            steps_per_sample = max(1, round(steps_per_sample_float))
            if steps_per_sample_float != steps_per_sample:
                gs.logger.warning(
                    f"Data collection hz={rec_options.hz} is not an integer multiple of step size of step dt. "
                    f"Using hz={1.0 / steps_per_sample / self._step_dt} instead."
                )
        self._options.append(rec_options)

    def start(self):
        """Start data recording."""
        self._is_paused = False
        if self._is_recording:
            # Resuming recording after pause, skip initialization
            return

        self._is_recording = True

        self._data_queues.clear()
        self._processor_threads.clear()

        for handler_idx, (handler, options) in enumerate(zip(self._handlers, self._options)):
            data_queue = queue.Queue(maxsize=options.buffer_size)
            self._data_queues.append(data_queue)

            handler.initialize()

            thread = threading.Thread(target=self._make_process_callback(handler_idx))
            thread.start()
            self._processor_threads.append(thread)

    def pause_recording(self):
        """Pause data recording. Resume with `start_recording()`."""
        if not self._is_recording:
            gs.logger.warning("Ignoring pause(): data recording is not active.")
            return
        self._is_paused = True

    def stop(self):
        """Stop and complete data recording."""
        if not self._is_recording:
            gs.logger.warning("Ignoring stop(): data recording is not active.")
            return
        self._is_recording = False
        for handler_idx, handler in enumerate(self._handlers):
            if handler_idx < len(self._processor_threads):
                self._processor_threads[handler_idx].join()
            handler.cleanup()

    def step(self, global_step: int, global_time: float):
        """
        Increment the step count and put sensor data in the data queue if recording is active.
        Each handler is processed every n steps based on its sampling frequency.
        """
        if not self.is_active:
            return

        sensor_measured_data = self._sensor.read()
        sensor_ground_truth_data = self._sensor.read_ground_truth()

        for options, data_queue in zip(self._options, self._data_queues):
            if global_step % options._steps_per_sample == 0:
                if options.preprocess_func is None:
                    sensor_data = sensor_measured_data
                else:
                    sensor_data = options.preprocess_func(sensor_measured_data, sensor_ground_truth_data)
                data = (sensor_data, global_time)

                try:
                    data_queue.put(data, block=False)
                except queue.Full:
                    try:
                        data_queue.put(data, block=True, timeout=options.buffer_full_wait_time)
                    except queue.Full:
                        gs.logger.debug("Data queue is full, dropping oldest data sample.")
                        try:
                            data_queue.get_nowait()
                            data_queue.put_nowait(data)
                        except queue.Empty:
                            # Queue became empty between operations, just put the data
                            data_queue.put_nowait(data)

    def _make_process_callback(self, handler_idx: int):
        """Create a callback function for processing data in a background thread."""

        def _process_data():
            """Background thread that processes and outputs data"""
            handler = self._handlers[handler_idx]
            data_queue = self._data_queues[handler_idx]

            while self._is_recording or not data_queue.empty():
                try:
                    data, timestamp = data_queue.get(timeout=1.0)
                    handler.process(data, timestamp)
                    data_queue.task_done()
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
    def sensor(self):
        return self._sensor
