import queue
import threading

import genesis as gs
from genesis.options.recording import RecordingOptions


class DataRecorder:
    """
    Automatically manage multiple data recordings. Each recording has its own data source, handler, and other options.

    Parameters
    ----------
    step_dt: float
        The simulation time step.
    """

    def __init__(self, step_dt: float):
        self._step_dt = step_dt

        self._recordings: list[RecordingOptions] = []
        self._data_queues: list[queue.Queue] = []
        self._processor_threads: list[threading.Thread] = []
        self._is_recording = False

    def add_recording(self, recording_options: RecordingOptions):
        """
        Add a recording configuration to the data recorder.

        Parameters
        ----------
        recording_options: RecordingOptions
            The complete recording configuration including data_func, handler, and other recording parameters.
        """
        if recording_options.hz:
            steps_per_sample_float = 1.0 / (recording_options.hz * self._step_dt)
            steps_per_sample = max(1, round(steps_per_sample_float))
            if abs(steps_per_sample_float - steps_per_sample) > 1e-6:
                gs.logger.warning(
                    f"Data collection hz={recording_options.hz} is not an integer multiple of step size of step dt. "
                    f"Using hz={1.0 / steps_per_sample / self._step_dt} instead."
                )
            recording_options._steps_per_sample = steps_per_sample
        else:
            recording_options._steps_per_sample = 1

        self._recordings.append(recording_options)

    def start(self):
        """Start data recording."""
        self._is_recording = True

        self._data_queues.clear()
        self._processor_threads.clear()

        for recording_idx, recording in enumerate(self._recordings):
            if recording.run_in_thread:
                data_queue = queue.Queue(maxsize=recording.buffer_size)
                self._data_queues.append(data_queue)
            else:
                self._data_queues.append(None)

            recording.handler.initialize()

            if recording.run_in_thread:
                thread = threading.Thread(target=self._make_process_callback(recording_idx))
                thread.start()
                self._processor_threads.append(thread)
            else:
                self._processor_threads.append(None)

    def stop(self):
        """Stop and complete data recording."""
        if not self._is_recording:
            gs.logger.warning("Ignoring stop(): data recording is not active.")
            return
        self._is_recording = False

        for recording_idx, recording in enumerate(self._recordings):
            if recording.run_in_thread and self._processor_threads[recording_idx]:
                self._processor_threads[recording_idx].join()
            recording.handler.cleanup()

    def step(self, global_step: int, global_time: float):
        """
        Increment the step count and process data from each recording configuration.
        In threaded mode, data is put in queues. In non-threaded mode, data is processed synchronously.
        """
        if not self._is_recording:
            return

        for recording_idx, recording in enumerate(self._recordings):
            if global_step % recording._steps_per_sample == 0:
                data = recording.data_func()

                if recording.run_in_thread:
                    data_and_time = (data, global_time)
                    data_queue = self._data_queues[recording_idx]
                    if data_queue is not None:
                        try:
                            data_queue.put(data_and_time, block=False)
                        except queue.Full:
                            try:
                                data_queue.put(data_and_time, block=True, timeout=recording.buffer_full_wait_time)
                            except queue.Full:
                                gs.logger.debug("Data queue is full, dropping oldest data sample.")
                                try:
                                    data_queue.get_nowait()
                                    data_queue.put_nowait(data_and_time)
                                except queue.Empty:
                                    # Queue became empty between operations, just put the data
                                    data_queue.put_nowait(data_and_time)
                else:
                    # Non-threaded mode: process data synchronously
                    recording.handler.process(data, global_time)

    def _make_process_callback(self, recording_idx: int):
        """Create a callback function for processing data in a background thread."""

        def _process_data():
            """Background thread that processes and outputs data"""
            recording = self._recordings[recording_idx]
            data_queue = self._data_queues[recording_idx]

            if data_queue is None:
                return

            while self._is_recording or not data_queue.empty():
                try:
                    data, timestamp = data_queue.get(timeout=1.0)
                    recording.handler.process(data, timestamp)
                    data_queue.task_done()
                except queue.Empty:
                    continue

        return _process_data
