import queue
import threading
from typing import Callable, Generic, TypeVar

import genesis as gs
from genesis.options import Options

from .recorder_manager import RecorderManager

T = TypeVar("T")


class RecorderOptions(Options):
    """
    Options for recording simulation data by automatically sampling data from a data source, e.g. a sensor.

    Parameters
    ----------
    hz: float, optional
        The frequency at which to sample data, in Hz (samples per second).
        If None, the data will be sampled every step.
    buffer_size: int, optional
        Applicable when run_in_thread is True. The size of the data queue buffer.
        Defaults to 0, which means infinite size.
    buffer_full_wait_time: float, optional
        Applicable when run_in_thread is True. The time to wait for buffer space to become available when the
        buffer is full. Defaults to 0.1 seconds.
    """

    hz: float | None = None
    buffer_size: int = 0
    buffer_full_wait_time: float = 0.1

    def validate(self):
        """Validate the recorder options values before the recorder is added to the scene."""
        if self.hz is not None and self.hz < gs.EPS:
            gs.raise_exception(f"[{type(self).__name__}] recording hz should be greater than 0.")
        if self.buffer_size < 0:
            gs.raise_exception(f"[{type(self).__name__}] buffer size should be 0 (infinite size) or greater.")
        if self.buffer_full_wait_time < gs.EPS:
            gs.raise_exception(f"[{type(self).__name__}] buffer full wait time should be greater than 0.")


class Recorder(Generic[T]):
    """
    Base class for all recorders.

    Note that modifying the signature of this class in recorder implementations should be avoided since instantiation is
    done through the RecorderManager.
    """

    def __init__(self, manager: RecorderManager, options: RecorderOptions, data_func: Callable[[], T]):
        self._options = options
        self._manager = manager
        self._data_func = data_func
        self._steps_per_sample = 1
        self._is_built = False
        self._is_recording = False

        self._data_queue: queue.Queue | None = None
        self._processor_thread: threading.Thread | None = None

        if options.hz:
            steps_per_sample_float = 1.0 / (options.hz * manager._step_dt)
            steps_per_sample = max(1, round(steps_per_sample_float))
            if abs(steps_per_sample_float - steps_per_sample) > gs.EPS:
                gs.logger.warning(
                    f"[Recorder] hz={options.hz} is not an integer multiple of step size of step dt. "
                    f"Using hz={1.0 / steps_per_sample / manager._step_dt} instead."
                )
            self._steps_per_sample = steps_per_sample

    # =============================== methods to implement ===============================

    @gs.assert_unbuilt
    def build(self):
        """
        Build the recorder, e.g. by initializing variables and creating widgets or file handles.
        """
        self._is_built = True

    @gs.assert_built
    def process(self, data, cur_time):
        """
        Process each incoming data sample.

        Parameters
        ----------
        data: Any
            The data to be processed.
        cur_time: float
            The current time of the simulation.
        """
        raise NotImplementedError(f"[{type(self).__name__}] process() is not implemented.")

    @gs.assert_built
    def cleanup(self):
        """
        Cleanup all resources, e.g. by closing widgets or files.

        This method is called when recording is stopped by `scene.stop_recording()`.
        """
        raise NotImplementedError(f"[{type(self).__name__}] cleanup() is not implemented.")

    @gs.assert_built
    def reset(self, envs_idx=None):
        """
        Reset the recorder, e.g. by flushing stored data.

        This method is called when the scene is reset by `scene.reset()`.

        Parameters
        ----------
        envs_idx: array_like, optional
            The indices of the environments to reset. If None, all environments are reset.
        """
        raise NotImplementedError(f"[{type(self).__name__}] reset() is not implemented.")

    @property
    def run_in_thread(self) -> bool:
        """
        Whether to run the recorder in a background thread.

        Running in a background thread allows for processing data without blocking the main thread, so this is
        encouraged for most recorders (simply `return True`), but implementers should check that the recorder is
        thread-safe on all devices (threading on macOS tends to be less supported).
        """
        raise NotImplementedError(f"[{type(self).__name__}] run_in_thread is not implemented.")

    # =============================== recording  ===============================

    def _process_data_loop(self):
        """Background thread that processes and outputs data."""
        if self._data_queue is None:
            return

        while self._is_recording or not self._data_queue.empty():
            try:
                data, timestamp = self._data_queue.get(timeout=1.0)
                self.process(data, timestamp)
                self._data_queue.task_done()
            except queue.Empty:
                continue

    @gs.assert_built
    def start(self):
        """Start the recording thread if run_in_thread is True."""
        self._is_recording = True

        if self.run_in_thread:
            self._data_queue = queue.Queue(maxsize=self._options.buffer_size)
            self._processor_thread = threading.Thread(target=self._process_data_loop)
            self._processor_thread.start()

    @gs.assert_built
    def stop(self):
        """Stop the recording thread and cleanup resources."""
        if self._is_recording:
            self._is_recording = False
            self.sync()
            self.cleanup()

    @gs.assert_built
    def sync(self):
        """Wait for the processor thread to finish."""
        if self.run_in_thread and self._processor_thread is not None:
            self._processor_thread.join()
            self._processor_thread = None
            self._data_queue = None

    @gs.assert_built
    def step(self, global_step: int):
        """Process a simulation step, potentially recording data."""
        if not self._is_recording:
            return

        if global_step % self._steps_per_sample != 0:
            return

        global_time = global_step * self._manager._step_dt
        data = self._data_func()

        if not self.run_in_thread:
            # non-threaded mode: process data synchronously
            self.process(data, global_time)
            return

        # threaded mode: put data in queue
        try:
            self._data_queue.put((data, global_time), block=False)
            return
        except queue.Full:
            try:
                self._data_queue.put((data, global_time), timeout=self._options.buffer_full_wait_time)
                return
            except queue.Full:
                pass

        gs.logger.debug("[Recorder] Data queue is full, dropping oldest data sample.")
        try:
            self._data_queue.get_nowait()
        except queue.Empty:
            # Queue became empty between operations, just put the data
            pass
        finally:
            self._data_queue.put_nowait((data, global_time))

    @property
    def is_built(self) -> bool:
        return self._is_built
