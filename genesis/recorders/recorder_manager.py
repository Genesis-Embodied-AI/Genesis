import queue
import threading
from typing import TYPE_CHECKING, Callable, Type

import genesis as gs

if TYPE_CHECKING:
    from .base_recorder import Recorder, RecorderOptions


class RecorderManager:
    """
    Manage the creation, processing, and cleanup of all data recorders.

    Parameters
    ----------
    step_dt: float
        The simulation time step.
    """

    RECORDER_TYPES_MAP = {}

    def __init__(self, step_dt: float):
        self._step_dt = step_dt

        self._data_funcs: list[Callable] = []
        self._recorders: list["Recorder"] = []
        self._data_queues: list[queue.Queue] = []
        self._processor_threads: list[threading.Thread] = []
        self._is_recording = False
        self._is_built = False

    @gs.assert_unbuilt
    def add_recorder(self, data_func: Callable, rec_options: "RecorderOptions"):
        """
        Automatically read and process data. See RecorderOptions for more details.

        Parameters
        ----------
        data_func: Callable
            A function with no arguments that returns the data to be recorded.
        rec_options: RecorderOptions
            The options for the recorder which determines how the data is recorded and processed.
        """
        self._data_funcs.append(data_func)
        rec_options.validate()
        self._recorders.append(RecorderManager.RECORDER_TYPES_MAP[type(rec_options)](self, rec_options))

    @gs.assert_unbuilt
    def build(self):
        """Start data recording."""
        for recorder in self._recorders:
            recorder.build()
            recorder._is_built = True
        self._start_threads()
        self._is_built = True

    def _start_threads(self):
        self._is_recording = True  # needed to enter loop in thread

        self._data_queues.clear()
        self._processor_threads.clear()

        for rec_idx, recorder in enumerate(self._recorders):
            if recorder.run_in_thread:
                data_queue = queue.Queue(maxsize=recorder._options.buffer_size)
                self._data_queues.append(data_queue)
                thread = threading.Thread(target=self._make_process_callback(rec_idx))
                thread.start()
                self._processor_threads.append(thread)
            else:
                self._data_queues.append(None)
                self._processor_threads.append(None)

    @gs.assert_built
    def stop(self):
        """Stop and complete data recording."""
        if not self._is_recording:
            gs.logger.warning("[DataRecorder] Ignoring stop(): data recording is not active.")
        else:
            self._stop()

    def _stop(self):
        self._join_threads()
        for recorder in self._recorders:
            recorder.cleanup()

        self._clear()

    def _join_threads(self):
        self._is_recording = False  # needed to exit loop in thread
        for thread, recorder in zip(self._processor_threads, self._recorders):
            if recorder.run_in_thread and thread is not None:
                thread.join()

    @gs.assert_built
    def reset(self, envs_idx=None):
        self._join_threads()  # finish processing data before the reset
        for recorder in self._recorders:
            recorder.reset(envs_idx)
        self._start_threads()

    def _clear(self):
        self._data_funcs.clear()
        self._recorders.clear()
        self._data_queues.clear()
        self._processor_threads.clear()

    def destroy(self):
        self._clear()

    @gs.assert_built
    def step(self, global_step: int):
        """
        Increment the step count and process data from each recording configuration.
        In threaded mode, data is put in queues. In non-threaded mode, data is processed synchronously.
        """
        if not self._is_recording:
            return

        global_time = global_step * self._step_dt

        for data_func, data_queue, recorder in zip(self._data_funcs, self._data_queues, self._recorders):
            if global_step % recorder._steps_per_sample == 0:
                data = data_func()

                if recorder.run_in_thread:
                    data_and_time = (data, global_time)
                    try:
                        data_queue.put(data_and_time, block=False)
                    except queue.Full:
                        try:
                            data_queue.put(data_and_time, block=True, timeout=recorder._options.buffer_full_wait_time)
                        except queue.Full:
                            gs.logger.debug("[DataRecorder] Data queue is full, dropping oldest data sample.")
                            try:
                                data_queue.get_nowait()
                            except queue.Empty:
                                pass  # Queue became empty between operations, just put the data
                            finally:
                                data_queue.put_nowait(data_and_time)
                else:
                    # non-threaded mode: process data synchronously
                    recorder.process(data, global_time)

    def _make_process_callback(self, rec_idx: int):
        """Create a callback function for processing data in a background thread."""

        def _process_data():
            """Background thread that processes and outputs data"""

            recorder = self._recorders[rec_idx]
            data_queue = self._data_queues[rec_idx]

            if data_queue is None:
                return

            while self._is_recording or not data_queue.empty():
                try:
                    data, timestamp = data_queue.get(timeout=1.0)
                    recorder.process(data, timestamp)
                    data_queue.task_done()
                except queue.Empty:
                    continue

        return _process_data

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def is_built(self) -> bool:
        return self._is_built


def register_recording(options_cls: "RecorderOptions"):
    def _impl(recorder_cls: Type["Recorder"]):
        RecorderManager.RECORDER_TYPES_MAP[options_cls] = recorder_cls
        return recorder_cls

    return _impl
