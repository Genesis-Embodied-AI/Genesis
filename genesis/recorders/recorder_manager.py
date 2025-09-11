from typing import TYPE_CHECKING, Any, Callable, Type

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

        self._recorders: list["Recorder"] = []
        self._is_recording = False
        self._is_built = False

    @gs.assert_unbuilt
    def add_recorder(self, data_func: Callable[[], Any], rec_options: "RecorderOptions"):
        """
        Automatically read and process data. See RecorderOptions for more details.

        Parameters
        ----------
        data_func: Callable[[], Any]
            A function with no arguments that returns the data to be recorded.
        rec_options: RecorderOptions
            The options for the recorder which determines how the data is recorded and processed.
        """
        rec_options.validate()
        recorder_cls = RecorderManager.RECORDER_TYPES_MAP[type(rec_options)]
        self._recorders.append(recorder_cls(self, rec_options, data_func))

    @gs.assert_unbuilt
    def build(self):
        """Start data recording."""
        for recorder in self._recorders:
            recorder.build()
            recorder.start()
        self._is_recording = True
        self._is_built = True

    @gs.assert_built
    def stop(self):
        """Stop and complete data recording."""
        if not self._is_recording:
            gs.logger.warning("[DataRecorder] Ignoring stop(): data recording is not active.")
        else:
            self._is_recording = False
            for recorder in self._recorders:
                recorder.stop()
            self._recorders.clear()

    @gs.assert_built
    def reset(self, envs_idx=None):
        for recorder in self._recorders:
            recorder.sync()
            recorder.reset(envs_idx)
            recorder.start()

    def destroy(self):
        self._recorders.clear()

    @gs.assert_built
    def step(self, global_step: int):
        """
        Increment the step count and process data from each recording configuration.
        In threaded mode, data is put in queues. In non-threaded mode, data is processed synchronously.
        """
        if not self._is_recording:
            return

        for recorder in self._recorders:
            recorder.step(global_step)

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
