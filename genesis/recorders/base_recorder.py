import genesis as gs
from genesis.options import Options

from .recorder_manager import RecorderManager


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
        if self.hz is not None and self.hz <= 0:
            gs.raise_exception(f"[{type(self).__name__}] recording hz should be greater than 0.")
        if self.buffer_size < 0:
            gs.raise_exception(f"[{type(self).__name__}] buffer size should be 0 (infinite size) or greater.")
        if self.buffer_full_wait_time <= 0:
            gs.raise_exception(f"[{type(self).__name__}] buffer full wait time should be greater than 0.")


class Recorder:
    """Base class for all recorders."""

    def __init__(self, manager: RecorderManager, options: RecorderOptions):
        self._options = options
        self._manager = manager
        self._steps_per_sample = 1
        self._initialized = False
        if options.hz:
            steps_per_sample_float = 1.0 / (options.hz * manager._step_dt)
            steps_per_sample = max(1, round(steps_per_sample_float))
            if abs(steps_per_sample_float - steps_per_sample) > 1e-6:
                gs.logger.warning(
                    f"[DataRecorder] hz={options.hz} is not an integer multiple of step size of step dt. "
                    f"Using hz={1.0 / steps_per_sample / manager._step_dt} instead."
                )
            self._steps_per_sample = steps_per_sample

    def initialize(self):
        raise NotImplementedError(f"[{type(self).__name__}] initialize() is not implemented.")

    def process(self, data, cur_time):
        raise NotImplementedError(f"[{type(self).__name__}] process() is not implemented.")

    def cleanup(self):
        raise NotImplementedError(f"[{type(self).__name__}] cleanup() is not implemented.")

    def reset(self, envs_idx=None):
        raise NotImplementedError(f"[{type(self).__name__}] reset() is not implemented.")

    @property
    def run_in_thread(self) -> bool:
        raise NotImplementedError(f"[{type(self).__name__}] run_in_thread is not implemented.")

    @property
    def initialized(self) -> bool:
        return self._initialized
