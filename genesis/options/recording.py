from typing import Callable

from .options import Options


class RecordingOptions(Options):
    """
    Options for recording data from a sensor.

    Parameters
    ----------
    preprocess_func: Callable | None
        A function that preprocesses the data before it is processed by the handler.
        preprocess_func should take in two arguments (measured_data, ground_truth_data).
        By default, only measured data from `sensor.read()` is given to the handler.
    hz: float, optional
        The frequency at which to sample data, in Hz (samples per second).
        If None, the sensor will be sampled every step.
    buffer_size: int
        The size of the data queue buffer. Defaults to 0, which means infinite size.
    buffer_full_wait_time: float
        The time to wait for buffer space to become available when the buffer is full. Defaults to 0.1 seconds.
    """

    preprocess_func: Callable | None = None
    hz: float | None = None
    buffer_size: int = 0
    buffer_full_wait_time: float = 0.1

    _steps_per_sample: int = 1  # how often to sample data, calculated based on hz if given
