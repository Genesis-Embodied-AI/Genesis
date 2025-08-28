from typing import TYPE_CHECKING, Callable

from .options import Options

if TYPE_CHECKING:
    from genesis.sensors.data_handlers import DataHandler


class RecordingOptions(Options):
    """
    Options for recording data from a data source.

    Parameters
    ----------
    data_func: Callable
        A function with no arguments that returns the data to be recorded.
    handler: DataHandler
        The data handler that will process the recorded data.
    hz: float, optional
        The frequency at which to sample data, in Hz (samples per second).
        If None, the data will be sampled every step.
    run_in_thread: bool, optional
        Whether to process data in a separate thread. Defaults to True.
    buffer_size: int, optional
        Applicable when run_in_thread is True. The size of the data queue buffer.
        Defaults to 0, which means infinite size.
    buffer_full_wait_time: float, optional
        Applicable when run_in_thread is True. The time to wait for buffer space to become available when the
        buffer is full. Defaults to 0.1 seconds.
    """

    data_func: Callable | None = None  # should never actually be None, but allow for setting a custom default
    handler: "DataHandler"
    hz: float | None = None
    buffer_size: int = 0
    buffer_full_wait_time: float = 0.1
    run_in_thread: bool = True

    _steps_per_sample: int = 1  # how often to sample data, calculated based on hz if given
