from genesis.sensors.data_handlers import DataHandler

from .options import Options


class RecordingOptions(Options):
    """
    Options for recording data from a sensor.

    Parameters
    ----------
    handler: DataHandler
        The handler that will process the recorded data.
    hz: float, optional
        The frequency at which to sample data, in Hz (samples per second).
        If None, the sensor will be sampled every step.
    buffer_size: int
        The size of the data queue buffer. Defaults to 0, which means infinite size.
    buffer_full_wait_time: float
        The time to wait for buffer space to become available when the buffer is full. Defaults to 0.1 seconds.
    """

    handler: DataHandler
    hz: float | None = None
    buffer_size: int = 0
    buffer_full_wait_time: float = 0.1

    _sensor_idx: int = -1  # index of sensor in SensorDataRecorder.sensors list; handled by add_sensor()
    _steps_per_sample: int = 1  # how often to sample data, calculated based on hz if given
