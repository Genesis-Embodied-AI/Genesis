from genesis.options import Options


class SensorOptions(Options):
    """
    Base class for all sensor options.
    Each sensor should have their own options class that inherits from this class.
    The options class should be registered with the SensorManager using the @register_sensor decorator.

    Parameters
    ----------
    read_delay : float
        The delay in seconds before the sensor data is read.
    """

    read_delay: float = 0.0

    def validate(self, scene):
        """
        Validate the sensor options values before the sensor is added to the scene.
        """
        pass
