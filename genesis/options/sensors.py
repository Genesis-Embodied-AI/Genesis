from genesis.options import Options


class SensorOptions(Options):
    """
    Base class for all sensor options.
    Each sensor should have their own options class that inherits from this class.
    The options class should be registered with the SensorManager using the @register_sensor decorator.
    """

    pass
