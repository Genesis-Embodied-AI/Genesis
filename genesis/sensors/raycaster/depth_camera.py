import torch

from genesis.sensors.sensor_manager import register_sensor

from .patterns import DepthCameraPattern
from .raycaster import RaycasterData, RaycasterOptions, RaycasterSensor, RaycasterSharedMetadata


class DepthCameraOptions(RaycasterOptions):
    """
    Depth camera that uses ray casting to obtain depth images.

    Parameters
    ----------
    pattern: DepthCameraPattern
        The raycasting pattern configuration for the sensor.
    """

    pattern: DepthCameraPattern


@register_sensor(DepthCameraOptions, RaycasterSharedMetadata, RaycasterData)
class DepthCameraSensor(RaycasterSensor):

    def build(self):
        super().build()
        self._shape = ((self._manager._sim._B,) if self._manager._sim.n_envs > 0 else ()) + (
            self._options.pattern.height,
            self._options.pattern.width,
        )

    def read_image(self) -> torch.Tensor:
        """
        Read the depth image from the sensor.

        This method uses the hit distances from the underlying RaycasterSensor.read() method and reshapes into image.

        Returns
        -------
        torch.Tensor
            The depth image with shape (height, width).
        """
        return self.read().distances.reshape(*self._shape)
