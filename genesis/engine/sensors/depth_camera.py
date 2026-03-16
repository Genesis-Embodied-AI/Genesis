import torch

from genesis.options.sensors import DepthCamera as DepthCameraOptions

from .base_sensor import Sensor
from .raycaster import RaycasterData, RaycasterSensor, RaycasterSharedMetadata


class DepthCameraSensor(RaycasterSensor, Sensor[DepthCameraOptions, RaycasterSharedMetadata, RaycasterData]):
    def build(self):
        super().build()

        batch_shape = (self._manager._sim._B,) if self._manager._sim.n_envs > 0 else ()
        self._shape = (*batch_shape, self._options.pattern.height, self._options.pattern.width)

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
