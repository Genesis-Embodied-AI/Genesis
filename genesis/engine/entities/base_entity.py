import taichi as ti

import genesis as gs
from genesis.repr_base import RBC


@ti.data_oriented
class Entity(RBC):
    """
    Base class for all types of entities.
    """

    def __init__(
        self,
        idx,
        scene,
        morph,
        solver,
        material,
        surface,
    ):
        self._uid = gs.UID()
        self._idx = idx
        self._scene = scene
        self._solver = solver
        self._material = material
        self._morph = morph
        self._surface = surface
        self._sensors = []
        self._sim = scene.sim

        gs.logger.info(
            f"Adding ~<{self._repr_type()}>~. idx: ~<{self._idx}>~, uid: ~~~<{self._uid}>~~~, morph: ~<{morph}>~, material: ~<{self._material}>~."
        )

    def add_sensor(self, sensor_type, **sensor_kwargs):
        """
        Add a sensor to the entity.

        Args:
            sensor_type: The type of the sensor to add.
            **sensor_kwargs: Additional keyword arguments for the sensor.

        Returns:
            The created sensor instance.
        """
        sensor = sensor_type(self, **sensor_kwargs)
        self._sensors.append(sensor)
        return sensor

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        return self._uid

    @property
    def idx(self):
        return self._idx

    @property
    def scene(self):
        return self._scene

    @property
    def sim(self):
        return self._sim

    @property
    def solver(self):
        return self._solver

    @property
    def surface(self):
        return self._surface

    @property
    def morph(self):
        return self._morph

    @property
    def material(self):
        return self._material

    @property
    def sensors(self):
        return self._sensors

    @property
    def is_built(self):
        return self._scene._is_built
