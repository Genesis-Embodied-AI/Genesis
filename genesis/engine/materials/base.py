import taichi as ti

import genesis as gs
from genesis.repr_base import RBC


@ti.data_oriented
class Material(RBC):
    def __init__(self):
        self._uid = gs.UID()

    @property
    def uid(self):
        return self._uid

    @classmethod
    def _repr_type(cls):
        return f"<gs.{cls.__module__.split('.')[-2]}.{cls.__name__}>"
