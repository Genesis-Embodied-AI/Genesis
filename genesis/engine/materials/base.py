from typing import TYPE_CHECKING, Generic, TypeVar

import genesis as gs
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity

EntityT = TypeVar("EntityT", bound="Entity")


class Material(RBC, Generic[EntityT]):
    """
    The base class of materials.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    def __init__(self):
        self._uid = gs.UID()

    @property
    def uid(self):
        return self._uid
