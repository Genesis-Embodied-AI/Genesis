from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import StrictBool

import genesis as gs
from genesis.options.options import Options
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity
    from genesis.engine.scene import Scene

EntityT = TypeVar("EntityT", bound="Entity")


class Material(Options, Generic[EntityT]):
    """
    The base class of materials.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    use_visual_raycasting: StrictBool = False


MaterialT = TypeVar("MaterialT", bound=Material)


class MaterialHandle(RBC, Generic[MaterialT]):
    """
    The base class of the materials registered on a scene through 'Scene.add_material'.

    A handle is the identity of one material: entities built from the same handle are known to share it, which is the
    granularity parameters resolved between two materials are keyed on. Each family subclasses this to expose the
    options it was registered from.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    def __init__(self, scene: "Scene", idx: int, options: MaterialT):
        self._scene: "Scene" = scene
        self._idx: int = idx
        self._options: MaterialT = options
        self._uid = gs.UID()

    def _repr_brief(self):
        return f"{self.__repr_name__()}: idx={self._idx}"

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        Get the unique ID of the material.
        """
        return self._uid

    @property
    def idx(self) -> int:
        """
        Get the index of the material in the scene, in registration order.
        """
        return self._idx

    @property
    def options(self) -> MaterialT:
        """
        Get the options this material was registered from.
        """
        return self._options
