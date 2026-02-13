from pathlib import Path
from typing import TYPE_CHECKING

import quadrants as ti

import genesis as gs
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


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
        name: str | None = None,
    ):
        uid = gs.UID()
        while any(entity.uid.match(uid, short_only=True) for entity in scene.entities):
            uid = gs.UID()
        self._uid = uid
        self._idx = idx
        self._scene: "Scene" = scene
        self._solver = solver
        self._material = material
        self._morph = morph
        self._surface = surface
        self._sim = scene.sim

        # Set entity name (auto-generate if not provided)
        existing_names = {entity.name for entity in scene.entities if entity.name is not None}
        if name is not None:
            if name in existing_names:
                gs.raise_exception(f"Entity name '{name}' already exists in scene.")
            self._name = name
        else:
            morph_name = self._get_morph_identifier()
            self._name = f"{morph_name}_{uid.short()}"

        gs.logger.info(
            f"Adding ~<{self._repr_type()}>~. idx: ~<{self._idx}>~, uid: ~~~<{self._uid}>~~~, morph: ~<{morph}>~, material: ~<{self._material}>~."
        )

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
    def is_built(self):
        return self._solver._scene._is_built

    @property
    def name(self) -> str:
        """
        The name of this entity.

        Returns
        -------
        str
            The entity's name. If a user-specified name was provided during creation,
            that name is returned. Otherwise, an auto-generated name based on the
            morph type and UID is returned.
        """
        return self._name

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        """
        Get the identifier string from the morph for name generation.

        Must be overridden in subclasses to provide type-specific identifiers.
        """
        raise NotImplementedError
