from pathlib import Path
from typing import TYPE_CHECKING

import gstaichi as ti

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
        self._uid = gs.UID()
        self._idx = idx
        self._scene: "Scene" = scene
        self._solver = solver
        self._material = material
        self._morph = morph
        self._surface = surface
        self._sim = scene.sim

        # Set entity name (auto-generate if not provided)
        self._set_name(name, scene)

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

    def _set_name(self, user_name: str | None, scene: "Scene") -> None:
        """
        Set the entity's name, auto-generating if not provided.

        Raises an exception if a user-specified name already exists in the scene.
        For auto-generated names, regenerates UID until unique.
        """
        existing_names = {entity.name for entity in scene.entities if entity.name is not None}

        if user_name is not None:
            # Validate uniqueness for user-specified names upfront
            if user_name in existing_names:
                gs.raise_exception(f"Entity name '{user_name}' already exists in scene.")
            self._name = user_name
        else:
            # Generate name, regenerating UID if collision occurs
            morph_name = self._get_morph_identifier()
            while True:
                self._name = f"{morph_name}_{self._uid.short()}"
                if self._name not in existing_names:
                    break
                # Redraw UID if name collision
                self._uid = gs.UID()

    def _get_morph_identifier(self) -> str:
        """
        Get the identifier string from the morph for name generation.

        Must be overridden in subclasses to provide type-specific identifiers.
        """
        raise NotImplementedError
