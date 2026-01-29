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
    ):
        self._uid = gs.UID()
        self._idx = idx
        self._scene: "Scene" = scene
        self._solver = solver
        self._material = material
        self._morph = morph
        self._surface = surface
        self._sim = scene.sim
        self._name: str | None = None  # Set by _set_name() after creation

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

        Raises an exception if the name already exists in the scene.
        """
        if user_name is not None:
            # Validate uniqueness for user-specified names upfront
            if user_name in scene._entity_name_registry:
                gs.raise_exception(f"Entity name '{user_name}' already exists in scene.")
            self._name = user_name
        else:
            self._name = self._generate_name()
            # Validate uniqueness for auto-generated names
            if self._name in scene._entity_name_registry:
                gs.raise_exception(f"Entity name '{self._name}' already exists in scene.")

        scene._entity_name_registry[self._name] = self

    def _generate_name(self) -> str:
        """Generate a default name based on morph type and UID."""
        uid_suffix = str(self._uid)[:8]
        morph_name = self._get_morph_identifier()
        return f"{morph_name}_{uid_suffix}"

    def _get_morph_identifier(self) -> str:
        """
        Get the identifier string from the morph for name generation.

        Must be overridden in subclasses to provide type-specific identifiers.
        """
        raise NotImplementedError
