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
        if isinstance(material, gs.materials.Rigid):
            # small sdf res is sufficient for primitives regardless of size
            if isinstance(morph, gs.morphs.Primitive):
                material._sdf_max_res = 32

        # some morph should not smooth surface normal
        if isinstance(morph, (gs.morphs.Box, gs.morphs.Cylinder, gs.morphs.Terrain)):
            surface.smooth = False

        if isinstance(morph, (gs.morphs.URDF, gs.morphs.MJCF, gs.morphs.Terrain)):
            if not isinstance(material, (gs.materials.Rigid, gs.materials.Avatar, gs.materials.Hybrid)):
                gs.raise_exception(f"Unsupported material for morph: {material} and {morph}.")

        if surface.double_sided is None:
            surface.double_sided = isinstance(material, gs.materials.PBD.Cloth)

        # validate and populate default surface.vis_mode considering morph type
        if isinstance(material, (gs.materials.Rigid, gs.materials.Avatar, gs.materials.Tool)):
            if surface.vis_mode is None:
                surface.vis_mode = "visual"

            if surface.vis_mode not in ("visual", "collision", "sdf"):
                gs.raise_exception(
                    f"Invalid `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Only supporting "
                    "'visual', 'collision' and 'sdf'."
                )
        elif isinstance(
            material,
            (
                gs.materials.PBD.Liquid,
                gs.materials.PBD.Particle,
                gs.materials.MPM.Liquid,
                gs.materials.MPM.Sand,
                gs.materials.MPM.Snow,
                gs.materials.SPH.Liquid,
            ),
        ):
            if surface.vis_mode is None:
                surface.vis_mode = "particle"

            if surface.vis_mode not in ("particle", "recon"):
                gs.raise_exception(
                    f"Invalid `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Only supporting "
                    "'particle' and 'recon'."
                )
        elif isinstance(material, (gs.materials.SF.Smoke)):
            if surface.vis_mode is None:
                surface.vis_mode = "particle"

            if surface.vis_mode not in ("particle",):
                gs.raise_exception(
                    f"Invalid `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Only supporting "
                    "'particle'."
                )
        elif isinstance(material, (gs.materials.PBD.Base, gs.materials.MPM.Base, gs.materials.SPH.Base)):
            if surface.vis_mode is None:
                surface.vis_mode = "visual"

            if surface.vis_mode not in ("visual", "particle", "recon"):
                gs.raise_exception(
                    f"Invalid `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Only supporting "
                    "'visual', 'particle' and 'recon'."
                )
        elif isinstance(material, (gs.materials.FEM.Base)):
            if surface.vis_mode is None:
                surface.vis_mode = "visual"

            if surface.vis_mode not in ("visual",):
                gs.raise_exception(
                    f"Invalid `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Only supporting "
                    "'visual'."
                )
        elif isinstance(material, (gs.materials.Hybrid)):  # determine the visual of the outer soft part
            if surface.vis_mode is None:
                surface.vis_mode = "particle"

            if surface.vis_mode not in ["particle", "visual"]:
                gs.raise_exception(
                    f"Invalid `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Only supporting "
                    "'particle' and 'visual'."
                )
        else:
            gs.raise_exception(f"Material not supported.: {material}")

        # Set material-dependent default options
        if isinstance(morph, gs.morphs.FileMorph):
            # Rigid entities will convexify geom by default
            if morph.convexify is None:
                morph.convexify = isinstance(material, (gs.materials.Rigid, gs.materials.Avatar))

        self._uid = gs.UID()
        self._idx = idx
        self._scene = scene
        self._solver = solver
        self._material = material
        self._morph = morph
        self._surface = surface
        self._sim = scene.sim

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
