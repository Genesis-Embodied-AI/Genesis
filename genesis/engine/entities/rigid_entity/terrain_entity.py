import numpy as np
import torch

import genesis as gs
from genesis.utils import geom as gu
from genesis.utils import terrain as tu

from .description import TerrainEntityDescription
from .rigid_entity import RigidEntity


class TerrainEntity(RigidEntity):
    """A rigid entity whose single fixed link is a height field, authored as a grid of elevations.

    Links rest on the surface those elevations define, and the stored grid answers a height query at any point.
    """

    _description_cls = TerrainEntityDescription

    def _load_model(self):
        super()._load_model()
        # Collider storage covers only the first collision terrain, while height queries support every terrain
        # entity, kinematic ones included. Assigned after the model loads, so it covers a morph and a description.
        self._terrain_height_field = torch.as_tensor(self.terrain_hf, dtype=gs.tc_float, device=gs.device)

    @property
    def terrain_hf(self) -> np.ndarray:
        """Elevation of every grid cell in meters, the vertical scale of the morph applied."""
        return self._desc.terrain_hf

    @property
    def terrain_scale(self) -> np.ndarray:
        """Horizontal size of a grid cell in meters, and the factor the elevation is scaled by."""
        return self._desc.terrain_scale

    def _load_morph(self, morph):
        """Load the height field a terrain morph describes, with its scales and the meshes standing for it."""
        vmesh, mesh, terrain_hf = tu.parse_terrain(morph, self._surface)
        self._desc.terrain_scale = np.array((morph.horizontal_scale, morph.vertical_scale), dtype=gs.np_float)
        self._desc.terrain_hf = terrain_hf * self._desc.terrain_scale[1]

        g_infos = []
        if morph.visualization:
            g_infos.append(dict(contype=0, conaffinity=0, vmesh=vmesh))
        if morph.collision:
            g_infos.append(
                dict(
                    contype=1,
                    conaffinity=1,
                    mesh=mesh,
                    type=gs.GEOM_TYPE.TERRAIN,
                    sol_params=gu.default_solver_params(),
                    pos=gu.zero_pos(),
                    quat=gu.identity_quat(),
                )
            )

        self._resolve_link(
            l_info=dict(
                name="baselink",
                parent_idx=-1,
                root_idx=None,
                pos=np.array(morph.pos),
                quat=np.array(morph.quat),
                is_robot=False,
                inertial_pos=None,
                inertial_quat=gu.identity_quat(),
                inertial_i=None,
                inertial_mass=None,
                invweight=None,
            ),
            j_infos=[dict(name="joint_baselink", type=gs.JOINT_TYPE.FIXED, n_qs=0, n_dofs=0)],
            g_infos=g_infos,
            morph=morph,
        )

        # Load heterogeneous variants (if any)
        self._load_heterogeneous_morphs()

    @gs.assert_built
    def get_terrain_height(self, positions, envs_idx=None):
        """
        Return terrain surface heights in meters at world-frame x-y positions.

        Heights match the piecewise-planar surface on which rigid bodies rest. Terrain translation, yaw, and
        environment-specific poses are applied to the query. Positions up to one grid cell outside the terrain are
        clamped to its edge. Terrain tilt up to 0.001 radians from world vertical is treated as numerical noise. The
        query returns not-a-number (NaN) heights for positions containing NaN or infinity, positions farther outside,
        and terrains with greater tilt.

        Parameters
        ----------
        positions : array_like
            World-frame x-y positions in meters, with shape (2,), (n_points, 2), or
            (n_selected_envs, n_points, 2). A two-dimensional array is shared across the selected environments; use a
            three-dimensional array for environment-specific positions. A leading dimension of 1 in the
            three-dimensional form is also treated as shared.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        heights : torch.Tensor
            World-frame surface heights in meters. The point dimension is preserved except for an explicit `(2,)`
            input, and an environment dimension is prepended in a parallelized scene.

        Raises
        ------
        GenesisException
            If the shape of `positions` is unsupported.
        """
        return self._solver.get_terrain_height(positions, self.base_link_idx, envs_idx)
