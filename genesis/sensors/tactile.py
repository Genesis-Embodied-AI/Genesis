from typing import Any, Dict, List, Optional

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.utils.geom import inv_transform_by_quat, ti_inv_transform_by_quat, transform_by_trans_quat
from genesis.utils.misc import tensor_to_array

from .base_sensor import Sensor


@ti.data_oriented
class RigidContactSensor(Sensor):
    """
    Sensor that returns bool based on whether associated RigidLink is in collision.

    Parameters
    ----------
    entity : RigidEntity
        The entity to monitor the contact state of.
    link_idx : int
        The index of the link to which this sensor is attached. If None, defaults to the base link of the entity.
    """

    _all_contacts: Dict[str, Any] | None = None
    _last_contacts_update_step = -1

    def __init__(self, entity: RigidEntity, link_idx=None):
        self._cls = self.__class__
        self._cached = None
        self._entity = entity
        self._sim = entity._sim
        self.link_idx = link_idx if link_idx is not None else entity.base_link_idx

    def read(self, envs_idx: Optional[List[int]] = None):
        if self._cls._last_contacts_update_step == self._sim.cur_step_global:
            return self._cached

        self._cls._last_contacts_update_step = self._sim.cur_step_global
        self._cls._all_contacts = self._sim.rigid_solver.collider.get_contacts(
            as_tensor=True, to_torch=True, keep_batch_dim=True
        )

        contact_links = torch.cat([self._cls._all_contacts["link_a"], self._cls._all_contacts["link_b"]], dim=1)
        is_contact = (contact_links == self.link_idx).any(dim=1)

        self._cached = is_contact

        return is_contact[envs_idx] if envs_idx is not None else is_contact


@ti.data_oriented
class RigidContactForceSensor(RigidContactSensor):
    """
    Sensor that returns contact force and position based on its associated RigidLink's collision info.

    Parameters
    ----------
    entity : RigidEntity
        The entity to monitor the contact forces of.
    link_idx : int, optional
        The index of the link to which this sensor is attached. If None, defaults to the base link of the entity.
    use_local_frame : bool
        Whether to return forces and positions in the local frame of the link. Defaults to True.
    """

    def __init__(self, entity: RigidEntity, link_idx=None, use_local_frame: bool = True):
        super().__init__(entity, link_idx)
        self._last_updated_step = -1
        self._use_local_frame = use_local_frame

    def read(self, envs_idx=None):
        if self._cls._last_contacts_update_step != self._sim.cur_step_global:
            self._cls._last_contacts_update_step = self._sim.cur_step_global
            self._cls._all_contacts = self._sim.rigid_solver.collider.get_contacts(
                as_tensor=True, to_torch=True, keep_batch_dim=True
            )
        if self._last_updated_step == self._sim.cur_step_global:
            return self._cached

        mask = (self._cls._all_contacts["link_a"] == self.link_idx) | (
            self._cls._all_contacts["link_b"] == self.link_idx
        )

        forces = self._cls._all_contacts["force"][mask]
        poss = self._cls._all_contacts["position"][mask]
        if self._use_local_frame:
            link_quat = self._sim.rigid_solver.get_links_quat(self.link_idx).squeeze(axis=1)
            forces = inv_transform_by_quat(forces, link_quat)
            poss = inv_transform_by_quat(poss, link_quat)

        self._cached = (tensor_to_array(forces), tensor_to_array(poss))
        self._last_updated_step = self._sim.cur_step_global

        return forces[envs_idx], poss[envs_idx] if envs_idx is not None else forces, poss


@ti.data_oriented
class RigidContactForceGridSensor(RigidContactForceSensor):
    """
    Sensor that returns local contact forces as a grid based on its associated RigidLink's collision info.

    Parameters
    ----------
    entity : RigidEntity
        The entity to monitor the contact forces of.
    link_idx : int, optional
        The index of the link to which this sensor is attached. If None, defaults to the base link of the entity.
    grid_size : tuple(int, int, int)
        The size of the grid in which contact forces will be recorded.
        The bounding box of the link is divided into this grid size, and forces are accumulated in each grid cell.
        The bounds are determined by the minimum and maximum vertex positions of the link's initial geometries
        from the mesh after scale but before any rigid transformations are applied.
    """

    def __init__(self, entity: RigidEntity, link_idx=None, grid_size=(1, 1, 1)):
        super().__init__(entity, link_idx)

        self.grid_size = np.array(grid_size, dtype=np.int32)

        link = self._sim.rigid_solver.links[self.link_idx]
        verts = np.concatenate(
            [transform_by_trans_quat(geom._init_verts, geom.init_pos, geom.init_quat) for geom in link._geoms]
        )
        self._min_bounds = np.array(verts.min(axis=-2, keepdims=True)[0], dtype=np.float32)
        self._max_bounds = np.array(verts.max(axis=-2, keepdims=True)[0], dtype=np.float32)
        self._bounds_size = self._max_bounds - self._min_bounds

    def read(self, envs_idx=None):
        if self._cls._last_contacts_update_step != self._sim.cur_step_global:
            self._cls._last_contacts_update_step = self._sim.cur_step_global
            self._cls._all_contacts = self._sim.rigid_solver.collider.get_contacts(
                as_tensor=True, to_torch=True, keep_batch_dim=True
            )
        if self._last_updated_step == self._sim.cur_step_global:
            return self._cached

        grid = np.zeros((self._sim._B, *self.grid_size, 3), dtype=np.float32)
        link_mask = (self._cls._all_contacts["link_a"] == self.link_idx) | (
            self._cls._all_contacts["link_b"] == self.link_idx
        )

        if link_mask.any():

            link_pos = self._sim.rigid_solver.get_links_pos(links_idx=self.link_idx).squeeze(axis=1)
            link_quat = self._sim.rigid_solver.get_links_quat(links_idx=self.link_idx).squeeze(axis=1)

            self._kernel_update_grid(
                grid,
                self._cls._all_contacts["force"].contiguous(),
                self._cls._all_contacts["position"].contiguous(),
                link_mask,
                link_pos,
                link_quat,
                self._min_bounds,
                self._bounds_size,
                self.grid_size,
            )

        self._cached = grid
        self._last_updated_step = self._sim.cur_step_global

        return grid[envs_idx] if envs_idx is not None else grid

    @ti.kernel
    def _kernel_update_grid(
        self,
        grid: ti.types.ndarray(),
        contact_forces: ti.types.ndarray(),
        contact_poss: ti.types.ndarray(),
        link_mask: ti.types.ndarray(),
        link_pos: ti.types.ndarray(),
        link_quat: ti.types.ndarray(),
        min_bounds: ti.types.ndarray(),
        bounds_size: ti.types.ndarray(),
        grid_size: ti.types.ndarray(),
    ):
        for i_b, i_c in ti.ndrange(contact_forces.shape[0], contact_forces.shape[1]):
            if link_mask[i_b, i_c]:
                relative_pos = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    relative_pos[j] = contact_poss[i_b, i_c, j] - link_pos[i_b, j]

                quat = ti.Vector.zero(gs.ti_float, 4)
                for j in ti.static(range(4)):
                    quat[j] = link_quat[i_b, j]

                pos = ti_inv_transform_by_quat(relative_pos, quat)

                contact_force = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    contact_force[j] = contact_forces[i_b, i_c, j]

                force = ti_inv_transform_by_quat(contact_force, quat)

                normalized_pos = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    normalized_pos[j] = (pos[j] - min_bounds[j]) / bounds_size[j]

                grid_x = ti.cast(normalized_pos[0] * grid_size[0], ti.i32)
                grid_y = ti.cast(normalized_pos[1] * grid_size[1], ti.i32)
                grid_z = ti.cast(normalized_pos[2] * grid_size[2], ti.i32)

                if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1] and 0 <= grid_z < grid_size[2]:
                    for j in ti.static(range(3)):
                        grid[i_b, grid_x, grid_y, grid_z, j] += force[j]

    @property
    def min_bounds(self):
        return self._min_bounds

    @property
    def max_bounds(self):
        return self._max_bounds
