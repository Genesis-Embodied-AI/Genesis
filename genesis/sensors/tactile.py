import torch
import taichi as ti
import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.utils.misc import tensor_to_array
from genesis.utils.geom import inv_transform_by_quat, transform_by_quat, ti_inv_transform_by_quat, ti_transform_by_quat
from typing import Dict, List, Any, Optional
from .base_sensor import Sensor
import numpy as np


@ti.data_oriented
class RigidContactSensor(Sensor):
    """
    Sensor that returns bool based on whether associated RigidLink is in collision.
    """

    _all_contacts: Dict[str, Any] | None = None
    _last_updated_step = -1

    @staticmethod
    def get_valid_entity_types():
        return RigidEntity

    def __init__(self, entity, link_idx=None):
        super().__init__(entity)
        self._cls = self.__class__
        self.link_idx = link_idx if link_idx is not None else entity.base_link_idx

    def read(self, envs_idx: Optional[List[int]] = None):
        self._check_envs_idx(envs_idx)
        self._cls._update_contacts_buffer(self._sim)

        if self.n_envs == 0:
            contact_links = torch.cat([self._cls._all_contacts["link_a"], self._cls._all_contacts["link_b"]], dim=0)
            is_contact = (contact_links == self.link_idx).any().item()
        else:
            contact_links = torch.cat([self._cls._all_contacts["link_a"], self._cls._all_contacts["link_b"]], dim=1)
            is_contact = (contact_links == self.link_idx).any(dim=1)
            if envs_idx is not None:
                is_contact = is_contact[envs_idx]

        return is_contact

    @classmethod
    def _update_contacts_buffer(cls, sim):
        if cls._last_updated_step != sim.cur_step_global:
            cls._last_updated_step = sim.cur_step_global
            cls._all_contacts = sim.rigid_solver.collider.get_contacts(as_tensor=sim.n_envs > 0, to_torch=True)


@ti.data_oriented
class RigidContactForceSensor(RigidContactSensor):
    """
    Sensor that returns local contact force and position based on its associated RigidLink's collision info.
    """

    def build(self):
        self._buffer = None
        self._last_updated_step = -1

    def _update_buffer(self):
        self._cls._update_contacts_buffer(self._sim)
        if self._last_updated_step == self._sim.cur_step_global:
            return

        mask = (self._cls._all_contacts["link_a"] == self.link_idx) | (
            self._cls._all_contacts["link_b"] == self.link_idx
        )

        forces = self._cls._all_contacts["force"][mask]
        poss = self._cls._all_contacts["position"][mask]

        self._last_updated_step = self._sim.cur_step_global
        self._buffer = (tensor_to_array(forces), tensor_to_array(poss))

    def read(self, envs_idx=None):
        self._check_envs_idx(envs_idx)
        self._update_buffer()
        forces, poss = self._buffer
        return forces, poss if envs_idx is None else forces[envs_idx], poss[envs_idx]


@ti.data_oriented
class RigidContactForceGridSensor(RigidContactSensor):
    """
    Sensor that returns local contact forces as a grid based on its associated RigidLink's collision info.
    """

    def __init__(self, entity, link_idx=None, grid_size=(1, 1, 1)):
        super().__init__(entity, link_idx)
        self.grid_size = np.array(grid_size, dtype=np.int32)

    def build(self):
        super().build()

        self._buffer = None
        self._last_updated_step = -1

        link = self._sim.rigid_solver.links[self.link_idx]
        verts = np.concatenate([geom._init_verts for geom in link._geoms])
        self.min_bounds = np.array(verts.min(axis=-2, keepdims=True)[0], dtype=np.float32)
        self.max_bounds = np.array(verts.max(axis=-2, keepdims=True)[0], dtype=np.float32)
        self.bounds_size = self.max_bounds - self.min_bounds

    def _update_buffer(self):
        self._cls._update_contacts_buffer(self._sim)

        if self._last_updated_step == self._sim.cur_step_global:
            return

        grid = np.zeros((self._B, *self.grid_size, 3), dtype=np.float32)
        link_mask = (self._cls._all_contacts["link_a"] == self.link_idx) | (
            self._cls._all_contacts["link_b"] == self.link_idx
        )

        if link_mask.any():

            if self.n_envs == 0:

                grid = np.zeros((*self.grid_size, 3), dtype=np.float32)
                contact_forces = self._cls._all_contacts["force"][link_mask]
                contact_poss = self._cls._all_contacts["position"][link_mask]

                link_pos = self._sim.rigid_solver.get_links_pos(links_idx=self.link_idx)
                link_quat = self._sim.rigid_solver.get_links_quat(links_idx=self.link_idx)

                relative_pos = contact_poss - link_pos
                poss = inv_transform_by_quat(relative_pos, link_quat)

                for i in range(contact_forces.shape[0]):
                    force = tensor_to_array(transform_by_quat(contact_forces[i], link_quat.squeeze()))
                    pos = tensor_to_array(poss[i])

                    normalized_pos = (pos - self.min_bounds) / self.bounds_size
                    grid_pos = (normalized_pos * self.grid_size).astype(int)

                    if np.all((grid_pos >= 0) & (grid_pos < self.grid_size)):
                        grid[grid_pos[0], grid_pos[1], grid_pos[2]] += force

            else:
                link_pos = self._sim.rigid_solver.get_links_pos(links_idx=self.link_idx).squeeze(axis=1)
                link_quat = self._sim.rigid_solver.get_links_quat(links_idx=self.link_idx).squeeze(axis=1)

                self._kernel_update_grid(
                    grid,
                    self._cls._all_contacts["force"].contiguous(),
                    self._cls._all_contacts["position"].contiguous(),
                    link_mask,
                    link_pos.contiguous(),
                    link_quat.contiguous(),
                    self.min_bounds,
                    self.bounds_size,
                    self.grid_size,
                )

        self._last_updated_step = self._sim.cur_step_global
        self._buffer = grid

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
        for i_b in range(self._B):
            for i_c in range(contact_forces.shape[1]):  # max contacts per env
                if link_mask[i_b, i_c]:  # only process contacts for this link
                    # Transform position to link frame
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

                    force = ti_transform_by_quat(contact_force, quat)

                    normalized_pos = ti.Vector.zero(gs.ti_float, 3)
                    for j in ti.static(range(3)):
                        normalized_pos[j] = (pos[j] - min_bounds[j]) / bounds_size[j]

                    grid_x = ti.cast(normalized_pos[0] * grid_size[0], ti.i32)
                    grid_y = ti.cast(normalized_pos[1] * grid_size[1], ti.i32)
                    grid_z = ti.cast(normalized_pos[2] * grid_size[2], ti.i32)

                    if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1] and 0 <= grid_z < grid_size[2]:
                        for j in ti.static(range(3)):
                            grid[i_b, grid_x, grid_y, grid_z, j] += force[j]

    def read(self, envs_idx=None):
        self._check_envs_idx(envs_idx)
        self._update_buffer()
        grid = self._buffer
        return grid[envs_idx] if envs_idx is not None else grid
