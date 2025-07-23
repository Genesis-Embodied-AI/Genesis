from collections.abc import Iterable
from typing import List, Optional

import numpy as np
import taichi as ti
import torch

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.utils.geom import ti_inv_transform_by_quat, transform_by_trans_quat
from genesis.utils.misc import tensor_to_array

from .base_sensor import Sensor


@ti.data_oriented
class RigidContactSensor(Sensor):
    """
    Sensor that returns bool based on whether associated RigidLink is in collision.

    read() -> np.ndarray of shape (batch_size,) of bool values

    Parameters
    ----------
    entity : RigidEntity
        The entity to monitor the contact state of.
    link_idx : int
        The index of the link to which this sensor is attached. If None, defaults to the base link of the entity.
    """

    _cached_link_is_colliding: torch.Tensor | None = None
    _last_shared_cache_step = -1

    def __init__(
        self,
        entity: RigidEntity,
        link_idx=None,
    ):
        self._entity = entity
        self._sim = entity._sim
        self._solver = self._sim.rigid_solver
        self.link_idx = link_idx if link_idx is not None else entity.base_link_idx
        self._cached = None
        self._last_cache_step = -1

    def read(self, envs_idx: Optional[List[int]] = None) -> np.ndarray:
        if type(self)._cached_link_is_colliding is None:
            type(self)._cached_link_is_colliding = torch.zeros(
                (self._sim._B, self._solver.n_links), dtype=gs.tc_bool, device=gs.device
            )

        if type(self)._last_shared_cache_step != self._sim.cur_step_global:
            type(self)._last_shared_cache_step = self._sim.cur_step_global
            type(self)._cached_link_is_colliding.fill_(False)
            self._kernel_get_colliding_links(
                self._solver.collider._collider_state,
                type(self)._cached_link_is_colliding,
            )

        if self._cached is None:
            self._cached = torch.zeros((self._sim._B,), dtype=gs.tc_bool, device=gs.device)

        if self._last_cache_step != self._sim.cur_step_global:
            self._last_cache_step = self._sim.cur_step_global
            self._cached = type(self)._cached_link_is_colliding[:, self.link_idx]

        return tensor_to_array(self._cached[envs_idx] if envs_idx is not None else self._cached)

    @ti.kernel
    def _kernel_get_colliding_links(
        self,
        collider_state: ti.template(),
        output: ti.types.ndarray(),
    ):
        for i_b in range(output.shape[0]):
            for i_c_ in range(collider_state.n_contacts[i_b]):
                output[i_b, collider_state.contact_data[i_c_, i_b].link_a] = True
                output[i_b, collider_state.contact_data[i_c_, i_b].link_b] = True


@ti.data_oriented
class RigidContactForceSensor(RigidContactSensor):
    """
    Sensor that returns contact force (Fx, Fy, Fz) in the associated RigidLink's local frame.

    read() -> np.ndarray of shape (batch_size, 3) representing the contact force.

    Parameters
    ----------
    entity : RigidEntity
        The entity to monitor the contact forces of.
    link_idx : int, optional
        The index of the link to which this sensor is attached. If None, defaults to the base link of the entity.
    """

    _cached_link_forces: torch.Tensor | None = None
    _last_shared_cache_step = -1

    def read(self, envs_idx: Iterable[int] | None = None):
        if type(self)._cached_link_forces is None:
            type(self)._cached_link_forces = torch.zeros(
                (self._sim._B, self._solver.n_links, 3), dtype=gs.tc_float, device=gs.device
            )

        if type(self)._last_shared_cache_step != self._sim.cur_step_global:
            type(self)._last_shared_cache_step = self._sim.cur_step_global
            type(self)._cached_link_forces.fill_(0.0)
            self._kernel_get_contacts_forces(
                self._solver.collider._collider_state,
                self._solver.links_state.quat.to_numpy(),
                type(self)._cached_link_forces,
            )

        if self._cached is None:
            self._cached = torch.zeros((self._sim._B, 3), dtype=gs.tc_float, device=gs.device)

        if self._last_cache_step != self._sim.cur_step_global:
            self._last_cache_step = self._sim.cur_step_global
            self._cached = type(self)._cached_link_forces[:, self.link_idx]

        return tensor_to_array(self._cached[envs_idx] if envs_idx is not None else self._cached)

    @ti.kernel
    def _kernel_get_contacts_forces(
        self,
        collider_state: ti.template(),
        links_quat: ti.types.ndarray(),
        output: ti.types.ndarray(),
    ):
        for i_b in range(output.shape[0]):
            for i_c_ in range(collider_state.n_contacts[i_b]):
                contact_data = collider_state.contact_data[i_c_, i_b]

                quat_a = ti.Vector.zero(ti.f32, 4)
                quat_b = ti.Vector.zero(ti.f32, 4)
                for j in ti.static(range(4)):
                    quat_a[j] = links_quat[contact_data.link_a, i_b, j]
                    quat_b[j] = links_quat[contact_data.link_b, i_b, j]

                force_a = ti_inv_transform_by_quat(-contact_data.force, quat_a)
                force_b = ti_inv_transform_by_quat(contact_data.force, quat_b)

                for j in ti.static(range(3)):
                    output[i_b, contact_data.link_a, j] = force_a[j]
                    output[i_b, contact_data.link_b, j] = force_b[j]


@ti.data_oriented
class RigidNormalTangentialForceSensor(RigidContactSensor):
    """
    Sensor that returns (|fn|, |ft|, tx, ty) for contact force normal, tangential shear force, and tangential
    direction as a unit vector in the associated RigidLink's local frame.
    The tangential direction will not be computed if the tangential force < FORCE_EPSILON, since it will be noise.

    read() -> np.ndarray of shape (batch_size, 4) representing the normal and tangential forces.

    Parameters
    ----------
    entity : RigidEntity
        The entity to monitor the contact forces of.
    link_idx : int, optional
        The index of the link to which this sensor is attached. If None, defaults to the base link of the entity.
    """

    FORCE_EPSILON = 1e-5
    _cached_link_norm_tan: torch.Tensor | None = None
    _last_shared_cache_step = -1

    def read(self, envs_idx: Iterable[int] | None = None):
        if type(self)._cached_link_norm_tan is None:
            type(self)._cached_link_norm_tan = torch.zeros(
                (self._sim._B, self._solver.n_links, 4), dtype=gs.tc_float, device=gs.device
            )

        if type(self)._last_shared_cache_step != self._sim.cur_step_global:
            type(self)._last_shared_cache_step = self._sim.cur_step_global
            type(self)._cached_link_norm_tan.fill_(0.0)
            self._kernel_get_contacts_norm_tan(
                self._solver.collider._collider_state,
                self._solver.links_state.quat.to_numpy(),
                type(self).FORCE_EPSILON,
                type(self)._cached_link_norm_tan,
            )

        if self._cached is None:
            self._cached = torch.zeros((self._sim._B, 4), dtype=gs.tc_float, device=gs.device)

        if self._last_cache_step != self._sim.cur_step_global:
            self._last_cache_step = self._sim.cur_step_global
            self._cached = type(self)._cached_link_norm_tan[:, self.link_idx]

        return tensor_to_array(self._cached[envs_idx] if envs_idx is not None else self._cached)

    @ti.kernel
    def _kernel_get_contacts_norm_tan(
        self,
        collider_state: ti.template(),
        links_quat: ti.types.ndarray(),
        force_eps: ti.f32,
        output: ti.types.ndarray(),
    ):
        for i_b in range(output.shape[0]):
            for i_c_ in range(collider_state.n_contacts[i_b]):
                contact_data = collider_state.contact_data[i_c_, i_b]

                quat_a = ti.Vector.zero(ti.f32, 4)
                quat_b = ti.Vector.zero(ti.f32, 4)
                for j in ti.static(range(4)):
                    quat_a[j] = links_quat[contact_data.link_a, i_b, j]
                    quat_b[j] = links_quat[contact_data.link_b, i_b, j]

                force_a_local = ti_inv_transform_by_quat(-contact_data.force, quat_a)
                normal_a_local = ti_inv_transform_by_quat(contact_data.normal, quat_a)
                force_b_local = ti_inv_transform_by_quat(contact_data.force, quat_b)
                normal_b_local = ti_inv_transform_by_quat(-contact_data.normal, quat_b)

                fn_a = ti.abs(force_a_local.dot(normal_a_local))
                force_tangential_a = force_a_local - force_a_local.dot(normal_a_local) * normal_a_local

                fn_b = ti.abs(force_b_local.dot(normal_b_local))
                force_tangential_b = force_b_local - force_b_local.dot(normal_b_local) * normal_b_local

                output[i_b, contact_data.link_a, 0] += fn_a
                output[i_b, contact_data.link_a, 2] += force_tangential_a[0]
                output[i_b, contact_data.link_a, 3] += force_tangential_a[1]

                output[i_b, contact_data.link_b, 0] += fn_b
                output[i_b, contact_data.link_b, 2] += force_tangential_b[0]
                output[i_b, contact_data.link_b, 3] += force_tangential_b[1]

            for i_l in range(output.shape[1]):
                ft_mag = ti.sqrt(output[i_b, i_l, 2] * output[i_b, i_l, 2] + output[i_b, i_l, 3] * output[i_b, i_l, 3])
                output[i_b, i_l, 1] = ft_mag
                if ft_mag > force_eps:
                    output[i_b, i_l, 2] /= ft_mag
                    output[i_b, i_l, 3] /= ft_mag
                else:
                    output[i_b, i_l, 2] = 0.0
                    output[i_b, i_l, 3] = 0.0


@ti.data_oriented
class RigidContactGridSensor(RigidContactSensor):
    """
    Sensor that returns ndarray of shape [grid_x, grid_y, grid_z] representing whether or not contact is detected
    in each grid cell based on the associated RigidLink's collision info.

    read() -> np.ndarray of shape (batch_size, grid_x, grid_y, grid_z) of bool values

    Parameters
    ----------
    entity : RigidEntity
        The entity to monitor the contact forces of.
    link_idx : int, optional
        The index of the link to which this sensor is attached. If None, defaults to the base link of the entity.
    grid_size : tuple(int, int, int)
        The size of the grid representing the resolution of which contact forces will be detected.
        The bounding box of the link is divided into this grid size, and forces are accumulated in each grid cell.
        The bounds are determined by the minimum and maximum vertex positions of the link's initial geometries
        from the mesh after scale but before any rigid transformations are applied.
    """

    _cached_contacts_local_pos: torch.Tensor | None = None
    _cached_contacts_link: torch.Tensor | None = None
    _last_shared_cache_step = -1

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

    def read(self, envs_idx: Iterable[int] | None = None):
        if type(self)._last_shared_cache_step != self._sim.cur_step_global:
            type(self)._last_shared_cache_step = self._sim.cur_step_global

            n_contacts = max(self._solver.collider._collider_state.n_contacts.to_numpy())
            n_local_contacts = n_contacts * 2  # two links per contact
            type(self)._cached_contacts_local_pos = torch.zeros(
                (self._sim._B, n_local_contacts, 3), dtype=gs.tc_float, device=gs.device
            )
            type(self)._cached_contacts_link = torch.zeros(
                (self._sim._B, n_local_contacts), dtype=gs.tc_int, device=gs.device
            )

            self._kernel_contacts_local_pos(
                self._solver.collider._collider_state,
                self._solver.links_state.pos.to_numpy(),
                self._solver.links_state.quat.to_numpy(),
                n_contacts,
                type(self)._cached_contacts_link,
                type(self)._cached_contacts_local_pos,
            )

        if self._cached is None:
            self._cached = torch.zeros((self._sim._B, *self.grid_size), dtype=gs.tc_bool, device=gs.device)

        if self._last_cache_step != self._sim.cur_step_global:
            self._last_cache_step = self._sim.cur_step_global
            self._cached.fill_(False)
            self._compute_grid(
                type(self)._cached_contacts_local_pos,
                type(self)._cached_contacts_link,
                self.link_idx,
                self.grid_size,
                self._min_bounds,
                self._bounds_size,
                self._cached,
            )

        return tensor_to_array(self._cached[envs_idx] if envs_idx is not None else self._cached)

    def _update_links_cache(self):
        pass

    @ti.kernel
    def _compute_grid(
        self,
        contacts_local_pos: ti.types.ndarray(),
        contacts_link: ti.types.ndarray(),
        link_idx: ti.i32,
        grid_size: ti.types.ndarray(),
        min_bounds: ti.types.ndarray(),
        bounds_size: ti.types.ndarray(),
        output_grid: ti.types.ndarray(),
    ):
        for i_b, i_c in ti.ndrange(contacts_local_pos.shape[0], contacts_local_pos.shape[1]):
            if contacts_link[i_b, i_c] == link_idx:
                pos = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    pos[j] = contacts_local_pos[i_b, i_c, j]

                normalized_pos = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    normalized_pos[j] = (pos[j] - min_bounds[j]) / bounds_size[j]

                grid_x = ti.cast(normalized_pos[0] * grid_size[0], ti.i32)
                grid_y = ti.cast(normalized_pos[1] * grid_size[1], ti.i32)
                grid_z = ti.cast(normalized_pos[2] * grid_size[2], ti.i32)

                if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1] and 0 <= grid_z < grid_size[2]:
                    output_grid[i_b, grid_x, grid_y, grid_z] = True

    @ti.kernel
    def _kernel_contacts_local_pos(
        self,
        collider_state: ti.template(),
        links_pos: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        n_contacts: ti.i32,
        output_links: ti.types.ndarray(),
        output_poss: ti.types.ndarray(),
    ):
        for i_b in range(output_poss.shape[0]):
            for i_c_ in range(collider_state.n_contacts[i_b]):
                contact_data = collider_state.contact_data[i_c_, i_b]

                quat_a = ti.Vector.zero(ti.f32, 4)
                quat_b = ti.Vector.zero(ti.f32, 4)
                for j in ti.static(range(4)):
                    quat_a[j] = links_quat[contact_data.link_a, i_b, j]
                    quat_b[j] = links_quat[contact_data.link_b, i_b, j]

                rel_pos_a = ti.Vector.zero(ti.f32, 3)
                rel_pos_b = ti.Vector.zero(ti.f32, 3)
                for j in ti.static(range(3)):
                    rel_pos_a[j] = contact_data.pos[j] - links_pos[contact_data.link_a, i_b, j]
                    rel_pos_b[j] = contact_data.pos[j] - links_pos[contact_data.link_b, i_b, j]

                pos_a_local = ti_inv_transform_by_quat(rel_pos_a, quat_a)
                pos_b_local = ti_inv_transform_by_quat(rel_pos_b, quat_b)

                output_links[i_b, i_c_] = contact_data.link_a
                output_links[i_b, i_c_ + n_contacts] = contact_data.link_b

                for j in ti.static(range(3)):
                    output_poss[i_b, i_c_, j] = pos_a_local[j]
                    output_poss[i_b, i_c_ + n_contacts, j] = pos_b_local[j]

    @property
    def min_bounds(self):
        return self._min_bounds

    @property
    def max_bounds(self):
        return self._max_bounds


@ti.data_oriented
class RigidContactForceGridSensor(RigidContactGridSensor):
    """
    Sensor that returns ndarray of shape (grid_x, grid_y, grid_z, 3) reprsenting the contact forces (Fx, Fy, Fz)
    in the associated RigidLink's local frame accumulated in a grid.

    read() -> np.ndarray of shape (batch_size, grid_x, grid_y, grid_z, 3) representing the contact forces.

    Parameters
    ----------
    entity : RigidEntity
        The entity to monitor the contact forces of.
    link_idx : int, optional
        The index of the link to which this sensor is attached. If None, defaults to the base link of the entity.
    grid_size : tuple(int, int, int)
        The size of the grid representing the resolution of which contact forces will be accumulated.
        The bounding box of the link is divided into this grid size, and forces are accumulated in each grid cell.
        The bounds are determined by the minimum and maximum vertex positions of the link's initial geometries
        from the mesh after scale but before any rigid transformations are applied.
    """

    _cached_contacts_local_forces: torch.Tensor | None = None
    _last_shared_cache_step = -1

    def __init__(self, entity: RigidEntity, link_idx=None, grid_size=(1, 1, 1)):
        super().__init__(entity, link_idx, grid_size)
        self._max_contacts = None

    def read(self, envs_idx: Iterable[int] | None = None):
        if self._max_contacts is None:
            self._max_contacts = self._solver.collider._collider_info._max_contact_pairs[None]

        if type(self)._cached_contacts_local_pos is None:
            type(self)._cached_contacts_local_pos = torch.zeros(
                (self._sim._B, self._max_contacts * 2, 3), dtype=gs.tc_float, device=gs.device
            )
            type(self)._cached_contacts_link = torch.zeros(
                (self._sim._B, self._max_contacts * 2), dtype=gs.tc_int, device=gs.device
            )
            type(self)._cached_contacts_local_forces = torch.zeros(
                (self._sim._B, self._max_contacts * 2, 3), dtype=gs.tc_float, device=gs.device
            )

        if type(self)._last_shared_cache_step != self._sim.cur_step_global:
            type(self)._last_shared_cache_step = self._sim.cur_step_global

            type(self)._cached_contacts_link.fill_(-1)
            type(self)._cached_contacts_local_pos.fill_(0.0)
            type(self)._cached_contacts_local_forces.fill_(0.0)

            self._kernel_contacts_local_pos_forces(
                self._solver.collider._collider_state,
                self._solver.links_state.pos.to_numpy(),
                self._solver.links_state.quat.to_numpy(),
                self._max_contacts,
                type(self)._cached_contacts_link,
                type(self)._cached_contacts_local_pos,
                type(self)._cached_contacts_local_forces,
            )

        if self._cached is None:
            self._cached = torch.zeros((self._sim._B, *self.grid_size, 3), dtype=gs.tc_float, device=gs.device)

        if self._last_cache_step != self._sim.cur_step_global:
            self._last_cache_step = self._sim.cur_step_global
            self._cached.fill_(0.0)
            self._compute_force_grid(
                type(self)._cached_contacts_link,
                type(self)._cached_contacts_local_pos,
                type(self)._cached_contacts_local_forces,
                self.link_idx,
                self.grid_size,
                self._min_bounds,
                self._bounds_size,
                self._cached,
            )

        return tensor_to_array(self._cached[envs_idx] if envs_idx is not None else self._cached)

    @ti.kernel
    def _compute_force_grid(
        self,
        contacts_link: ti.types.ndarray(),
        contacts_local_pos: ti.types.ndarray(),
        contacts_local_forces: ti.types.ndarray(),
        link_idx: ti.i32,
        grid_size: ti.types.ndarray(),
        min_bounds: ti.types.ndarray(),
        bounds_size: ti.types.ndarray(),
        output_grid: ti.types.ndarray(),
    ):
        for i_b, i_c in ti.ndrange(contacts_local_pos.shape[0], contacts_local_pos.shape[1]):
            if contacts_link[i_b, i_c] == link_idx:
                pos = ti.Vector.zero(gs.ti_float, 3)
                force = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    pos[j] = contacts_local_pos[i_b, i_c, j]
                    force[j] = contacts_local_forces[i_b, i_c, j]

                normalized_pos = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    if bounds_size[j] > 0:
                        normalized_pos[j] = ti.math.clamp((pos[j] - min_bounds[j]) / bounds_size[j], 0.0, 1.0)
                    else:
                        normalized_pos[j] = 0.5

                grid_x = ti.min(ti.cast(normalized_pos[0] * grid_size[0], ti.i32), grid_size[0] - 1)
                grid_y = ti.min(ti.cast(normalized_pos[1] * grid_size[1], ti.i32), grid_size[1] - 1)
                grid_z = ti.min(ti.cast(normalized_pos[2] * grid_size[2], ti.i32), grid_size[2] - 1)

                if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1] and 0 <= grid_z < grid_size[2]:
                    for j in ti.static(range(3)):
                        output_grid[i_b, grid_x, grid_y, grid_z, j] += force[j]

    @ti.kernel
    def _kernel_contacts_local_pos_forces(
        self,
        collider_state: ti.template(),
        links_pos: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        n_contacts: ti.i32,
        output_links: ti.types.ndarray(),
        output_poss: ti.types.ndarray(),
        output_forces: ti.types.ndarray(),
    ):
        for i_b in range(output_poss.shape[0]):
            for i_c_ in range(collider_state.n_contacts[i_b]):
                contact_data = collider_state.contact_data[i_c_, i_b]

                quat_a = ti.Vector.zero(ti.f32, 4)
                quat_b = ti.Vector.zero(ti.f32, 4)
                for j in ti.static(range(4)):
                    quat_a[j] = links_quat[contact_data.link_a, i_b, j]
                    quat_b[j] = links_quat[contact_data.link_b, i_b, j]

                rel_pos_a = ti.Vector.zero(ti.f32, 3)
                rel_pos_b = ti.Vector.zero(ti.f32, 3)
                for j in ti.static(range(3)):
                    rel_pos_a[j] = contact_data.pos[j] - links_pos[contact_data.link_a, i_b, j]
                    rel_pos_b[j] = contact_data.pos[j] - links_pos[contact_data.link_b, i_b, j]

                pos_a_local = ti_inv_transform_by_quat(rel_pos_a, quat_a)
                pos_b_local = ti_inv_transform_by_quat(rel_pos_b, quat_b)

                force_a_local = ti_inv_transform_by_quat(-contact_data.force, quat_a)
                force_b_local = ti_inv_transform_by_quat(contact_data.force, quat_b)

                output_links[i_b, i_c_] = contact_data.link_a
                output_links[i_b, i_c_ + n_contacts] = contact_data.link_b

                for j in ti.static(range(3)):
                    output_poss[i_b, i_c_, j] = pos_a_local[j]
                    output_poss[i_b, i_c_ + n_contacts, j] = pos_b_local[j]

                    output_forces[i_b, i_c_, j] = force_a_local[j]
                    output_forces[i_b, i_c_ + n_contacts, j] = force_b_local[j]


@ti.data_oriented
class RigidNormalTangentialForceGridSensor(RigidContactForceGridSensor):
    """
    Sensor that returns ndarray of shape (grid_x, grid_y, grid_z, 4) representing the accumulated contact forces.
    Each grid cell contains (|fn|, |ft|, tx, ty) for contact force normal, tangential shear force, and tangential
    direction as a unit vector in the associated RigidLink's local frame.

    The tangential direction will not be computed if the tangential force < FORCE_EPSILON, since it will be noise.

    read() -> np.ndarray of shape (batch_size, grid_x, grid_y, grid_z, 4) representing the contact forces.

    Parameters
    ----------
    entity : RigidEntity
        The entity to monitor the contact forces of.
    link_idx : int, optional
        The index of the link to which this sensor is attached. If None, defaults to the base link of the entity.
    grid_size : tuple(int, int, int)
        The size of the grid representing the resolution of which contact forces will be accumulated.
        The bounding box of the link is divided into this grid size, and forces are accumulated in each grid cell.
        The bounds are determined by the minimum and maximum vertex positions of the link's initial geometries
        from the mesh after scale but before any rigid transformations are applied.
    """

    FORCE_EPSILON = 1e-5
    _cached_contacts_local_norm_tans: torch.Tensor | None = None
    _last_shared_cache_step = -1

    def read(self, envs_idx: Iterable[int] | None = None):
        if self._max_contacts is None:
            self._max_contacts = self._solver.collider._collider_info._max_contact_pairs[None]

        if type(self)._cached_contacts_local_pos is None:
            type(self)._cached_contacts_local_pos = torch.zeros(
                (self._sim._B, self._max_contacts * 2, 3), dtype=gs.tc_float, device=gs.device
            )

        if type(self)._cached_contacts_link is None:
            type(self)._cached_contacts_link = torch.zeros(
                (self._sim._B, self._max_contacts * 2), dtype=gs.tc_int, device=gs.device
            )

        if type(self)._cached_contacts_local_norm_tans is None:
            type(self)._cached_contacts_local_norm_tans = torch.zeros(
                (self._sim._B, self._max_contacts * 2, 4), dtype=gs.tc_float, device=gs.device
            )

        if type(self)._last_shared_cache_step != self._sim.cur_step_global:
            type(self)._last_shared_cache_step = self._sim.cur_step_global

            type(self)._cached_contacts_local_pos.fill_(0.0)
            type(self)._cached_contacts_link.fill_(-1)
            type(self)._cached_contacts_local_norm_tans.fill_(0.0)

            self._kernel_contacts_local_pos_norm_tans(
                self._solver.collider._collider_state,
                self._solver.links_state.pos.to_numpy(),
                self._solver.links_state.quat.to_numpy(),
                self._max_contacts,
                type(self)._cached_contacts_link,
                type(self)._cached_contacts_local_pos,
                type(self)._cached_contacts_local_norm_tans,
            )

        if self._cached is None:
            self._cached = torch.zeros((self._sim._B, *self.grid_size, 4), dtype=gs.tc_float, device=gs.device)

        if self._last_cache_step != self._sim.cur_step_global:
            self._last_cache_step = self._sim.cur_step_global
            self._cached.fill_(0.0)
            self._compute_norm_tan_grid(
                type(self)._cached_contacts_link,
                type(self)._cached_contacts_local_pos,
                type(self)._cached_contacts_local_norm_tans,
                self.link_idx,
                self.grid_size,
                self._min_bounds,
                self._bounds_size,
                type(self).FORCE_EPSILON,
                self._cached,
            )

        return tensor_to_array(self._cached[envs_idx] if envs_idx is not None else self._cached)

    @ti.kernel
    def _compute_norm_tan_grid(
        self,
        contacts_link: ti.types.ndarray(),
        contacts_local_pos: ti.types.ndarray(),
        contacts_local_norm_tans: ti.types.ndarray(),
        link_idx: ti.i32,
        grid_size: ti.types.ndarray(),
        min_bounds: ti.types.ndarray(),
        bounds_size: ti.types.ndarray(),
        force_eps: ti.f32,
        output_grid: ti.types.ndarray(),
    ):
        for i_b, i_c in ti.ndrange(contacts_local_pos.shape[0], contacts_local_pos.shape[1]):
            if contacts_link[i_b, i_c] == link_idx:
                pos = ti.Vector.zero(gs.ti_float, 3)
                norm_tan = ti.Vector.zero(gs.ti_float, 4)
                for j in ti.static(range(3)):
                    pos[j] = contacts_local_pos[i_b, i_c, j]
                for j in ti.static(range(4)):
                    norm_tan[j] = contacts_local_norm_tans[i_b, i_c, j]

                normalized_pos = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    if bounds_size[j] > 0:
                        normalized_pos[j] = ti.math.clamp((pos[j] - min_bounds[j]) / bounds_size[j], 0.0, 1.0)
                    else:
                        normalized_pos[j] = 0.5

                grid_x = ti.min(ti.cast(normalized_pos[0] * grid_size[0], ti.i32), grid_size[0] - 1)
                grid_y = ti.min(ti.cast(normalized_pos[1] * grid_size[1], ti.i32), grid_size[1] - 1)
                grid_z = ti.min(ti.cast(normalized_pos[2] * grid_size[2], ti.i32), grid_size[2] - 1)

                if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1] and 0 <= grid_z < grid_size[2]:
                    output_grid[i_b, grid_x, grid_y, grid_z, 0] += norm_tan[0]
                    output_grid[i_b, grid_x, grid_y, grid_z, 2] += norm_tan[2]
                    output_grid[i_b, grid_x, grid_y, grid_z, 3] += norm_tan[3]

            for i_x, i_y, i_z in ti.ndrange(output_grid.shape[1], output_grid.shape[2], output_grid.shape[3]):
                tx_accum = output_grid[i_b, i_x, i_y, i_z, 2]
                ty_accum = output_grid[i_b, i_x, i_y, i_z, 3]

                ft_mag = ti.sqrt(tx_accum * tx_accum + ty_accum * ty_accum)

                output_grid[i_b, i_x, i_y, i_z, 1] = ft_mag
                if ft_mag > force_eps:
                    output_grid[i_b, i_x, i_y, i_z, 2] /= ft_mag
                    output_grid[i_b, i_x, i_y, i_z, 3] /= ft_mag
                else:
                    output_grid[i_b, i_x, i_y, i_z, 2] = 0.0
                    output_grid[i_b, i_x, i_y, i_z, 3] = 0.0

    @ti.kernel
    def _kernel_contacts_local_pos_norm_tans(
        self,
        collider_state: ti.template(),
        links_pos: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        n_contacts: ti.i32,
        output_links: ti.types.ndarray(),
        output_poss: ti.types.ndarray(),
        output_norm_tans: ti.types.ndarray(),
    ):
        for i_b in range(output_poss.shape[0]):
            for i_c_ in range(collider_state.n_contacts[i_b]):
                contact_data = collider_state.contact_data[i_c_, i_b]

                quat_a = ti.Vector.zero(ti.f32, 4)
                quat_b = ti.Vector.zero(ti.f32, 4)
                for j in ti.static(range(4)):
                    quat_a[j] = links_quat[contact_data.link_a, i_b, j]
                    quat_b[j] = links_quat[contact_data.link_b, i_b, j]

                rel_pos_a = ti.Vector.zero(ti.f32, 3)
                rel_pos_b = ti.Vector.zero(ti.f32, 3)
                for j in ti.static(range(3)):
                    rel_pos_a[j] = contact_data.pos[j] - links_pos[contact_data.link_a, i_b, j]
                    rel_pos_b[j] = contact_data.pos[j] - links_pos[contact_data.link_b, i_b, j]

                pos_a_local = ti_inv_transform_by_quat(rel_pos_a, quat_a)
                pos_b_local = ti_inv_transform_by_quat(rel_pos_b, quat_b)

                force_a_local = ti_inv_transform_by_quat(-contact_data.force, quat_a)
                normal_a_local = ti_inv_transform_by_quat(contact_data.normal, quat_a)
                force_b_local = ti_inv_transform_by_quat(contact_data.force, quat_b)
                normal_b_local = ti_inv_transform_by_quat(-contact_data.normal, quat_b)

                output_links[i_b, i_c_] = contact_data.link_a
                output_links[i_b, i_c_ + n_contacts] = contact_data.link_b

                for j in ti.static(range(3)):
                    output_poss[i_b, i_c_, j] = pos_a_local[j]
                    output_poss[i_b, i_c_ + n_contacts, j] = pos_b_local[j]

                # Store |fn| and tangential components (not normalized)
                output_norm_tans[i_b, i_c_, 0] = ti.abs(force_a_local.dot(normal_a_local))
                force_tangential_a = force_a_local - force_a_local.dot(normal_a_local) * normal_a_local
                output_norm_tans[i_b, i_c_, 2] = force_tangential_a[0]
                output_norm_tans[i_b, i_c_, 3] = force_tangential_a[1]

                output_norm_tans[i_b, i_c_ + n_contacts, 0] = ti.abs(force_b_local.dot(normal_b_local))
                force_tangential_b = force_b_local - force_b_local.dot(normal_b_local) * normal_b_local
                output_norm_tans[i_b, i_c_ + n_contacts, 2] = force_tangential_b[0]
                output_norm_tans[i_b, i_c_ + n_contacts, 3] = force_tangential_b[1]
