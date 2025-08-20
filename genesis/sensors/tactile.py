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

    Parameters
    ----------
    entity : RigidEntity
        The entity to monitor the contact state of.
    link_idx : int
        The index of the link to which this sensor is attached. If None, defaults to the base link of the entity.
    """

    def __init__(self, entity: RigidEntity, link_idx=None):
        assert isinstance(entity, RigidEntity), "entity must be a RigidEntity"
        self._cached: torch.Tensor | None = None
        self._cached_step: int = -1
        self._entity: RigidEntity = entity
        self._sim = entity._sim
        self._scene = entity._scene
        self._solver = self._sim.rigid_solver
        self.link_idx = link_idx if link_idx is not None else entity.base_link_idx

    def read(self, envs_idx: Optional[List[int]] = None) -> np.ndarray:
        """Returns np.ndarray of shape (batch_size,) of bool values."""
        if self._cached_step == self._sim.cur_step_global:
            return self._cached
        self._cached_step = self._sim.cur_step_global

        all_contacts = self._solver.collider.get_contacts(as_tensor=True, to_torch=True)
        contact_links = torch.cat(
            [
                self._preprocess_kernel_input(all_contacts["link_a"]),
                self._preprocess_kernel_input(all_contacts["link_b"]),
            ],
            dim=1,
        )
        is_contact = (contact_links == self.link_idx).any(dim=1)

        self._cached = tensor_to_array(is_contact)
        return self._cached[envs_idx] if envs_idx is not None else self._cached

    def _preprocess_kernel_input(self, tensor: torch.Tensor) -> torch.Tensor:
        # adds missing batch dim for n_envs=0 and ensures tensor is continuous
        if self._scene.n_envs > 0:
            return tensor.contiguous()
        else:
            return tensor.unsqueeze(0).contiguous()


@ti.data_oriented
class RigidContactForceSensor(RigidContactSensor):
    """
    Sensor that returns contact force (Fx, Fy, Fz) in the associated RigidLink's local frame.

    Parameters
    ----------
    entity : RigidEntity
        The entity to monitor the contact forces of.
    link_idx : int, optional
        The index of the link to which this sensor is attached. If None, defaults to the base link of the entity.
    """

    def read(self, envs_idx: Optional[List[int]] = None) -> np.ndarray:
        """Returns np.ndarray of shape (batch_size, 3) representing the contact force."""

        if self._cached is None:
            self._cached = torch.zeros((self._sim._B, 3), dtype=gs.tc_float, device=gs.device)

        if self._cached_step == self._sim.cur_step_global:
            return self._cached
        self._cached_step = self._sim.cur_step_global

        all_contacts = self._solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_mask = (all_contacts["link_a"] == self.link_idx) | (all_contacts["link_b"] == self.link_idx)

        self._cached.fill_(0.0)

        if link_mask.any():
            self._kernel_get_contacts_forces(
                self._preprocess_kernel_input(all_contacts["force"]),
                self._preprocess_kernel_input(all_contacts["link_a"]),
                self._preprocess_kernel_input(all_contacts["link_b"]),
                self._preprocess_kernel_input(link_mask),
                self._preprocess_kernel_input(self._solver.get_links_quat()),
                self.link_idx,
                self._cached,
            )

        return tensor_to_array(self._cached[envs_idx] if envs_idx is not None else self._cached)

    @ti.kernel
    def _kernel_get_contacts_forces(
        self,
        contact_forces: ti.types.ndarray(),
        link_a: ti.types.ndarray(),
        link_b: ti.types.ndarray(),
        link_mask: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        target_link_idx: ti.i32,
        output: ti.types.ndarray(),
    ):
        for i_b, i_c in ti.ndrange(contact_forces.shape[0], contact_forces.shape[1]):
            if link_mask[i_b, i_c]:
                contact_data_link_a = link_a[i_b, i_c]
                contact_data_link_b = link_b[i_b, i_c]

                quat_a = ti.Vector.zero(ti.f32, 4)
                quat_b = ti.Vector.zero(ti.f32, 4)
                for j in ti.static(range(4)):
                    quat_a[j] = links_quat[i_b, contact_data_link_a, j]
                    quat_b[j] = links_quat[i_b, contact_data_link_b, j]

                force_vec = ti.Vector.zero(ti.f32, 3)
                for j in ti.static(range(3)):
                    force_vec[j] = contact_forces[i_b, i_c, j]

                force_a = ti_inv_transform_by_quat(-force_vec, quat_a)
                force_b = ti_inv_transform_by_quat(force_vec, quat_b)

                if contact_data_link_a == target_link_idx:
                    for j in ti.static(range(3)):
                        output[i_b, j] += force_a[j]
                if contact_data_link_b == target_link_idx:
                    for j in ti.static(range(3)):
                        output[i_b, j] += force_b[j]


@ti.data_oriented
class RigidNormalTangentialForceSensor(RigidContactSensor):
    """
    Sensor that returns (|fn|, |ft|, tx, ty) for contact force normal, tangential shear force, and tangential
    direction as a unit vector in the associated RigidLink's local frame.

    Parameters
    ----------
    entity : RigidEntity
        The entity to monitor the contact forces of.
    link_idx : int, optional
        The index of the link to which this sensor is attached. If None, defaults to the base link of the entity.
    force_eps : float, optional
        The threshold for the tangential force to be considered noise. If the tangential force is less than this value,
        the tangential direction will not be computed.
    """

    def __init__(self, entity: RigidEntity, link_idx=None, force_eps=1e-5):
        super().__init__(entity, link_idx)
        self.force_eps = force_eps

    def read(self, envs_idx: Optional[List[int]] = None) -> np.ndarray:
        """Returns np.ndarray of shape (batch_size, 4) representing the normal and tangential forces."""

        if self._cached is None:
            self._cached = torch.zeros((self._sim._B, 4), dtype=gs.tc_float, device=gs.device)

        if self._cached_step == self._sim.cur_step_global:
            return self._cached
        self._cached_step = self._sim.cur_step_global

        all_contacts = self._solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_mask = (all_contacts["link_a"] == self.link_idx) | (all_contacts["link_b"] == self.link_idx)

        self._cached.fill_(0.0)

        if link_mask.any():
            self._kernel_get_contacts_norm_tan(
                self._preprocess_kernel_input(all_contacts["force"]),
                self._preprocess_kernel_input(all_contacts["normal"]),
                self._preprocess_kernel_input(all_contacts["link_a"]),
                self._preprocess_kernel_input(all_contacts["link_b"]),
                self._preprocess_kernel_input(link_mask),
                self._preprocess_kernel_input(self._solver.get_links_quat()),
                self.force_eps,
                self.link_idx,
                self._cached,
            )

        return tensor_to_array(self._cached[envs_idx] if envs_idx is not None else self._cached)

    @ti.kernel
    def _kernel_get_contacts_norm_tan(
        self,
        contact_forces: ti.types.ndarray(),
        contact_normals: ti.types.ndarray(),
        link_a: ti.types.ndarray(),
        link_b: ti.types.ndarray(),
        link_mask: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        force_eps: ti.f32,
        target_link_idx: ti.i32,
        output: ti.types.ndarray(),
    ):
        for i_b, i_c in ti.ndrange(contact_forces.shape[0], contact_forces.shape[1]):
            if link_mask[i_b, i_c]:
                contact_data_link_a = link_a[i_b, i_c]
                contact_data_link_b = link_b[i_b, i_c]

                quat_a = ti.Vector.zero(ti.f32, 4)
                quat_b = ti.Vector.zero(ti.f32, 4)
                for j in ti.static(range(4)):
                    quat_a[j] = links_quat[i_b, contact_data_link_a, j]
                    quat_b[j] = links_quat[i_b, contact_data_link_b, j]

                force_vec = ti.Vector.zero(ti.f32, 3)
                normal_vec = ti.Vector.zero(ti.f32, 3)
                for j in ti.static(range(3)):
                    force_vec[j] = contact_forces[i_b, i_c, j]
                    normal_vec[j] = contact_normals[i_b, i_c, j]

                force_a_local = ti_inv_transform_by_quat(-force_vec, quat_a)
                normal_a_local = ti_inv_transform_by_quat(normal_vec, quat_a)
                force_b_local = ti_inv_transform_by_quat(force_vec, quat_b)
                normal_b_local = ti_inv_transform_by_quat(-normal_vec, quat_b)

                if contact_data_link_a == target_link_idx:
                    fn_a = ti.abs(force_a_local.dot(normal_a_local))
                    force_tangential_a = force_a_local - force_a_local.dot(normal_a_local) * normal_a_local

                    output[i_b, 0] += fn_a
                    output[i_b, 2] += force_tangential_a[0]
                    output[i_b, 3] += force_tangential_a[1]

                if contact_data_link_b == target_link_idx:
                    fn_b = ti.abs(force_b_local.dot(normal_b_local))
                    force_tangential_b = force_b_local - force_b_local.dot(normal_b_local) * normal_b_local

                    output[i_b, 0] += fn_b
                    output[i_b, 2] += force_tangential_b[0]
                    output[i_b, 3] += force_tangential_b[1]

        # normalize tangential direction vectors
        for i_b in range(output.shape[0]):
            ft_mag = ti.sqrt(output[i_b, 2] * output[i_b, 2] + output[i_b, 3] * output[i_b, 3])
            output[i_b, 1] = ft_mag
            if ft_mag > force_eps:
                output[i_b, 2] /= ft_mag
                output[i_b, 3] /= ft_mag
            else:
                output[i_b, 2] = 0.0
                output[i_b, 3] = 0.0


@ti.data_oriented
class RigidContactGridSensor(RigidContactSensor):
    """
    Sensor that returns ndarray of shape [grid_x, grid_y, grid_z] representing whether or not contact is detected
    in each grid cell based on the associated RigidLink's collision info.

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

    def read(self, envs_idx: Optional[List[int]] = None) -> np.ndarray:
        """Returns np.ndarray of shape (batch_size, grid_x, grid_y, grid_z) of bool values"""

        if self._cached is None:
            self._cached = torch.zeros((self._sim._B, *self.grid_size), dtype=gs.tc_bool, device=gs.device)

        if self._cached_step == self._sim.cur_step_global:
            return self._cached
        self._cached_step = self._sim.cur_step_global

        all_contacts = self._solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_mask = (all_contacts["link_a"] == self.link_idx) | (all_contacts["link_b"] == self.link_idx)

        self._cached.fill_(False)

        if link_mask.any():
            self._kernel_compute_grid(
                self._preprocess_kernel_input(all_contacts["position"]),
                self._preprocess_kernel_input(all_contacts["link_a"]),
                self._preprocess_kernel_input(all_contacts["link_b"]),
                self._preprocess_kernel_input(link_mask),
                self._preprocess_kernel_input(self._solver.get_links_pos()),
                self._preprocess_kernel_input(self._solver.get_links_quat()),
                self.link_idx,
                self.grid_size,
                self._min_bounds,
                self._bounds_size,
                self._cached,
            )

        return tensor_to_array(self._cached[envs_idx] if envs_idx is not None else self._cached)

    @ti.kernel
    def _kernel_compute_grid(
        self,
        contact_positions: ti.types.ndarray(),
        link_a: ti.types.ndarray(),
        link_b: ti.types.ndarray(),
        link_mask: ti.types.ndarray(),
        links_pos: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        target_link_idx: ti.i32,
        grid_size: ti.types.ndarray(),
        min_bounds: ti.types.ndarray(),
        bounds_size: ti.types.ndarray(),
        output_grid: ti.types.ndarray(),
    ):
        for i_b, i_c in ti.ndrange(contact_positions.shape[0], contact_positions.shape[1]):
            if link_mask[i_b, i_c]:
                contact_data_link_a = link_a[i_b, i_c]
                contact_data_link_b = link_b[i_b, i_c]

                target_link = contact_data_link_a if contact_data_link_a == target_link_idx else contact_data_link_b

                rel_pos = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    rel_pos[j] = contact_positions[i_b, i_c, j] - links_pos[target_link, i_b, j]

                quat = ti.Vector.zero(gs.ti_float, 4)
                for j in ti.static(range(4)):
                    quat[j] = links_quat[target_link, i_b, j]

                pos_local = ti_inv_transform_by_quat(rel_pos, quat)

                normalized_pos = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    normalized_pos[j] = (pos_local[j] - min_bounds[j]) / bounds_size[j]

                grid_x = ti.cast(normalized_pos[0] * grid_size[0], ti.i32)
                grid_y = ti.cast(normalized_pos[1] * grid_size[1], ti.i32)
                grid_z = ti.cast(normalized_pos[2] * grid_size[2], ti.i32)

                if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1] and 0 <= grid_z < grid_size[2]:
                    output_grid[i_b, grid_x, grid_y, grid_z] = True

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

    def read(self, envs_idx: Optional[List[int]] = None) -> np.ndarray:
        """Returns np.ndarray of shape (batch_size, grid_x, grid_y, grid_z, 3) representing the contact forces."""
        if self._cached is None:
            self._cached = torch.zeros((self._sim._B, *self.grid_size, 3), dtype=gs.tc_float, device=gs.device)

        if self._cached_step == self._sim.cur_step_global:
            return self._cached
        self._cached_step = self._sim.cur_step_global

        all_contacts = self._solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_mask = (all_contacts["link_a"] == self.link_idx) | (all_contacts["link_b"] == self.link_idx)

        self._cached.fill_(0.0)

        if link_mask.any():
            self._kernel_compute_force_grid(
                self._preprocess_kernel_input(all_contacts["position"]),
                self._preprocess_kernel_input(all_contacts["force"]),
                self._preprocess_kernel_input(all_contacts["link_a"]),
                self._preprocess_kernel_input(all_contacts["link_b"]),
                self._preprocess_kernel_input(link_mask),
                self._preprocess_kernel_input(self._solver.get_links_pos()),
                self._preprocess_kernel_input(self._solver.get_links_quat()),
                self.link_idx,
                self.grid_size,
                self._min_bounds,
                self._bounds_size,
                self._cached,
            )

        return tensor_to_array(self._cached[envs_idx] if envs_idx is not None else self._cached)

    @ti.kernel
    def _kernel_compute_force_grid(
        self,
        contact_positions: ti.types.ndarray(),
        contact_forces: ti.types.ndarray(),
        link_a: ti.types.ndarray(),
        link_b: ti.types.ndarray(),
        link_mask: ti.types.ndarray(),
        links_pos: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        target_link_idx: ti.i32,
        grid_size: ti.types.ndarray(),
        min_bounds: ti.types.ndarray(),
        bounds_size: ti.types.ndarray(),
        output_grid: ti.types.ndarray(),
    ):
        for i_b, i_c in ti.ndrange(contact_positions.shape[0], contact_positions.shape[1]):
            if link_mask[i_b, i_c]:
                contact_data_link_a = link_a[i_b, i_c]
                contact_data_link_b = link_b[i_b, i_c]

                target_link = contact_data_link_a if contact_data_link_a == target_link_idx else contact_data_link_b

                rel_pos = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    rel_pos[j] = contact_positions[i_b, i_c, j] - links_pos[i_b, target_link, j]

                quat = ti.Vector.zero(gs.ti_float, 4)
                for j in ti.static(range(4)):
                    quat[j] = links_quat[i_b, target_link, j]

                pos_local = ti_inv_transform_by_quat(rel_pos, quat)

                force_vec = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    force_vec[j] = contact_forces[i_b, i_c, j]

                force_local = ti_inv_transform_by_quat(
                    -force_vec if contact_data_link_a == target_link_idx else force_vec, quat
                )

                normalized_pos = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    if bounds_size[j] > 0:
                        normalized_pos[j] = ti.math.clamp((pos_local[j] - min_bounds[j]) / bounds_size[j], 0.0, 1.0)
                    else:
                        normalized_pos[j] = 0.5

                grid_x = ti.min(ti.cast(normalized_pos[0] * grid_size[0], ti.i32), grid_size[0] - 1)
                grid_y = ti.min(ti.cast(normalized_pos[1] * grid_size[1], ti.i32), grid_size[1] - 1)
                grid_z = ti.min(ti.cast(normalized_pos[2] * grid_size[2], ti.i32), grid_size[2] - 1)

                if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1] and 0 <= grid_z < grid_size[2]:
                    for j in ti.static(range(3)):
                        output_grid[i_b, grid_x, grid_y, grid_z, j] += force_local[j]


@ti.data_oriented
class RigidNormalTangentialForceGridSensor(RigidContactForceGridSensor):
    """
    Sensor that returns ndarray of shape (grid_x, grid_y, grid_z, 4) representing the accumulated contact forces.
    Each grid cell contains (|fn|, |ft|, tx, ty) for contact force normal, tangential shear force, and tangential
    direction as a unit vector in the associated RigidLink's local frame.

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
    force_eps : float, optional
        When the tangential force is below this threshold, it is considered noise and the tangential direction is
        not computed.
    """

    def __init__(self, entity: RigidEntity, link_idx=None, grid_size=(1, 1, 1), force_eps=1e-5):
        super().__init__(entity, link_idx, grid_size)
        self.force_eps = force_eps

    def read(self, envs_idx: Optional[List[int]] = None) -> np.ndarray:
        """Returns np.ndarray of shape (batch_size, grid_x, grid_y, grid_z, 4) representing the contact forces."""
        if self._cached is None:
            self._cached = torch.zeros((self._sim._B, *self.grid_size, 4), dtype=gs.tc_float, device=gs.device)

        if self._cached_step == self._sim.cur_step_global:
            return self._cached
        self._cached_step = self._sim.cur_step_global

        all_contacts = self._solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_mask = (all_contacts["link_a"] == self.link_idx) | (all_contacts["link_b"] == self.link_idx)

        self._cached.fill_(0.0)

        if link_mask.any():
            self._kernel_compute_norm_tan_grid(
                self._preprocess_kernel_input(all_contacts["position"]),
                self._preprocess_kernel_input(all_contacts["force"]),
                self._preprocess_kernel_input(all_contacts["normal"]),
                self._preprocess_kernel_input(all_contacts["link_a"]),
                self._preprocess_kernel_input(all_contacts["link_b"]),
                self._preprocess_kernel_input(link_mask),
                self._preprocess_kernel_input(self._solver.get_links_pos()),
                self._preprocess_kernel_input(self._solver.get_links_quat()),
                self.link_idx,
                self.grid_size,
                self._min_bounds,
                self._bounds_size,
                self.force_eps,
                self._cached,
            )

        return tensor_to_array(self._cached[envs_idx] if envs_idx is not None else self._cached)

    @ti.kernel
    def _kernel_compute_norm_tan_grid(
        self,
        contact_positions: ti.types.ndarray(),
        contact_forces: ti.types.ndarray(),
        contact_normals: ti.types.ndarray(),
        link_a: ti.types.ndarray(),
        link_b: ti.types.ndarray(),
        link_mask: ti.types.ndarray(),
        links_pos: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        target_link_idx: ti.i32,
        grid_size: ti.types.ndarray(),
        min_bounds: ti.types.ndarray(),
        bounds_size: ti.types.ndarray(),
        force_eps: ti.f32,
        output_grid: ti.types.ndarray(),
    ):
        for i_b, i_c in ti.ndrange(contact_positions.shape[0], contact_positions.shape[1]):
            if link_mask[i_b, i_c]:
                contact_data_link_a = link_a[i_b, i_c]
                contact_data_link_b = link_b[i_b, i_c]

                target_link = contact_data_link_a if contact_data_link_a == target_link_idx else contact_data_link_b

                rel_pos = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    rel_pos[j] = contact_positions[i_b, i_c, j] - links_pos[i_b, target_link, j]

                quat = ti.Vector.zero(gs.ti_float, 4)
                for j in ti.static(range(4)):
                    quat[j] = links_quat[i_b, target_link, j]

                pos_local = ti_inv_transform_by_quat(rel_pos, quat)

                force_vec = ti.Vector.zero(gs.ti_float, 3)
                normal_vec = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    force_vec[j] = contact_forces[i_b, i_c, j]
                    normal_vec[j] = contact_normals[i_b, i_c, j]

                force_local = ti_inv_transform_by_quat(
                    -force_vec if contact_data_link_a == target_link_idx else force_vec, quat
                )
                normal_local = ti_inv_transform_by_quat(
                    normal_vec if contact_data_link_a == target_link_idx else -normal_vec, quat
                )

                normalized_pos = ti.Vector.zero(gs.ti_float, 3)
                for j in ti.static(range(3)):
                    if bounds_size[j] > 0:
                        normalized_pos[j] = ti.math.clamp((pos_local[j] - min_bounds[j]) / bounds_size[j], 0.0, 1.0)
                    else:
                        normalized_pos[j] = 0.5

                grid_x = ti.min(ti.cast(normalized_pos[0] * grid_size[0], ti.i32), grid_size[0] - 1)
                grid_y = ti.min(ti.cast(normalized_pos[1] * grid_size[1], ti.i32), grid_size[1] - 1)
                grid_z = ti.min(ti.cast(normalized_pos[2] * grid_size[2], ti.i32), grid_size[2] - 1)

                if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1] and 0 <= grid_z < grid_size[2]:
                    # compute normal and tangential components
                    fn = ti.abs(force_local.dot(normal_local))
                    force_tangential = force_local - force_local.dot(normal_local) * normal_local

                    output_grid[i_b, grid_x, grid_y, grid_z, 0] += fn
                    output_grid[i_b, grid_x, grid_y, grid_z, 2] += force_tangential[0]
                    output_grid[i_b, grid_x, grid_y, grid_z, 3] += force_tangential[1]

        # normalize tangential direction vectors for each grid cell
        for i_b, i_x, i_y, i_z in ti.ndrange(
            output_grid.shape[0], output_grid.shape[1], output_grid.shape[2], output_grid.shape[3]
        ):
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
