import numpy as np
import taichi as ti
import torch
from scipy.spatial import KDTree

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
import genesis.utils.particle as pu
from genesis.engine.states.cache import QueriedStates
from genesis.utils.misc import to_gs_tensor
from genesis.ext import trimesh

from .base_entity import Entity


@ti.data_oriented
class ParticleEntity(Entity):
    """
    Base class for particle-based entity.
    """

    def __init__(
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        particle_size,
        idx,
        particle_start,
        vvert_start=None,
        vface_start=None,
        need_skinning=True,
    ):

        super().__init__(idx, scene, morph, solver, material, surface)

        self._particle_size = particle_size
        self._particle_start = particle_start

        self._need_skinning = need_skinning
        if need_skinning:
            self._vvert_start = vvert_start
            self._vface_start = vface_start
        else:
            self._vvert_start = -1
            self._vface_start = -1

        # visual mesh
        if isinstance(self._morph, gs.options.morphs.MeshSet):
            self._vmesh = gs.Mesh.from_morph_surface(self.morph, self.surface)
            self._surface = self._vmesh[0].surface

        elif isinstance(self._morph, (gs.options.morphs.Primitive, gs.options.morphs.Mesh)):
            self._vmesh = gs.Mesh.from_morph_surface(self.morph, self.surface)
            if isinstance(self._vmesh, list):
                if len(self._vmesh) > 1:
                    gs.raise_exception("Mesh file with multiple sub-meshes are not supported.")
                else:
                    self._vmesh = self._vmesh[0]
            self._surface = self._vmesh.surface

        else:
            surface.update_texture()
            self._vmesh = None

        self.sample()

        self.init_tgt_vars()
        self.init_ckpt()
        self._queried_states = QueriedStates()

        self.active = False  # This attribute is only used in forward pass. It should NOT be used during backward pass.

    def init_sampler(self):
        self.sampler = self._material.sampler

        valid = True
        if self.sampler == "regular":
            pass
        elif self.sampler == "random":
            pass
        elif "pbs" in self.sampler:
            splits = self.sampler.split("-")
            if len(splits) == 1:  # using default sdf_res=32
                self.sampler += "-32"
            elif len(splits) == 2 and splits[0] == "pbs" and splits[1].isnumeric():
                pass
            else:
                valid = False
        else:
            valid = False

        if not valid:
            gs.raise_exception(
                f"Only one of the following samplers is supported: [`regular`, `random`, `pbs`, `pbs-sdf_res`]. Got: {self.sampler}."
            )

    def init_tgt_keys(self):
        self._tgt_keys = ["vel", "pos", "act"]

    def _add_to_solver(self):
        if isinstance(self._morph, gs.options.morphs.Nowhere):
            # for nowhere entity, it will be activated when particles are added later.
            # Note that self.active is not informative for nowhere entity, and such entity is not supported in differentiable mode.
            self.active = False
        else:
            self.active = True

        self._add_to_solver_()

        if self._need_skinning:
            self._add_vverts_to_solver()

        # TODO: make more elegant when merge with PBD
        if hasattr(self, "set_muscle_group") and hasattr(self, "mesh_set_group_ids"):
            self.set_muscle_group(self.mesh_set_group_ids)

    def _add_vverts_to_solver(self):
        # compute supports for rendering vverts using neighboring particles
        # need to put this kernel here because type hint for P_invs requires access to self.solver._n_vvert_supports
        @ti.kernel
        def _kernel_add_vverts_to_solver(
            vverts: ti.types.ndarray(dtype=ti.math.vec3),
            particles: ti.types.ndarray(dtype=ti.math.vec3),
            P_invs: ti.types.ndarray(dtype=ti.types.matrix(self.solver._n_vvert_supports - 1, 3, gs.ti_float)),
            support_idxs_local: ti.types.ndarray(dtype=ti.i32),
        ):
            for i_vv_ in range(self.n_vverts):
                i_vv = i_vv_ + self._vvert_start
                v = vverts[i_vv_]
                P_inv = P_invs[i_vv_]
                b = P_inv @ (v - particles[support_idxs_local[i_vv_, self.solver._n_vvert_supports - 1]])
                for j in range(self.solver._n_vvert_supports - 1):
                    self.solver.vverts_info.support_weights[i_vv][j] = b[j]
                self.solver.vverts_info.support_weights[i_vv][self.solver._n_vvert_supports - 1] = 1.0 - b.sum()
                for j in range(self.solver._n_vvert_supports):
                    self.solver.vverts_info.support_idxs[i_vv][j] = support_idxs_local[i_vv_, j] + self._particle_start

        kdtree = KDTree(self._particles)
        _, support_idxs = kdtree.query(self._vverts, k=self.solver._n_vvert_supports)
        support_idxs = np.clip(support_idxs, 0, len(self._particles) - 1)
        all_ps = self._particles[support_idxs]
        Ps = np.stack([all_ps[:, i, :] for i in range(self.solver._n_vvert_supports - 1)], axis=2) - np.expand_dims(
            all_ps[:, -1, :], axis=2
        )

        _kernel_add_vverts_to_solver(
            vverts=self._vverts.astype(gs.np_float),
            particles=self._particles.astype(gs.np_float),
            P_invs=np.linalg.pinv(Ps).astype(gs.np_float),
            support_idxs_local=support_idxs.astype(gs.np_int),
        )

    def sample(self):
        self.init_sampler()

        if isinstance(self._morph, gs.options.morphs.MeshSet):
            particles = []
            for i, file in enumerate(self._morph.files):
                morph_i = self._morph.model_copy()
                morph_i.file = file
                morph_i.pos = morph_i.poss[i]
                morph_i.euler = morph_i.eulers[i]
                mesh_i = morph_i.file.copy()
                mesh_i.vertices = mesh_i.vertices * morph_i.scale
                if "pbs" in self.sampler:
                    particles_i = pu.trimesh_to_particles_pbs(
                        mesh=mesh_i,
                        p_size=self._particle_size,
                        sampler=self.sampler,
                    )
                else:
                    particles_i = pu.trimesh_to_particles_simple(
                        mesh=mesh_i,
                        p_size=self._particle_size,
                        sampler=self.sampler,
                    )
                particles_i += np.array(morph_i.pos)
                particles.append(particles_i)

        elif isinstance(self._morph, (gs.options.morphs.Primitive, gs.options.morphs.Mesh)):
            particles = self._vmesh.particlize(self._particle_size, self.sampler)

        elif isinstance(self._morph, gs.options.morphs.Nowhere):
            particles = pu.nowhere_particles(self._morph.n_particles)

        else:
            gs.raise_exception(f"Unsupported morph: {self._morph}.")

        if not isinstance(self._morph, gs.options.morphs.MeshSet) and particles.shape[0] == 0:
            gs.raise_exception("Entity has zero particles.")

        if isinstance(self._morph, gs.options.morphs.Nowhere):
            origin = gu.nowhere().astype(gs.np_float)
            self._vverts = np.array([])
            self._vfaces = np.array([])

        elif isinstance(self._morph, gs.options.morphs.MeshSet):
            for i in range(len(self._morph.files)):
                pos_i = np.array(self._morph.poss[i])
                quat_i = np.array(gu.euler_to_quat(self._morph.eulers[i]))
                self._vmesh[i].apply_transform(gu.trans_quat_to_T(pos_i, quat_i))

                # NOTE: particles are transformed already
                # particles[i] = gu.transform_by_trans_quat(particles[i], pos_i, quat_i)

            self.mesh_set_group_ids = np.concatenate(
                [np.ones((v.shape[0],), dtype=gs.np_int) * i for i, v in enumerate(particles)]
            )
            particles = np.concatenate(particles)
            if not self._solver.boundary.is_inside(particles):  # HACK no check
                gs.raise_exception(
                    f"Entity has particles outside solver boundary. Note that for MPMSolver, boundary is slightly tighter than the specified domain due to safety padding.\n\nCurrent boundary:\n{self._solver.boundary}\n\nEntity to be added:\nmin: {particles.min(0)}\nmax: {particles.max(0)}\n"
                )

            combined_verts, combined_faces = trimesh.util.append_faces(
                [v.verts for v in self._vmesh],
                [v.faces for v in self._vmesh],
            )
            combined_vert_normals = np.concatenate([v.trimesh.vertex_normals for v in self._vmesh])
            combined_tmesh = trimesh.Trimesh(
                vertices=combined_verts,
                faces=combined_faces,
                vertex_normals=combined_vert_normals,
            )
            self._vmesh = mu.trimesh_to_mesh(combined_tmesh, 1, self._surface)
            if self._need_skinning:
                self._vverts = np.array(self._vmesh.verts)
                self._vfaces = np.array(self._vmesh.faces)
            else:
                self._vverts = np.array([])
                self._vfaces = np.array([])
            origin = np.mean(self._morph.poss)

        else:
            # transform vmesh
            self._vmesh.apply_transform(gu.trans_quat_to_T(np.array(self._morph.pos), np.array(self._morph.quat)))
            # transform particles
            origin = np.array(self._morph.pos, dtype=gs.np_float)
            particles = gu.transform_by_trans_quat(particles, np.array(self._morph.pos), np.array(self._morph.quat))
            # rotate

            if not self._solver.boundary.is_inside(particles):
                gs.raise_exception(
                    f"Entity has particles outside solver boundary. Note that for MPMSolver, boundary is slightly tighter than the specified domain due to safety padding.\n\nCurrent boundary:\n{self._solver.boundary}\n\nEntity to be added:\nmin: {particles.min(0)}\nmax: {particles.max(0)}\n"
                )

            if self._need_skinning:
                self._vverts = np.array(self._vmesh.verts)
                self._vfaces = np.array(self._vmesh.faces)
            else:
                self._vverts = np.array([])
                self._vfaces = np.array([])

        self._particles = particles.astype(gs.np_float)
        self._init_particles_offset = (gs.tensor(particles).contiguous() - gs.tensor(origin)).contiguous()
        self._n_particles = len(self._particles)

        gs.logger.info(f"Sampled ~~<{self._n_particles:,}>~~ particles.")

    def init_tgt_vars(self):
        # temp variable to store targets for next step
        self._tgt = dict()
        self._tgt_buffer = dict()
        self.init_tgt_keys()

        for key in self._tgt_keys:
            self._tgt[key] = None
            self._tgt_buffer[key] = list()

    def init_ckpt(self):
        self._ckpt = dict()

    def save_ckpt(self, ckpt_name):
        if ckpt_name not in self._ckpt:
            self._ckpt[ckpt_name] = {
                "_tgt_buffer": dict(),
            }

        for key in self._tgt_keys:
            self._ckpt[ckpt_name]["_tgt_buffer"][key] = list(self._tgt_buffer[key])
            self._tgt_buffer[key].clear()

    def load_ckpt(self, ckpt_name):
        for key in self._tgt_keys:
            self._tgt_buffer[key] = list(self._ckpt[ckpt_name]["_tgt_buffer"][key])

    def reset_grad(self):
        for key in self._tgt_keys:
            self._tgt_buffer[key].clear()
        self._queried_states.clear()

    def set_velocity(self, vel):
        """
        Accepted tensor shape: (3,) or (self._n_particles, 3).
        """
        self._assert_active()
        if self.sim.requires_grad:
            gs.logger.warning(
                "Manually setting particle velocities. This is not recommended and could break gradient flow."
            )

        vel = to_gs_tensor(vel)

        if len(vel.shape) == 1:
            assert vel.shape == (3,)
            self._tgt["vel"] = torch.tile(vel, [self._n_particles, 1])

        elif len(vel.shape) == 2:
            assert vel.shape == (self._n_particles, 3)
            self._tgt["vel"] = vel

        else:
            gs.raise_exception("Tensor shape not supported.")

    def set_position(self, pos):
        """
        Accepted tensor shape: (3,) for COM position or (self._n_particles, 3) for particle-wise position.
        When COM position is given, the particles will be restored to the entity's initial shape.
        """
        self._assert_active()
        if self.sim.requires_grad:
            gs.logger.warning(
                "Manually setting particle positions. This is not recommended and could break gradient flow. This also resets particle stress and velocities."
            )

        pos = to_gs_tensor(pos)

        if len(pos.shape) == 1:
            assert pos.shape == (3,)
            self._tgt["pos"] = self._init_particles_offset + pos

        elif len(pos.shape) == 2:
            assert pos.shape == (self._n_particles, 3)
            self._tgt["pos"] = pos

        else:
            gs.raise_exception("Tensor shape not supported.")

    def get_mass(self):
        mass = np.zeros(1, dtype=gs.np_float)
        self._kernel_get_mass(mass)
        return mass[0]

    def deactivate(self):
        gs.logger.info(f"{self.__class__.__name__} <{self._uid}> deactivated.")
        self._tgt["act"] = gs.INACTIVE
        self.active = False

    def activate(self):
        gs.logger.info(f"{self.__class__.__name__} <{self._uid}> activated.")
        self._tgt["act"] = gs.ACTIVE
        self.active = True

    def _assert_active(self):
        if not self.active:
            gs.raise_exception(f"{self.__class__.__name__} is inactive. Call `entity.activate()` first.")

    def process_input(self, in_backward=False):
        if in_backward:
            # use negative index because buffer length might not be full
            index = self._sim.cur_step_local - self._sim.max_steps_local
            for key in self._tgt_keys:
                self._tgt[key] = self._tgt_buffer[key][index]

        else:
            for key in self._tgt_keys:
                self._tgt_buffer[key].append(self._tgt[key])

        # set_pos followed by set_vel, because set_pos resets velocity.
        if self._tgt["pos"] is not None:
            self._tgt["pos"].assert_contiguous()
            self._tgt["pos"].assert_sceneless()
            self.set_pos(self._sim.cur_substep_local, self._tgt["pos"])

        if self._tgt["vel"] is not None:
            self._tgt["vel"].assert_contiguous()
            self._tgt["vel"].assert_sceneless()
            self.set_vel(self._sim.cur_substep_local, self._tgt["vel"])

        if self._tgt["act"] is not None:
            assert self._tgt["act"] in [gs.ACTIVE, gs.INACTIVE]
            self.set_active(self._sim.cur_substep_local, self._tgt["act"])

        for key in self._tgt_keys:
            self._tgt[key] = None

    def process_input_grad(self):
        _tgt_vel = self._tgt_buffer["vel"].pop()
        _tgt_pos = self._tgt_buffer["pos"].pop()

        if _tgt_vel is not None and _tgt_vel.requires_grad:
            _tgt_vel._backward_from_ti(self.set_vel_grad, self._sim.cur_substep_local)

        if _tgt_pos is not None and _tgt_pos.requires_grad:
            _tgt_pos._backward_from_ti(self.set_pos_grad, self._sim.cur_substep_local)

        if _tgt_vel is not None or _tgt_pos is not None:
            # manually zero the grad since manually setting state breaks gradient flow
            self.clear_grad(self._sim.cur_substep_local)

    def collect_output_grads(self):
        """
        Collect gradients from external queried states.
        """
        if self._sim.cur_step_global in self._queried_states:
            # one step could have multiple states
            for state in self._queried_states[self._sim.cur_step_global]:
                self.add_grad_from_state(state)

    # ------------------------------------------------------------------------------------
    # ---------------------------------- io & control ------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_particles(self):
        particles = np.empty((self.n_particles, 3), dtype=gs.np_float)
        self._kernel_get_particles(particles)
        return particles

    @property
    def uid(self):
        return self._uid

    @property
    def idx(self):
        return self._idx

    @property
    def morph(self):
        return self._morph

    @property
    def vmesh(self):
        return self._vmesh

    @property
    def n_vverts(self):
        return len(self._vverts)

    @property
    def n_vfaces(self):
        return len(self._vfaces)

    @property
    def n_particles(self):
        return self._n_particles

    @property
    def particle_start(self):
        return self._particle_start

    @property
    def particle_end(self):
        return self._particle_start + self._n_particles

    @property
    def vvert_start(self):
        return self._vvert_start

    @property
    def vvert_end(self):
        return self._vvert_start + self.n_vverts

    @property
    def vface_start(self):
        return self._vface_start

    @property
    def vface_end(self):
        return self._vface_start + self.n_vfaces

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def init_particles(self):
        return self._particles

    @property
    def material(self):
        return self._material

    @property
    def surface(self):
        return self._surface
