import numpy as np

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
import genesis.utils.particle as pu

from genesis.ext import pyrender, trimesh
from genesis.ext.pyrender.jit_render import JITRenderer
from genesis.utils.misc import tensor_to_array


class RasterizerContext:
    def __init__(self, options):
        self.show_world_frame = options.show_world_frame
        self.world_frame_size = options.world_frame_size
        self.show_link_frame = options.show_link_frame
        self.link_frame_size = options.link_frame_size
        self.show_cameras = options.show_cameras
        self.shadow = options.shadow
        self.plane_reflection = options.plane_reflection
        self.ambient_light = options.ambient_light
        self.background_color = options.background_color
        self.segmentation_level = options.segmentation_level
        self.lights = options.lights
        self.visualize_mpm_boundary = options.visualize_mpm_boundary
        self.visualize_sph_boundary = options.visualize_sph_boundary
        self.visualize_pbd_boundary = options.visualize_pbd_boundary
        self.particle_size_scale = options.particle_size_scale
        self.contact_force_scale = options.contact_force_scale
        self.render_particle_as = options.render_particle_as
        self.n_rendered_envs = options.n_rendered_envs
        self.env_separate_rigid = options.env_separate_rigid

        self.init_meshes()

    def init_meshes(self):
        self.world_frame_shown = False
        self.link_frame_shown = False
        self.camera_frustum_shown = False

        self.world_frame_mesh = mu.create_frame(
            origin_radius=0.012,
            axis_radius=0.005,
            axis_length=self.world_frame_size,
            head_radius=0.01,
            head_length=0.03,
        )

        self.link_frame_mesh = trimesh.creation.axis(origin_size=0.03, axis_radius=0.025, axis_length=1.0)
        self.link_frame_mesh.visual.face_colors[:, :3] = (0.7 * self.link_frame_mesh.visual.face_colors[:, :3]).astype(
            int
        )
        self.link_frame_mesh.vertices *= self.link_frame_size

    def build(self, scene):
        self.scene = scene
        self.sim = scene.sim
        self.visualizer = scene.visualizer
        self.visualizer.update_visual_states()

        if self.n_rendered_envs is None:
            self.n_rendered_envs = self.sim._B

        # pyrender scene
        self._scene = pyrender.Scene(
            ambient_light=self.ambient_light,
            bg_color=self.background_color,
            n_envs=self.n_rendered_envs,
        )

        self.jit = JITRenderer(self._scene, [], [])

        # nodes
        self.world_frame_node = None
        self.link_frame_nodes = dict()
        self.frustum_nodes = dict()  # nodes camera frustums
        self.rigid_nodes = dict()
        self.static_nodes = dict()  # used across all frames
        self.dynamic_nodes = list()  # nodes that live within single frame
        self.external_nodes = dict()  # nodes added by external user

        self.on_lights()

        if self.show_world_frame:
            self.on_world_frame()
        if self.show_link_frame:
            self.on_link_frame()
        if self.show_cameras:
            self.on_camera_frustum()

        self.on_tool()
        self.on_rigid()
        self.on_avatar()
        self.on_mpm()
        self.on_sph()
        self.on_pbd()
        self.on_fem()

        # segmentation mapping
        self.generate_seg_vars()

    def reset(self):
        self._t = -1

    def create_node(self, *args, **kwargs):
        return pyrender.node.Node(*args, **kwargs)

    def add_node(self, *args, **kwargs):
        return self._scene.add(*args, **kwargs)

    def remove_node(self, *args, **kwargs):
        self._scene.remove_node(*args, **kwargs)

    def add_static_node(self, node_id, *args, **kwargs):
        self.static_nodes[node_id] = self.add_node(*args, **kwargs)

    def add_dynamic_node(self, *args, **kwargs):
        self.dynamic_nodes.append(self.add_node(*args, **kwargs))

    def add_external_node(self, *args, **kwargs):
        node = args[0]
        # Check if the node has a valid name
        if not hasattr(node, "name") or not node.name:
            raise ValueError("Node must have a valid 'name' attribute.")

        # Check if the name is already in use
        if node.name in self.external_nodes:
            raise KeyError(f"A node with the name '{node.name}' already exists.")

        self.external_nodes[node.name] = self.add_node(*args, **kwargs)

    def clear_dynamic_nodes(self):
        for dynamic_node in self.dynamic_nodes:
            self.remove_node(dynamic_node)
        self.dynamic_nodes.clear()

    def clear_external_node(self, node):
        if node.name in self.external_nodes:
            self.remove_node(self.external_nodes[node.name])
            del self.external_nodes[node.name]

    def clear_external_nodes(self):
        for external_node in self.external_nodes.values():
            self.remove_node(external_node)
        self.external_nodes.clear()

    def set_node_pose(self, node, pose):
        self._scene.set_pose(node, pose)

    def update_camera_frustum(self, camera):
        if self.camera_frustum_shown:
            self.set_node_pose(self.frustum_nodes[camera.uid], camera.transform)

    def on_camera_frustum(self):
        if not self.camera_frustum_shown:
            for camera in self.cameras:
                self.frustum_nodes[camera.uid] = self.add_node(
                    pyrender.Mesh.from_trimesh(
                        mu.create_camera_frustum(camera, color=(1.0, 1.0, 1.0, 0.3)),
                        smooth=False,
                    ),
                    pose=camera.transform,
                )
            self.camera_frustum_shown = True

    def off_camera_frustum(self):
        if self.camera_frustum_shown:
            for camera in self.cameras:
                self.remove_node(self.frustum_nodes[camera.uid])
            self.frustum_nodes.clear()
            self.camera_frustum_shown = False

    def on_world_frame(self):
        if not self.world_frame_shown:
            self.world_frame_node = self.add_node(pyrender.Mesh.from_trimesh(self.world_frame_mesh, smooth=True))
            self.world_frame_shown = True

    def off_world_frame(self):
        if self.world_frame_shown:
            self.remove_node(self.world_frame_node)
            self.world_frame_node = None
            self.world_frame_shown = False

    def on_link_frame(self):
        if not self.link_frame_shown:
            if self.sim.rigid_solver.is_active():
                links = self.sim.rigid_solver.links
                links_pos = self.sim.rigid_solver.links_state.pos.to_numpy() + self.scene.envs_offset[None, :]
                links_quat = self.sim.rigid_solver.links_state.quat.to_numpy()

                for link in links:
                    self.link_frame_nodes[link.uid] = self.add_node(
                        pyrender.Mesh.from_trimesh(
                            mesh=self.link_frame_mesh,
                            poses=gu.trans_quat_to_T(links_pos[link.idx], links_quat[link.idx]),
                            env_shared=not self.env_separate_rigid,
                        )
                    )
            self.link_frame_shown = True

    def off_link_frame(self):
        if self.link_frame_shown:
            for node in self.link_frame_nodes.values():
                self.remove_node(node)
            self.link_frame_nodes.clear()
            self.link_frame_shown = False

    def update_link_frame(self, buffer_updates):
        if self.link_frame_shown:
            if self.sim.rigid_solver.is_active():
                links = self.sim.rigid_solver.links

                links_pos = self.sim.rigid_solver.links_state.pos.to_numpy() + self.scene.envs_offset[None, :]
                links_quat = self.sim.rigid_solver.links_state.quat.to_numpy()

                for link in links:
                    link_T = gu.trans_quat_to_T(links_pos[link.idx], links_quat[link.idx])
                    buffer_updates[
                        self._scene.get_buffer_id(
                            self.link_frame_nodes[link.uid],
                            "model",
                        )
                    ] = link_T.transpose([0, 2, 1])

    def on_tool(self):
        if self.sim.tool_solver.is_active():
            for tool_entity in self.sim.tool_solver.entities:
                if tool_entity.mesh is not None:
                    mesh = trimesh.Trimesh(
                        tool_entity.mesh.init_vertices_np,
                        tool_entity.mesh.faces_np.reshape([-1, 3]),
                        tool_entity.mesh.init_vertex_normals_np,
                        process=False,
                    )
                    mesh.visual = mu.surface_uvs_to_trimesh_visual(tool_entity.surface, n_verts=len(mesh.vertices))

                    pose = gu.trans_quat_to_T(tool_entity.init_pos, tool_entity.init_quat)
                    double_sided = tool_entity.surface.double_sided
                    self.add_static_node(
                        tool_entity.uid, pyrender.Mesh.from_trimesh(mesh, double_sided=double_sided), pose=pose
                    )

    def update_tool(self, buffer_updates):
        if self.sim.tool_solver.is_active():
            for tool_entity in self.sim.tool_solver.entities:
                pos = tool_entity.pos[self.sim.cur_substep_local].to_numpy()
                quat = tool_entity.quat[self.sim.cur_substep_local].to_numpy()
                pose = gu.trans_quat_to_T(pos, quat)
                self.set_node_pose(self.static_nodes[tool_entity.uid], pose=pose)

    def set_reflection_mat(self, geom_T):
        height = geom_T[0, 2, 3]
        self.jit.reflection_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, height * 2],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def on_rigid(self):
        if self.sim.rigid_solver.is_active():
            # TODO: support dynamic switching in GUI later
            for rigid_entity in self.sim.rigid_solver.entities:
                if rigid_entity.surface.vis_mode == "visual":
                    geoms = rigid_entity.vgeoms
                    geoms_T = self.sim.rigid_solver._vgeoms_render_T
                else:
                    geoms = rigid_entity.geoms
                    geoms_T = self.sim.rigid_solver._geoms_render_T

                for geom in geoms:
                    if "sdf" in rigid_entity.surface.vis_mode:
                        mesh = geom.get_sdf_trimesh()
                    else:
                        mesh = geom.get_trimesh()
                    geom_T = geoms_T[geom.idx][: self.n_rendered_envs]
                    self.rigid_nodes[geom.uid] = self.add_node(
                        pyrender.Mesh.from_trimesh(
                            mesh=mesh,
                            poses=geom_T,
                            smooth=geom.surface.smooth if "collision" not in rigid_entity.surface.vis_mode else False,
                            double_sided=(
                                geom.surface.double_sided if "collision" not in rigid_entity.surface.vis_mode else False
                            ),
                            is_floor=isinstance(rigid_entity._morph, gs.morphs.Plane),
                            env_shared=not self.env_separate_rigid,
                        )
                    )
                    if isinstance(rigid_entity._morph, gs.morphs.Plane):
                        self.set_reflection_mat(geom_T)

    def update_rigid(self, buffer_updates):
        if self.sim.rigid_solver.is_active():
            for rigid_entity in self.sim.rigid_solver.entities:
                if rigid_entity.surface.vis_mode == "visual":
                    geoms = rigid_entity.vgeoms
                    geoms_T = self.sim.rigid_solver._vgeoms_render_T
                else:
                    geoms = rigid_entity.geoms
                    geoms_T = self.sim.rigid_solver._geoms_render_T

                for geom in geoms:
                    geom_T = geoms_T[geom.idx][: self.n_rendered_envs]
                    buffer_updates[self._scene.get_buffer_id(self.rigid_nodes[geom.uid], "model")] = geom_T.transpose(
                        [0, 2, 1]
                    )
                    if isinstance(rigid_entity._morph, gs.morphs.Plane):
                        self.set_reflection_mat(geom_T)

    def update_contact(self, buffer_updates):
        if self.sim.rigid_solver.is_active():
            batch_idx = 0  # only visualize contact for the first scene
            for i_con in range(self.sim.rigid_solver.collider.n_contacts[batch_idx]):
                contact_data = self.sim.rigid_solver.collider.contact_data[i_con, batch_idx]
                contact_pos = np.array(contact_data.pos) + self.scene.envs_offset[batch_idx]

                if self.sim.rigid_solver.links[contact_data.link_a].visualize_contact:
                    self.draw_contact_arrow(pos=contact_pos, force=-contact_data.force)
                if self.sim.rigid_solver.links[contact_data.link_b].visualize_contact:
                    self.draw_contact_arrow(pos=contact_pos, force=contact_data.force)

    def on_avatar(self):
        if self.sim.avatar_solver.is_active():
            # TODO: support dynamic switching in GUI later
            for avatar_entity in self.sim.avatar_solver.entities:
                if avatar_entity.surface.vis_mode == "visual":
                    geoms = avatar_entity.vgeoms
                    geoms_T = self.sim.avatar_solver._vgeoms_render_T
                else:
                    geoms = avatar_entity.geoms
                    geoms_T = self.sim.avatar_solver._geoms_render_T

                for geom in geoms:
                    if "sdf" in avatar_entity.surface.vis_mode:
                        mesh = geom.get_sdf_trimesh()
                    else:
                        mesh = geom.get_trimesh()
                    geom_T = geoms_T[geom.idx]
                    self.rigid_nodes[geom.uid] = self.add_node(
                        pyrender.Mesh.from_trimesh(
                            mesh=mesh,
                            poses=geom_T,
                            smooth=geom.surface.smooth if "collision" not in avatar_entity.surface.vis_mode else False,
                            double_sided=(
                                geom.surface.double_sided
                                if "collision" not in avatar_entity.surface.vis_mode
                                else False
                            ),
                        )
                    )

    def update_avatar(self, buffer_updates):
        if self.sim.avatar_solver.is_active():
            for avatar_entity in self.sim.avatar_solver.entities:
                if avatar_entity.surface.vis_mode == "visual":
                    geoms = avatar_entity.vgeoms
                    geoms_T = self.sim.avatar_solver._vgeoms_render_T
                else:
                    geoms = avatar_entity.geoms
                    geoms_T = self.sim.avatar_solver._geoms_render_T

                for geom in geoms:
                    geom_T = geoms_T[geom.idx]
                    buffer_updates[self._scene.get_buffer_id(self.rigid_nodes[geom.uid], "model")] = geom_T.transpose(
                        [0, 2, 1]
                    )

    def on_mpm(self):
        if self.sim.mpm_solver.is_active():
            for mpm_entity in self.sim.mpm_solver.entities:
                if mpm_entity.surface.vis_mode == "recon":
                    pass

                elif mpm_entity.surface.vis_mode == "particle":
                    mesh = mu.create_sphere(
                        self.sim.mpm_solver.particle_radius * self.particle_size_scale, subdivisions=1
                    )
                    mesh.visual = mu.surface_uvs_to_trimesh_visual(mpm_entity.surface, n_verts=len(mesh.vertices))

                    tfs = np.tile(np.eye(4), (mpm_entity.n_particles, 1, 1))
                    tfs[:, :3, 3] = mpm_entity.init_particles
                    self.add_static_node(mpm_entity.uid, pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs))

                elif mpm_entity.surface.vis_mode == "visual":
                    # self.add_static_node(mpm_entity.uid, pyrender.Mesh.from_trimesh(mesh, smooth=mpm_entity.surface.smooth))
                    self.add_dynamic_node(
                        pyrender.Mesh.from_trimesh(mpm_entity.vmesh.trimesh, smooth=mpm_entity.surface.smooth)
                    )

            # boundary
            if self.visualize_mpm_boundary:
                self.add_node(
                    pyrender.Mesh.from_trimesh(
                        mu.create_box(
                            bounds=np.array(
                                [
                                    self.sim.mpm_solver.boundary.lower,
                                    self.sim.mpm_solver.boundary.upper,
                                ]
                            ),
                            wireframe=True,
                            color=(1.0, 1.0, 0.0, 1.0),
                        ),
                        smooth=True,
                    )
                )

    def update_mpm(self, buffer_updates):
        if self.sim.mpm_solver.is_active():
            particles_all = self.sim.mpm_solver.particles_render.pos.to_numpy()
            active_all = self.sim.mpm_solver.particles_render.active.to_numpy().astype(bool)
            vverts_all = self.sim.mpm_solver.vverts_render.pos.to_numpy()

            for mpm_entity in self.sim.mpm_solver.entities:
                if mpm_entity.surface.vis_mode == "recon":
                    mesh = pu.particles_to_mesh(
                        positions=particles_all[mpm_entity.particle_start : mpm_entity.particle_end][
                            active_all[mpm_entity.particle_start : mpm_entity.particle_end]
                        ],
                        radius=self.sim.mpm_solver.particle_radius,
                        backend=mpm_entity.surface.recon_backend,
                    )
                    mesh.visual = mu.surface_uvs_to_trimesh_visual(mpm_entity.surface, n_verts=len(mesh.vertices))
                    self.add_dynamic_node(pyrender.Mesh.from_trimesh(mesh, smooth=True))

                elif mpm_entity.surface.vis_mode == "particle":
                    tfs = np.tile(np.eye(4), (mpm_entity.n_particles, 1, 1))
                    tfs[:, :3, 3] = particles_all[mpm_entity.particle_start : mpm_entity.particle_end]

                    buffer_updates[
                        self._scene.get_buffer_id(
                            self.static_nodes[mpm_entity.uid],
                            "model",
                        )
                    ] = tfs.transpose([0, 2, 1])

                elif mpm_entity.surface.vis_mode == "visual":
                    mpm_entity._vmesh.verts = vverts_all[mpm_entity.vvert_start : mpm_entity.vvert_end]
                    self.add_dynamic_node(
                        pyrender.Mesh.from_trimesh(mpm_entity.vmesh.trimesh, smooth=mpm_entity.surface.smooth)
                    )

    def on_sph(self):
        if self.sim.sph_solver.is_active():
            for sph_entity in self.sim.sph_solver.entities:
                if sph_entity.surface.vis_mode == "recon":
                    pass

                elif sph_entity.surface.vis_mode == "particle":
                    mesh = mu.create_sphere(
                        self.sim.sph_solver.particle_radius * self.particle_size_scale, subdivisions=1
                    )
                    mesh.visual = mu.surface_uvs_to_trimesh_visual(sph_entity.surface, n_verts=len(mesh.vertices))

                    tfs = np.tile(np.eye(4), (sph_entity.n_particles, 1, 1))
                    tfs[:, :3, 3] = sph_entity.init_particles
                    self.add_static_node(sph_entity.uid, pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs))

            # boundary
            if self.visualize_sph_boundary:
                self.add_node(
                    pyrender.Mesh.from_trimesh(
                        mu.create_box(
                            bounds=np.array(
                                [
                                    self.sim.sph_solver.boundary.lower,
                                    self.sim.sph_solver.boundary.upper,
                                ]
                            ),
                            wireframe=True,
                            color=(0.0, 1.0, 1.0, 1.0),
                        ),
                        smooth=True,
                    )
                )

    def update_sph(self, buffer_updates):
        if self.sim.sph_solver.is_active():
            particles_all = self.sim.sph_solver.particles_render.pos.to_numpy()
            active_all = self.sim.sph_solver.particles_render.active.to_numpy().astype(bool)

            for sph_entity in self.sim.sph_solver.entities:
                if sph_entity.surface.vis_mode == "recon":
                    mesh = pu.particles_to_mesh(
                        positions=particles_all[sph_entity.particle_start : sph_entity.particle_end][
                            active_all[sph_entity.particle_start : sph_entity.particle_end]
                        ],
                        radius=self.sim.sph_solver.particle_radius,
                        backend=sph_entity.surface.recon_backend,
                    )
                    mesh.visual = mu.surface_uvs_to_trimesh_visual(sph_entity.surface, n_verts=len(mesh.vertices))
                    self.add_dynamic_node(pyrender.Mesh.from_trimesh(mesh, smooth=True))

                elif sph_entity.surface.vis_mode == "particle":
                    tfs = np.tile(np.eye(4), (sph_entity.n_particles, 1, 1))
                    tfs[:, :3, 3] = particles_all[sph_entity.particle_start : sph_entity.particle_end]

                    buffer_updates[
                        self._scene.get_buffer_id(
                            self.static_nodes[sph_entity.uid],
                            "model",
                        )
                    ] = tfs.transpose([0, 2, 1])

    def on_pbd(self):
        if self.sim.pbd_solver.is_active():
            for pbd_entity in self.sim.pbd_solver.entities:
                if pbd_entity.surface.vis_mode == "recon":
                    pass

                elif pbd_entity.surface.vis_mode == "particle":
                    if self.render_particle_as == "sphere":
                        mesh = mu.create_sphere(
                            self.sim.pbd_solver.particle_radius * self.particle_size_scale, subdivisions=1
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(pbd_entity.surface, n_verts=len(mesh.vertices))
                        tfs = np.tile(np.eye(4), (pbd_entity.n_particles, 1, 1))
                        tfs[:, :3, 3] = pbd_entity.init_particles
                        self.add_static_node(pbd_entity.uid, pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs))

                    elif self.render_particle_as == "tet":
                        mesh = mu.create_tets_mesh(
                            pbd_entity.n_particles, self.sim.pbd_solver.particle_radius * self.particle_size_scale
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(pbd_entity.surface, n_verts=len(mesh.vertices))
                        pbd_entity._tets_mesh = mesh
                        self.add_static_node(pbd_entity.uid, pyrender.Mesh.from_trimesh(mesh, smooth=False))

                elif pbd_entity.surface.vis_mode == "visual":
                    self.add_static_node(
                        pbd_entity.uid,
                        pyrender.Mesh.from_trimesh(
                            pbd_entity.vmesh.trimesh,
                            smooth=pbd_entity.surface.smooth,
                            double_sided=pbd_entity._surface.double_sided,
                        ),
                    )

            # boundary
            if self.visualize_pbd_boundary:
                self.add_node(
                    pyrender.Mesh.from_trimesh(
                        mu.create_box(
                            bounds=np.array(
                                [
                                    self.sim.pbd_solver.boundary.lower,
                                    self.sim.pbd_solver.boundary.upper,
                                ]
                            ),
                            wireframe=True,
                            color=(0.0, 1.0, 1.0, 1.0),
                        ),
                        smooth=True,
                    )
                )

    def update_pbd(self, buffer_updates):
        if self.sim.pbd_solver.is_active():
            particles_all = self.sim.pbd_solver.particles_render.pos.to_numpy()
            particles_vel_all = self.sim.pbd_solver.particles_render.vel.to_numpy()
            active_all = self.sim.pbd_solver.particles_render.active.to_numpy().astype(bool)
            vverts_all = self.sim.pbd_solver.vverts_render.pos.to_numpy()

            for pbd_entity in self.sim.pbd_solver.entities:
                if pbd_entity.surface.vis_mode == "recon":
                    mesh = pu.particles_to_mesh(
                        positions=particles_all[pbd_entity.particle_start : pbd_entity.particle_end][
                            active_all[pbd_entity.particle_start : pbd_entity.particle_end]
                        ],
                        radius=self.sim.mpm_solver.particle_radius,
                        backend=pbd_entity.surface.recon_backend,
                    )
                    mesh.visual = mu.surface_uvs_to_trimesh_visual(pbd_entity.surface, n_verts=len(mesh.vertices))
                    self.add_dynamic_node(pyrender.Mesh.from_trimesh(mesh, smooth=True))

                elif pbd_entity.surface.vis_mode == "particle":
                    if self.render_particle_as == "sphere":
                        tfs = np.tile(np.eye(4), (pbd_entity.n_particles, 1, 1))
                        tfs[:, :3, 3] = particles_all[pbd_entity.particle_start : pbd_entity.particle_end]

                        buffer_updates[
                            self._scene.get_buffer_id(
                                self.static_nodes[pbd_entity.uid],
                                "model",
                            )
                        ] = tfs.transpose([0, 2, 1])

                    elif self.render_particle_as == "tet":
                        new_verts = mu.transform_tets_mesh_verts(
                            pbd_entity._tets_mesh.vertices,
                            positions=particles_all[pbd_entity.particle_start : pbd_entity.particle_end],
                            zs=particles_vel_all[pbd_entity.particle_start : pbd_entity.particle_end],
                        )
                        node = self.static_nodes[pbd_entity.uid]
                        update_data = self._scene.reorder_vertices(node, new_verts.astype(np.float32))
                        buffer_updates[self._scene.get_buffer_id(node, "pos")] = update_data
                        normal_data = self.jit.update_normal(node, update_data)
                        if normal_data is not None:
                            buffer_updates[self._scene.get_buffer_id(node, "normal")] = normal_data

                elif pbd_entity.surface.vis_mode == "visual":
                    vverts = vverts_all[pbd_entity.vvert_start : pbd_entity.vvert_end]
                    node = self.static_nodes[pbd_entity.uid]
                    update_data = self._scene.reorder_vertices(node, vverts)
                    buffer_updates[self._scene.get_buffer_id(node, "pos")] = update_data
                    normal_data = self.jit.update_normal(node, update_data)
                    if normal_data is not None:
                        buffer_updates[self._scene.get_buffer_id(node, "normal")] = normal_data

    def on_fem(self):
        if self.sim.fem_solver.is_active():
            vertices_all, triangles_all = self.sim.fem_solver.get_state_render(self.sim.cur_substep_local)
            vertices_all = vertices_all.to_numpy(dtype="float")
            triangles_all = triangles_all.to_numpy(dtype="int").reshape([-1, 3])

            for fem_entity in self.sim.fem_solver.entities:
                if fem_entity.surface.vis_mode == "visual":
                    vertices = vertices_all[fem_entity.v_start : fem_entity.v_start + fem_entity.n_vertices]
                    triangles = (
                        triangles_all[fem_entity.s_start : (fem_entity.s_start + fem_entity.n_surfaces)]
                        - fem_entity.v_start
                    )
                    mesh = trimesh.Trimesh(vertices, triangles, process=False)
                    mesh.visual = mu.surface_uvs_to_trimesh_visual(
                        fem_entity.surface, n_verts=fem_entity.n_surface_vertices
                    )
                    self.add_static_node(
                        fem_entity.uid, pyrender.Mesh.from_trimesh(mesh, double_sided=fem_entity.surface.double_sided)
                    )

    def update_fem(self, buffer_updates):
        if self.sim.fem_solver.is_active():
            vertices_all, triangles_all = self.sim.fem_solver.get_state_render(self.sim.cur_substep_local)
            vertices_all = vertices_all.to_numpy(dtype="float")
            triangles_all = triangles_all.to_numpy(dtype="int").reshape([-1, 3])

            for fem_entity in self.sim.fem_solver.entities:
                if fem_entity.surface.vis_mode == "visual":
                    vertices = vertices_all[fem_entity.v_start : fem_entity.v_start + fem_entity.n_vertices]
                    triangles = (
                        triangles_all[fem_entity.s_start : (fem_entity.s_start + fem_entity.n_surfaces)]
                        - fem_entity.v_start
                    )
                    node = self.static_nodes[fem_entity.uid]
                    update_data = self._scene.reorder_vertices(node, vertices)
                    buffer_updates[self._scene.get_buffer_id(node, "pos")] = update_data

    def on_lights(self):
        for light in self.lights:
            self.add_light(light)

    def draw_debug_line(self, start, end, radius=0.002, color=(1.0, 0.0, 0.0, 1.0)):
        mesh = mu.create_line(start, end, radius, color)
        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_line_{gs.UID()}")
        self.add_external_node(node)
        return node

    def draw_debug_arrow(self, pos, vec=(0, 0, 1), radius=0.006, color=(1.0, 0.0, 0.0, 0.5)):
        length = np.linalg.norm(vec)
        if length > 0:
            mesh = mu.create_arrow(length=length, radius=radius, body_color=color, head_color=color)
            pose = np.eye(4)
            pose[:3, 3] = tensor_to_array(pos)
            pose[:3, :3] = gu.z_to_R(tensor_to_array(vec))
            node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_arrow_{gs.UID()}")
            self.add_external_node(node, pose=pose)
            return node

    def draw_debug_frame(self, T, axis_length=1.0, origin_size=0.015, axis_radius=0.01):
        node = pyrender.Mesh.from_trimesh(
            trimesh.creation.axis(
                origin_size=origin_size,
                axis_radius=axis_radius,
                axis_length=axis_length,
            ),
            name=f"debug_frame_{gs.UID()}",
        )
        self.add_external_node(node, pose=T)
        return node

    def draw_debug_mesh(self, mesh, pos=np.zeros(3), T=None):
        if T is None:
            T = gu.trans_to_T(tensor_to_array(pos))
        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_mesh_{gs.UID()}")
        self.add_external_node(node, pose=T)
        return node

    def draw_contact_arrow(self, pos, radius=0.006, force=(0, 0, 1), color=(0.0, 0.9, 0.8, 1.0)):
        force_vec = tensor_to_array(force) * self.contact_force_scale
        length = np.linalg.norm(force_vec)
        if length > 0:
            mesh = mu.create_arrow(length=length, radius=radius, body_color=color, head_color=color)
            pose = np.eye(4)
            pose[:3, 3] = tensor_to_array(pos)
            pose[:3, :3] = gu.z_to_R(force_vec)
            self.add_dynamic_node(pyrender.Mesh.from_trimesh(mesh), pose=pose)

    def draw_debug_sphere(self, pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        mesh = mu.create_sphere(radius=radius, color=color)
        pose = gu.trans_to_T(tensor_to_array(pos))
        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_sphere_{gs.UID()}", smooth=True)
        self.add_external_node(node, pose=pose)
        return node

    def draw_debug_spheres(self, poss, radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        mesh = mu.create_sphere(radius=radius, color=color)
        poses = gu.trans_to_T(tensor_to_array(poss))
        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_spheres_{gs.UID()}", smooth=True, poses=poses)
        self.add_external_node(node)
        return node

    def draw_debug_box(self, bounds, color=(1.0, 0.0, 0.0, 1.0), wireframe=True, wireframe_radius=0.002):
        bounds = tensor_to_array(bounds)
        mesh = mu.create_box(
            bounds=bounds,
            wireframe=wireframe,
            wireframe_radius=wireframe_radius,
            color=color,
        )
        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_box_{gs.UID()}")
        self.add_external_node(node)
        return node

    def draw_debug_points(self, poss, colors=(1.0, 0.0, 0.0, 0.5)):
        poss = tensor_to_array(poss)
        colors = tensor_to_array(colors)
        if len(colors.shape) == 1:
            colors = np.tile(colors, [len(poss), 1])
        elif len(colors.shape) == 2:
            assert colors.shape[0] == len(poss)

        node = pyrender.Mesh.from_points(poss, name=f"debug_box_{gs.UID()}", colors=colors)
        self.add_external_node(node)
        return node

    def clear_debug_object(self, object):
        self.clear_external_node(object)

    def clear_debug_objects(self):
        self.clear_external_nodes()

    def update(self):
        buffer_updates = dict()

        if self._t >= self.scene._t:  # already updated
            return buffer_updates
        else:
            self._t = self.scene._t

        # clear up all dynamic nodes
        self.clear_dynamic_nodes()

        # update variables not used in simulation
        self.visualizer.update_visual_states()

        self.update_link_frame(buffer_updates)
        self.update_tool(buffer_updates)
        self.update_rigid(buffer_updates)
        self.update_contact(buffer_updates)
        self.update_avatar(buffer_updates)
        self.update_mpm(buffer_updates)
        self.update_sph(buffer_updates)
        self.update_pbd(buffer_updates)
        self.update_fem(buffer_updates)

        return buffer_updates

    def add_light(self, light):
        # light direction is light pose's -z frame
        if light["type"] == "directional":
            z = -np.array(light["dir"])
            R = gu.z_up_to_R(z)
            pose = gu.R_to_T(R)
            self.add_node(pyrender.DirectionalLight(color=light["color"], intensity=light["intensity"]), pose=pose)
        elif light["type"] == "point":
            pose = gu.trans_to_T(np.array(light["pos"]))
            self.add_node(pyrender.PointLight(color=light["color"], intensity=light["intensity"]), pose=pose)
        else:
            gs.raise_exception(f"Unsupported light type: {light['type']}")

    def generate_seg_vars(self):
        # seg_idx: same as entity/link/geom's idx
        # seg_idxc: seg_idx offset by 1, as 0 is reserved for empty space
        # seg_idxc_rgb: colorized seg_idxc internally used by renderer

        # render node to seg_idxc_rgb
        self.seg_node_map = dict()
        seg_idxc_max = 0
        if self.sim.rigid_solver.is_active():
            for rigid_entity in self.sim.rigid_solver.entities:
                if rigid_entity.surface.vis_mode == "visual":
                    geoms = rigid_entity.vgeoms
                else:
                    geoms = rigid_entity.geoms

                for geom in geoms:
                    seg_idx = None
                    if self.segmentation_level == "geom":
                        seg_idx = geom.idx
                        assert False, "geom level segmentation not supported yet"
                    elif self.segmentation_level == "link":
                        seg_idx = geom.link.idx
                    elif self.segmentation_level == "entity":
                        seg_idx = geom.entity.idx
                    else:
                        gs.raise_exception(f"Unsupported segmentation level: {self.segmentation_level}")
                    seg_idxc = self.seg_idx_to_idxc(seg_idx)
                    self.seg_node_map[self.rigid_nodes[geom.uid]] = self.seg_idxc_to_idxc_rgb(seg_idxc)
                    seg_idxc_max = max(seg_idxc_max, seg_idxc)

        # to make sure color is consistent across runs
        prev_seed = np.random.get_state()
        np.random.seed(42)
        self.seg_idxc_to_color = np.random.randint(0, 255, (seg_idxc_max + 1, 3), dtype=np.uint8)
        # background uses black
        self.seg_idxc_to_color[0, :] = 0
        np.random.set_state(prev_seed)

    def seg_idx_to_idxc(self, seg_idx):
        # offset by 1 since (0, 0, 0) is reserved for background (nothing)
        return seg_idx + 1

    def seg_idxc_to_idxc_rgb(self, seg_idxc):
        seg_idxc_rgb = np.array(
            [
                (seg_idxc >> 16) & 0xFF,
                (seg_idxc >> 8) & 0xFF,
                seg_idxc & 0xFF,
            ]
        )
        return seg_idxc_rgb

    def seg_idxc_to_idx(self, seg_idxc):
        # offset by 1 since (0, 0, 0) is reserved for background (nothing)
        return seg_idxc - 1

    def seg_idxc_rgb_arr_to_idxc_arr(self, seg_idxc_rgb_arr):
        # Combine the RGB components into a single integer
        seg_idxc_rgb_arr = seg_idxc_rgb_arr.astype(np.int64, copy=False)
        return seg_idxc_rgb_arr[..., 0] * (256 * 256) + seg_idxc_rgb_arr[..., 1] * 256 + seg_idxc_rgb_arr[..., 2]

    def colorize_seg_idxc_arr(self, seg_idxc_arr):
        return self.seg_idxc_to_color[seg_idxc_arr]

    @property
    def cameras(self):
        return self.visualizer.cameras
