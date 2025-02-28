import numpy as np

from OpenGL.GL import *

from .material import MetallicRoughnessMaterial, SpecularGlossinessMaterial
from .light import DirectionalLight, PointLight
from .constants import RenderFlags, MAX_N_LIGHTS

RenderFlags_DEPTH_ONLY = RenderFlags.DEPTH_ONLY
RenderFlags_SEG = RenderFlags.SEG
RenderFlags_FLIP_WIREFRAME = RenderFlags.FLIP_WIREFRAME
RenderFlags_ALL_WIREFRAME = RenderFlags.ALL_WIREFRAME
RenderFlags_SKIP_CULL_FACES = RenderFlags.SKIP_CULL_FACES
RenderFlags_SHADOWS_DIRECTIONAL = RenderFlags.SHADOWS_DIRECTIONAL
RenderFlags_SHADOWS_POINT = RenderFlags.SHADOWS_POINT
RenderFlags_SKIP_FLOOR = RenderFlags.SKIP_FLOOR
RenderFlags_REFLECTIVE_FLOOR = RenderFlags.REFLECTIVE_FLOOR
RenderFlags_FLAT = RenderFlags.FLAT


# Helper methods for uniform setting
def set_uniform_matrix_4fv(pid, name, value):
    """Helper method for setting 4x4 matrix uniforms"""
    loc = glGetUniformLocation(pid, name)
    if loc >= 0:
        glUniformMatrix4fv(loc, 1, GL_TRUE, value)
    else:
        print(f"uniform not found: {name}")


def set_uniform_1i(pid, name, value):
    """Helper method for setting integer uniforms"""
    loc = glGetUniformLocation(pid, name)
    if loc >= 0:
        glUniform1i(loc, value)
    else:
        print(f"uniform not found: {name}")


def set_uniform_1f(pid, name, value):
    """Helper method for setting float uniforms"""
    loc = glGetUniformLocation(pid, name)
    if loc >= 0:
        glUniform1f(loc, value)
    else:
        print(f"uniform not found: {name}")


def set_uniform_2f(pid, name, value1, value2):
    """Helper method for setting vec2 uniforms"""
    loc = glGetUniformLocation(pid, name)
    if loc >= 0:
        glUniform2f(loc, value1, value2)
    else:
        print(f"uniform not found: {name}")


def set_uniform_3fv(pid, name, value):
    """Helper method for setting vec3 uniforms"""
    loc = glGetUniformLocation(pid, name)
    if loc >= 0:
        glUniform3fv(loc, 1, value)
    else:
        print(f"uniform not found: {name}")


def set_uniform_4fv(pid, name, value):
    """Helper method for setting vec4 uniforms"""
    loc = glGetUniformLocation(pid, name)
    if loc >= 0:
        glUniform4fv(loc, 1, value)
    else:
        print(f"uniform not found: {name}")


def bind_lighting(pid, flags, light, shadow_map, light_matrix, ambient_light):
    """define another function for binding"""
    n = len(light)
    set_uniform_3fv(pid, "ambient_light", ambient_light)
    n_dir, n_pt, n_spot = 0, 0, 0
    active_texture = 0
    cube_tex = -1

    for i in range(n):
        if abs(light[i, 7] - 0.0) < 0.1:  # Directional Light
            b = f"directional_lights[{n_dir}]."
            set_uniform_3fv(pid, b + "color", light[i, :3])
            set_uniform_3fv(pid, b + "direction", light[i, 3:6])
            set_uniform_1f(pid, b + "intensity", light[i, 6])
            if flags & RenderFlags_SHADOWS_DIRECTIONAL:
                glActiveTexture(GL_TEXTURE0 + active_texture)
                glBindTexture(GL_TEXTURE_2D, shadow_map[i])
                set_uniform_1i(pid, b + "shadow_map", active_texture)
                set_uniform_matrix_4fv(pid, b + "light_matrix", light_matrix[i, 0])
                active_texture += 1
            n_dir += 1
        elif abs(light[i, 7] - 1.0) < 0.1:  # Point Light
            b = f"point_lights[{n_pt}]."
            set_uniform_3fv(pid, b + "color", light[i, :3])
            set_uniform_3fv(pid, b + "position", light[i, 3:6])
            set_uniform_1f(pid, b + "intensity", light[i, 6])
            if flags & RenderFlags_SHADOWS_POINT:
                if active_texture == 0:
                    active_texture += 1
                glActiveTexture(GL_TEXTURE0 + active_texture)
                glBindTexture(GL_TEXTURE_CUBE_MAP, shadow_map[i])
                set_uniform_1i(pid, b + "shadow_map", active_texture)
                cube_tex = active_texture
                active_texture += 1
            n_pt += 1

    set_uniform_1i(pid, "n_directional_lights", n_dir)
    set_uniform_1i(pid, "n_spot_lights", n_spot)
    set_uniform_1i(pid, "n_point_lights", n_pt)

    if flags & RenderFlags_SHADOWS_POINT:
        if cube_tex < 0:
            if active_texture == 0:
                active_texture += 1
            cube_tex = active_texture
            active_texture += 1

        for i in range(n_pt, MAX_N_LIGHTS):
            b = f"point_lights[{i}]."
            set_uniform_1i(pid, b + "shadow_map", cube_tex)

        return active_texture


class SimpleNonJITRenderer:
    def __init__(self, scene, node_list, primitive_list):
        self.set_primitive(scene, node_list, primitive_list)
        self.set_light(scene, scene.light_nodes, scene.ambient_light)
        # self.gen_func_ptr()
        self.reflection_mat = np.identity(4, np.float32)

    def update(self, scene):
        """Update the renderer's state based on the scene.

        Parameters
        ----------
        scene : Scene
            The scene to update from
        """
        if scene.meshes_updated:
            # Complete rebuild of node and primitive lists
            node_list, primitive_list = [], []
            for node in scene.sorted_mesh_nodes():
                mesh = node.mesh
                if not mesh.is_visible:
                    continue
                for primitive in mesh.primitives:
                    node_list.append(node)
                    primitive_list.append(primitive)

            # Set new primitives and their properties
            self.set_primitive(scene, node_list, primitive_list)
            scene.reset_meshes_updated()
        else:
            # Just update poses
            for i, node in enumerate(self.node_list):
                self.pose[i] = scene.get_pose(node)

        # Update lights
        self.set_light(scene, scene.light_nodes, scene.ambient_light)

    def set_light(self, scene, light_nodes, ambient_light):
        """Set lighting information for the renderer.

        Parameters
        ----------
        scene : Scene
            The scene containing the lights
        light_nodes : set of Node
            The nodes containing lights
        ambient_light : ndarray
            The ambient light color
        """
        self.light_list = light_nodes

        n = len(light_nodes)
        if n > MAX_N_LIGHTS:
            raise ValueError("Number of lights exceeds limit.")

        # Initialize arrays for light data
        # directional: color <- 3, direction <- 3, intensity, 0
        # point:      color <- 3, position <- 3, intensity, 1
        self.light = np.zeros((n, 8), np.float32)
        self.shadow_map = np.zeros(n, np.int32)
        self.light_matrix = np.zeros((n, 6, 4, 4), np.float32)

        self.ambient_light = np.array(ambient_light, np.float32)

        for i, node in enumerate(light_nodes):
            light = node.light
            pose = scene.get_pose(node)
            position = pose[:3, 3]
            direction = -pose[:3, 2]

            if isinstance(light, DirectionalLight):
                self.light[i, :3] = light.color
                self.light[i, 3:6] = direction
                self.light[i, 6] = light.intensity
                self.light[i, 7] = 0

                if light.shadow_texture:
                    self.shadow_map[i] = light.shadow_texture._texid

                pose = pose.copy()
                camera = light._get_shadow_camera(scene.scale)
                P = camera.get_projection_matrix()
                c = scene.centroid
                loc = c - direction * scene.scale
                pose[:3, 3] = loc
                V = np.linalg.inv(pose)  # V maps from world to camera
                self.light_matrix[i, 0] = P.dot(V)

            elif isinstance(light, PointLight):
                self.light[i, :3] = light.color
                self.light[i, 3:6] = position
                self.light[i, 6] = light.intensity
                self.light[i, 7] = 1

                if light.shadow_texture:
                    self.shadow_map[i] = light.shadow_texture._texid

                camera = light._get_shadow_camera(scene.scale)
                projection = camera.get_projection_matrix()
                view = light._get_view_matrices(position)
                self.light_matrix[i] = projection @ view

            else:
                raise TypeError("Light type not supported yet.")

    def set_primitive(self, scene, node_list, primitive_list):
        """Set primitives for rendering.

        Parameters
        ----------
        scene : Scene
            The scene containing the primitives
        node_list : list of Node
            List of nodes containing the primitives
        primitive_list : list of Primitive
            List of primitives to render
        """
        # print(f"set_primitive: {len(primitive_list)}")
        self.node_list = node_list
        self.primitive_list = primitive_list

        n = len(primitive_list)
        # Initialize arrays
        self.vao_id = np.zeros(n, np.int32)
        self.program_id = {}  # Will be populated in load_programs
        self.pose = np.zeros((n, 4, 4), np.float32)
        self.textures = np.zeros((n, 8), np.int32)  # 0: flag, 1-7: texture ids
        self.pbr_mat = np.zeros((n, 9), np.float32)  # base_color(4), metallic(1), roughness(1), emissive(3)
        self.spec_mat = np.zeros((n, 11), np.float32)  # diffuse(4), specular(3), glossiness(1), emissive(3)
        self.render_flags = np.zeros(
            (n, 7), np.int8
        )  # (blend, wireframe, double sided, pbr texture, reflective floor, transparent, env shared)
        self.mode = np.zeros(n, np.int32)
        self.n_instances = np.zeros(n, np.int32)
        self.n_indices = np.zeros(n, np.int32)  # positive: indices, negative: positions

        floor_existed = False

        for i, primitive in enumerate(primitive_list):
            # Set basic properties
            self.vao_id[i] = primitive._vaid
            self.pose[i] = scene.get_pose(node_list[i])

            # Handle material properties
            material = primitive.material
            tf = material.tex_flags
            self.textures[i, 0] = tf

            # Set texture IDs if they exist
            if tf & 1:  # Normal texture
                self.textures[i, 1] = material.normalTexture._texid
            if tf & 2:  # Occlusion texture
                self.textures[i, 2] = material.occlusiontexture._texid
            if tf & 4:  # Emissive texture
                self.textures[i, 3] = material.emissiveTexture._texid
            if tf & 8:  # Base color texture
                self.textures[i, 4] = material.baseColorTexture._texid
            if tf & 16:  # Metallic roughness texture
                self.textures[i, 5] = material.metallicRoughnessTexture._texid
            if tf & 32:  # Diffuse texture
                self.textures[i, 6] = material.diffuseTexture._texid
            if tf & 64:  # Specular glossiness texture
                self.textures[i, 7] = material.specularGlossinessTexture._texid

            # Set material properties based on type
            if isinstance(material, MetallicRoughnessMaterial):
                self.pbr_mat[i, :4] = material.baseColorFactor
                self.pbr_mat[i, 4] = material.metallicFactor
                self.pbr_mat[i, 5] = material.roughnessFactor
                self.pbr_mat[i, 6:9] = material.emissiveFactor
            elif isinstance(material, SpecularGlossinessMaterial):
                self.spec_mat[i, :4] = material.diffuseFactor
                self.spec_mat[i, 4:7] = material.specularFactor
                self.spec_mat[i, 7] = material.glossinessFactor
                self.spec_mat[i, 8:11] = material.emissiveFactor

            # Set render flags
            self.render_flags[i, 0] = material.alphaMode == "BLEND"
            self.render_flags[i, 1] = material.wireframe
            self.render_flags[i, 2] = material.doubleSided
            self.render_flags[i, 3] = isinstance(material, MetallicRoughnessMaterial)
            self.render_flags[i, 4] = primitive.is_floor and not floor_existed
            self.render_flags[i, 5] = node_list[i].mesh.is_transparent
            self.render_flags[i, 6] = primitive.env_shared

            if primitive.is_floor:
                floor_existed = True

            # Set geometry properties
            self.mode[i] = primitive.mode
            self.n_instances[i] = len(primitive.poses) if primitive.poses is not None else 1
            self.n_indices[i] = primitive.indices.size if primitive.indices is not None else -len(primitive.positions)

    def forward_pass(
        self,
        renderer,
        V,
        P,
        cam_pos,
        flags,
        program_flags,
        screen_size,
        color_list=None,
        reflection_mat=None,
        floor_tex=0,
        env_idx=-1,
    ):
        """Forward rendering pass implementation"""
        if reflection_mat is None:
            reflection_mat = np.identity(4, np.float32)

        self.load_programs(renderer, flags, program_flags)
        det_reflection = np.linalg.det(reflection_mat)
        last_pid = -1

        # Sort objects by transparency
        solid_idx = [i for i in range(len(self.vao_id)) if not self.render_flags[i, 5]]
        trans_idx = [i for i in range(len(self.vao_id)) if self.render_flags[i, 5]]
        idx = solid_idx + trans_idx

        for id in idx:
            # Skip floor if needed
            if self.render_flags[id, 4] and (flags & RenderFlags_SKIP_FLOOR):
                continue

            pid = self.program_id[(flags, program_flags)][id]
            if pid != last_pid:
                glUseProgram(pid)
                if not (flags & RenderFlags_DEPTH_ONLY or flags & RenderFlags_SEG or flags & RenderFlags_FLAT):
                    lighting_texture = bind_lighting(
                        pid, flags, self.light, self.shadow_map, self.light_matrix, self.ambient_light
                    )
                    set_uniform_3fv(pid, "cam_pos", cam_pos)
                    set_uniform_matrix_4fv(pid, "reflection_mat", reflection_mat)

                set_uniform_matrix_4fv(pid, "V", V)
                set_uniform_matrix_4fv(pid, "P", P)
                last_pid = pid

            active_texture = lighting_texture if "lighting_texture" in locals() else 0

            # Handle reflective floor
            if self.render_flags[id, 4] and (flags & RenderFlags_REFLECTIVE_FLOOR):
                glActiveTexture(GL_TEXTURE0 + active_texture)
                glBindTexture(GL_TEXTURE_2D, floor_tex)
                set_uniform_1i(pid, "floor_tex", active_texture)
                set_uniform_1i(pid, "floor_flag", 1)
                set_uniform_2f(pid, "screen_size", screen_size[0], screen_size[1])
                active_texture += 1
            elif flags & RenderFlags_REFLECTIVE_FLOOR:
                set_uniform_1i(pid, "floor_tex", 0)
                set_uniform_1i(pid, "floor_flag", 0)

            # Set model matrix and bind VAO
            set_uniform_matrix_4fv(pid, "M", self.pose[id])
            glBindVertexArray(self.vao_id[id])

            # Handle material properties and rendering flags
            if not (flags & RenderFlags_DEPTH_ONLY or flags & RenderFlags_SEG or flags & RenderFlags_FLAT):
                # Handle textures
                tf = self.textures[id, 0]
                texture_list = [
                    "normal_texture",
                    "occlusion_texture",
                    "emissive_texture",
                    "base_color_texture",
                    "metallic_roughness_texture",
                    "diffuse_texture",
                    "specular_glossiness_texture",
                ]
                for i in range(7):
                    if tf & (1 << i):
                        glActiveTexture(GL_TEXTURE0 + active_texture)
                        glBindTexture(GL_TEXTURE_2D, self.textures[id, i + 1])
                        set_uniform_1i(pid, f"material.{texture_list[i]}", active_texture)
                        active_texture += 1

                # Set material properties
                if self.render_flags[id, 3]:  # PBR material
                    set_uniform_4fv(pid, "material.base_color_factor", self.pbr_mat[id, :4])
                    set_uniform_1f(pid, "material.metallic_factor", self.pbr_mat[id, 4])
                    set_uniform_1f(pid, "material.roughness_factor", self.pbr_mat[id, 5])
                    set_uniform_3fv(pid, "material.emissive_factor", self.pbr_mat[id, 6:9])
                else:  # Specular material
                    set_uniform_4fv(pid, "material.diffuse_factor", self.spec_mat[id, :4])
                    set_uniform_3fv(pid, "material.specular_factor", self.spec_mat[id, 4:7])
                    set_uniform_1f(pid, "material.roughness_factor", self.spec_mat[id, 7])
                    set_uniform_3fv(pid, "material.emissive_factor", self.spec_mat[id, 8:11])

                # Set blend mode
                if self.render_flags[id, 0]:
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                else:
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_ONE, GL_ZERO)

                # Set wireframe mode
                wf = self.render_flags[id, 1]
                if flags & RenderFlags_FLIP_WIREFRAME:
                    wf = not wf
                if (flags & RenderFlags_ALL_WIREFRAME) or wf:
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                else:
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                # Set face culling
                if self.render_flags[id, 2] or flags & RenderFlags_SKIP_CULL_FACES:
                    glDisable(GL_CULL_FACE)
                else:
                    glEnable(GL_CULL_FACE)
                    glCullFace(GL_BACK if det_reflection > 0 else GL_FRONT)
            else:
                # Default state for depth-only or segmentation rendering
                glEnable(GL_CULL_FACE)
                glEnable(GL_BLEND)
                glCullFace(GL_BACK if det_reflection > 0 else GL_FRONT)
                glBlendFunc(GL_ONE, GL_ZERO)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            glDisable(GL_PROGRAM_POINT_SIZE)

            # Handle segmentation color if needed
            if flags & RenderFlags_SEG:
                if color_list[id, 0] < -1:
                    glBindVertexArray(0)
                    continue
                set_uniform_3fv(pid, "color", color_list[id])

            # Draw the geometry
            if self.render_flags[id, 6] or env_idx == -1:
                if self.n_indices[id] > 0:
                    glDrawElementsInstanced(
                        self.mode[id], self.n_indices[id], GL_UNSIGNED_INT, None, self.n_instances[id]
                    )
                else:
                    glDrawArraysInstanced(self.mode[id], 0, -self.n_indices[id], self.n_instances[id])
            else:
                if self.n_indices[id] > 0:
                    gl.glDrawElementsInstancedBaseInstance(
                        self.mode[id], self.n_indices[id], GL_UNSIGNED_INT, None, 1, env_idx
                    )
                else:
                    gl.glDrawArraysInstancedBaseInstance(self.mode[id], 0, -self.n_indices[id], 1, env_idx)

            glBindVertexArray(0)

        # Clean up
        glUseProgram(0)
        glFlush()

    def shadow_mapping_pass(self, renderer, V, P, flags, program_flags, env_idx):
        """Render shadow map for directional lights"""
        self.load_programs(renderer, flags, program_flags)
        last_pid = -1

        for id in range(len(self.vao_id)):
            if self.render_flags[id, 5]:  # Skip transparent objects
                continue

            pid = self.program_id[(flags, program_flags)][id]
            if pid != last_pid:
                glUseProgram(pid)
                # Set view and projection matrices
                set_uniform_matrix_4fv(pid, "V", V)
                set_uniform_matrix_4fv(pid, "P", P)
                last_pid = pid

            # Set model matrix
            set_uniform_matrix_4fv(pid, "M", self.pose[id])
            glBindVertexArray(self.vao_id[id])

            # Set rendering state
            glEnable(GL_CULL_FACE)
            glEnable(GL_BLEND)
            glCullFace(GL_BACK)
            glBlendFunc(GL_ONE, GL_ZERO)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDisable(GL_PROGRAM_POINT_SIZE)

            # Draw geometry
            if self.n_indices[id] > 0:
                glDrawElementsInstanced(self.mode[id], self.n_indices[id], GL_UNSIGNED_INT, None, self.n_instances[id])
            else:
                glDrawArraysInstanced(self.mode[id], 0, -self.n_indices[id], self.n_instances[id])

            glBindVertexArray(0)

        glUseProgram(0)
        glFlush()

    def point_shadow_mapping_pass(self, renderer, light_matrix, light_pos, flags, program_flags, env_idx=-1):
        """Render shadow maps for point lights"""
        self.load_programs(renderer, flags, program_flags)
        last_pid = -1

        for id in range(len(self.vao_id)):
            if self.render_flags[id, 5]:  # Skip transparent objects
                continue

            pid = self.program_id[(flags, program_flags)][id]
            if pid != last_pid:
                glUseProgram(pid)
                # Set light matrices and position
                for i in range(6):
                    set_uniform_matrix_4fv(pid, f"light_matrix[{i}]", light_matrix[i])
                set_uniform_3fv(pid, "light_pos", light_pos)
                last_pid = pid

            # Set model matrix
            set_uniform_matrix_4fv(pid, "M", self.pose[id])
            glBindVertexArray(self.vao_id[id])

            # Set rendering state
            glEnable(GL_CULL_FACE)
            glEnable(GL_BLEND)
            glCullFace(GL_BACK)
            glBlendFunc(GL_ONE, GL_ZERO)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDisable(GL_PROGRAM_POINT_SIZE)

            # Draw geometry
            if self.n_indices[id] > 0:
                glDrawElementsInstanced(self.mode[id], self.n_indices[id], GL_UNSIGNED_INT, None, self.n_instances[id])
            else:
                glDrawArraysInstanced(self.mode[id], 0, -self.n_indices[id], self.n_instances[id])

            glBindVertexArray(0)

        glUseProgram(0)
        glFlush()

    def update_normal(self, node, vertices):
        """Update normal vectors for the given vertices

        Parameters
        ----------
        node : Node
            The node containing the mesh to update
        vertices : ndarray
            Vertex positions to calculate normals for

        Returns
        -------
        normals : ndarray or None
            Calculated normal vectors, or None if normals can't be calculated
        """
        primitive = node.mesh.primitives[0]
        if primitive.normals is None:
            return None

        if primitive.indices is not None:
            return self._update_normal_smooth(vertices, primitive.indices)
        else:
            return self._update_normal_flat(vertices.reshape((-1, 3, 3)))

    def _update_normal_smooth(self, p, idx):
        """Calculate smooth normals using vertex indices"""
        # Calculate face normals
        face_normal = np.cross(p[idx[:, 1]] - p[idx[:, 0]], p[idx[:, 2]] - p[idx[:, 0]])

        # Initialize vertex normals
        vertex_normal = np.zeros_like(p)

        # Accumulate face normals to vertices
        for f in range(face_normal.shape[0]):
            vertex_normal[idx[f, 0]] += face_normal[f]
            vertex_normal[idx[f, 1]] += face_normal[f]
            vertex_normal[idx[f, 2]] += face_normal[f]

        # Normalize vectors
        for v in range(vertex_normal.shape[0]):
            vertex_normal[v] /= np.linalg.norm(vertex_normal[v])

        return vertex_normal

    def _update_normal_flat(self, p):
        """Calculate flat normals for triangle vertices"""
        # Calculate face normals
        face_normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])

        # Initialize vertex normals array
        vertex_normal = np.zeros((p.shape[0] * 3, 3), p.dtype)

        # Assign face normal to all vertices of the face
        for f in range(face_normal.shape[0]):
            n = face_normal[f]
            n /= np.linalg.norm(n)
            vertex_normal[f * 3 + 0] = n
            vertex_normal[f * 3 + 1] = n
            vertex_normal[f * 3 + 2] = n

        return vertex_normal

    def update_buffer(self, buffer_updates):
        """Update OpenGL buffer data

        Parameters
        ----------
        buffer_updates : dict
            Dictionary mapping buffer IDs to their updated data
        """
        for buffer_id, data in buffer_updates.items():
            if buffer_id >= 0:  # Valid buffer ID check
                # Flatten and ensure data is contiguous
                flattened = data.astype(np.float32, order="C", copy=False).reshape((-1,))
                buffer_size = 4 * len(flattened)  # 4 bytes per float32

                # Bind buffer and update its data
                glBindBuffer(GL_ARRAY_BUFFER, buffer_id)
                # Allocate new buffer
                glBufferData(GL_ARRAY_BUFFER, buffer_size, None, GL_STREAM_DRAW)
                # Update buffer data
                glBufferSubData(GL_ARRAY_BUFFER, 0, buffer_size, flattened)
                # Unbind buffer
                glBindBuffer(GL_ARRAY_BUFFER, 0)

    def read_depth_buf(self, width, height, z_near, z_far):
        """Read depth buffer and convert to actual depth values

        Parameters
        ----------
        width : int
            Width of the viewport
        height : int
            Height of the viewport
        z_near : float
            Near plane distance
        z_far : float
            Far plane distance

        Returns
        -------
        depth_im : ndarray
            Depth image with actual depth values
        """
        # Initialize buffer to receive the depth data
        buf = np.zeros((height, width), np.float32)

        # Read depth buffer
        glReadPixels(
            0,
            0,
            width,
            height,
            GL_DEPTH_COMPONENT,
            GL_FLOAT,
            buf,  # (x,y) position  # size  # format  # type  # output array
        )

        # Flip vertically since OpenGL has origin at bottom-left
        depth_im = buf[::-1, :]

        # Convert from [0,1] to [-1,1]
        depth_im = depth_im * 2 - 1

        # Convert to actual depth values
        if z_far < 0:
            # Special case for infinite far plane
            depth_im = z_near / (1 - depth_im) * 2
        else:
            # Regular perspective projection
            depth_im = (z_near * z_far) / (z_far + z_near - depth_im * (z_far - z_near)) * 2

        return depth_im

    def read_color_buf(self, width, height, rgba):
        """Read color buffer

        Parameters
        ----------
        width : int
            Width of the viewport
        height : int
            Height of the viewport
        rgba : bool
            If True, read RGBA values, else read RGB

        Returns
        -------
        color_im : ndarray
            Color image array
        """
        # Initialize buffer with appropriate shape
        if rgba:
            buf = np.zeros((height, width, 4), np.uint8)
            format = GL_RGBA
        else:
            buf = np.zeros((height, width, 3), np.uint8)
            format = GL_RGB

        # Read color buffer
        glReadPixels(
            0, 0, width, height, format, GL_UNSIGNED_BYTE, buf
        )  # (x,y) position  # size  # format  # type  # output array

        # Flip vertically to match typical image convention
        return buf[::-1, :, :]

    def load_programs(self, renderer, flags, program_flags):
        if (flags, program_flags) not in self.program_id:
            program_id = np.zeros_like(self.vao_id)
            for i, primitive in enumerate(self.primitive_list):
                program = renderer._get_primitive_program(primitive, flags, program_flags)
                program_id[i] = program._program_id
            self.program_id[(flags, program_flags)] = program_id

    def gen_func_ptr(self):
        # dummy function
        pass
