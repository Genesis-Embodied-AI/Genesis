import numpy as np
from numba import *
from numba.extending import intrinsic
import OpenGL.GL as GL
import OpenGL.constant as GL_constant
from OpenGL.GL import GLint, GLuint, GLvoidp, GLvoid, GLfloat, GLsizei, GLboolean, GLenum, GLsizeiptr, GLintptr
from .material import MetallicRoughnessMaterial, SpecularGlossinessMaterial
from .light import DirectionalLight, PointLight
from .constants import RenderFlags, MAX_N_LIGHTS
from time import time
from .numba_gl_wrapper import GLWrapper

import os
import genesis as gs

os.environ["NUMBA_CACHE_DIR"] = os.path.join(gs.utils.misc.get_cache_dir(), "numba")


def load_const(const_name):
    c = GL.__getattribute__(const_name)
    if isinstance(c, GL_constant.FloatConstant):
        return float(c)
    if isinstance(c, GL_constant.IntConstant) or isinstance(c, GL_constant.LongConstant):
        return int(c)
    if isinstance(c, GL_constant.StringConstant):
        return str(c, "utf-8")
    raise TypeError("Unknown OpenGL constant type")


GL_TRUE = load_const("GL_TRUE")
GL_TEXTURE0 = load_const("GL_TEXTURE0")
GL_TEXTURE_2D = load_const("GL_TEXTURE_2D")
GL_TEXTURE_CUBE_MAP = load_const("GL_TEXTURE_CUBE_MAP")
GL_BLEND = load_const("GL_BLEND")
GL_SRC_ALPHA = load_const("GL_SRC_ALPHA")
GL_ONE_MINUS_SRC_ALPHA = load_const("GL_ONE_MINUS_SRC_ALPHA")
GL_ONE = load_const("GL_ONE")
GL_ZERO = load_const("GL_ZERO")
GL_FRONT_AND_BACK = load_const("GL_FRONT_AND_BACK")
GL_LINE = load_const("GL_LINE")
GL_FILL = load_const("GL_FILL")
GL_CULL_FACE = load_const("GL_CULL_FACE")
GL_BACK = load_const("GL_BACK")
GL_FRONT = load_const("GL_FRONT")
GL_PROGRAM_POINT_SIZE = load_const("GL_PROGRAM_POINT_SIZE")
GL_UNSIGNED_INT = load_const("GL_UNSIGNED_INT")
GL_RGBA = load_const("GL_RGBA")
GL_RGB = load_const("GL_RGB")
GL_DEPTH_COMPONENT = load_const("GL_DEPTH_COMPONENT")
GL_UNSIGNED_BYTE = load_const("GL_UNSIGNED_BYTE")
GL_FLOAT = load_const("GL_FLOAT")
GL_ARRAY_BUFFER = load_const("GL_ARRAY_BUFFER")
GL_STREAM_DRAW = load_const("GL_STREAM_DRAW")

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


@njit
def get_uniform_location(pid, name, gl):
    n = len(name)
    arr = np.zeros(n + 1, np.uint8)
    for i in range(n):
        arr[i] = ord(name[i])
    return gl.glGetUniformLocation(pid, arr.ctypes.data)


@njit
def set_uniform_matrix_4fv(pid, name, value, gl):
    loc = get_uniform_location(pid, name, gl)
    if loc >= 0:
        gl.glUniformMatrix4fv(loc, 1, GL_TRUE, address_to_ptr(value.ctypes.data))
    else:
        print("uniform not found:", name)


@njit
def set_uniform_1i(pid, name, value, gl):
    loc = get_uniform_location(pid, name, gl)
    if loc >= 0:
        gl.glUniform1i(loc, value)
    else:
        print("uniform not found:", name)


@njit
def set_uniform_1f(pid, name, value, gl):
    loc = get_uniform_location(pid, name, gl)
    if loc >= 0:
        gl.glUniform1f(loc, value)
    else:
        print("uniform not found:", name)


@njit
def set_uniform_2f(pid, name, value1, value2, gl):
    loc = get_uniform_location(pid, name, gl)
    if loc >= 0:
        gl.glUniform2f(loc, value1, value2)
    else:
        print("uniform not found:", name)


@njit
def set_uniform_3fv(pid, name, value, gl):
    loc = get_uniform_location(pid, name, gl)
    if loc >= 0:
        gl.glUniform3fv(loc, 1, address_to_ptr(value.ctypes.data))
    else:
        print("uniform not found:", name)


@njit
def set_uniform_4fv(pid, name, value, gl):
    loc = get_uniform_location(pid, name, gl)
    if loc >= 0:
        gl.glUniform4fv(loc, 1, address_to_ptr(value.ctypes.data))
    else:
        print("uniform not found:", name)


@njit
def bind_lighting(pid, flags, light, shadow_map, light_matrix, ambient_light, gl):
    n = len(light)
    set_uniform_3fv(pid, "ambient_light", ambient_light, gl)
    n_dir, n_pt, n_spot = 0, 0, 0
    active_texture = 0
    cube_tex = -1

    for i in range(n):
        if abs(light[i, 7] - 0.0) < 0.1:
            b = "directional_lights[" + str(n_dir) + "]."
            set_uniform_3fv(pid, b + "color", light[i, :3], gl)
            set_uniform_3fv(pid, b + "direction", light[i, 3:6], gl)
            set_uniform_1f(pid, b + "intensity", light[i, 6], gl)
            if flags & RenderFlags_SHADOWS_DIRECTIONAL:
                gl.glActiveTexture(GL_TEXTURE0 + active_texture)
                gl.glBindTexture(GL_TEXTURE_2D, shadow_map[i])
                set_uniform_1i(pid, b + "shadow_map", active_texture, gl)
                set_uniform_matrix_4fv(pid, b + "light_matrix", light_matrix[i, 0], gl)
                active_texture += 1
            n_dir += 1
        elif abs(light[i, 7] - 1.0) < 0.1:
            b = "point_lights[" + str(n_pt) + "]."
            set_uniform_3fv(pid, b + "color", light[i, :3], gl)
            set_uniform_3fv(pid, b + "position", light[i, 3:6], gl)
            set_uniform_1f(pid, b + "intensity", light[i, 6], gl)
            if flags & RenderFlags_SHADOWS_POINT:
                if active_texture == 0:
                    active_texture += 1
                gl.glActiveTexture(GL_TEXTURE0 + active_texture)
                gl.glBindTexture(GL_TEXTURE_CUBE_MAP, shadow_map[i])
                set_uniform_1i(pid, b + "shadow_map", active_texture, gl)
                cube_tex = active_texture
                active_texture += 1
            n_pt += 1

    set_uniform_1i(pid, "n_directional_lights", n_dir, gl)
    set_uniform_1i(pid, "n_spot_lights", n_spot, gl)
    set_uniform_1i(pid, "n_point_lights", n_pt, gl)

    if flags & RenderFlags_SHADOWS_POINT:
        if cube_tex < 0:
            if active_texture == 0:
                active_texture += 1
            cube_tex = active_texture
            active_texture += 1

        for i in range(n_pt, MAX_N_LIGHTS):
            b = "point_lights[" + str(i) + "]."
            set_uniform_1i(pid, b + "shadow_map", cube_tex, gl)

    return active_texture


@intrinsic
def address_to_ptr(typingctx, src):
    """returns a void pointer from a given memory address"""
    from numba.core import types, cgutils

    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return sig, codegen


class JITRenderer:
    def __init__(self, scene, node_list, primitive_list):
        self._forward_pass = None
        self._shadow_mapping_pass = None
        self._point_shadow_mapping_pass = None
        self._read_depth_buf = None
        self._read_color_buf = None
        self._update_normal_flat = None
        self._update_normal_smooth = None
        self._update_buffer = None
        self.set_primitive(scene, node_list, primitive_list)
        self.set_light(scene, scene.light_nodes, scene.ambient_light)
        self.reflection_mat = np.identity(4, np.float32)

    def update(self, scene):
        if scene.meshes_updated:
            node_list, primitive_list = [], []
            for node in scene.sorted_mesh_nodes():
                mesh = node.mesh
                if not mesh.is_visible:
                    continue
                for primitive in mesh.primitives:
                    node_list.append(node)
                    primitive_list.append(primitive)
            self.set_primitive(scene, node_list, primitive_list)
            scene.reset_meshes_updated()
        else:
            # TODO: more efficient pose update
            for i, node in enumerate(self.node_list):
                self.pose[i] = scene.get_pose(node)

        # TODO: update lights
        self.set_light(scene, scene.light_nodes, scene.ambient_light)

    def set_light(self, scene, light_nodes, ambient_light):
        self.light_list = light_nodes

        n = len(light_nodes)
        if n > MAX_N_LIGHTS:
            raise ValueError("Number of lights exceeds limit.")

        # directional: color <- 3, direction <- 3, intensity, 0
        # point:       color <- 3, position <- 3, intensity, 1
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
        self.node_list = node_list
        self.primitive_list = primitive_list

        n = len(primitive_list)
        self.vao_id = np.zeros(n, np.int32)
        self.program_id = {}
        self.pose = np.zeros((n, 4, 4), np.float32)
        self.textures = np.zeros((n, 8), np.int32)  # 0: flag, 1-7: texture id
        self.pbr_mat = np.zeros((n, 9), np.float32)  # base_color <- 4, metallic <- 1, roughness <- 1, emissive <- 3
        self.spec_mat = np.zeros((n, 11), np.float32)  # diffuse <- 4, specular <- 3, glossiness <- 1, emissive <- 3
        self.render_flags = np.zeros(
            (n, 7), np.int8
        )  # (blend, wireframe, double sided, pbr texture, reflective floor, transparent, env shared)
        self.mode = np.zeros(n, np.int32)
        self.n_instances = np.zeros(n, np.int32)
        self.n_indices = np.zeros(n, np.int32)  # positive: indices, negative: positions

        floor_existed = False

        for i, primitive in enumerate(primitive_list):
            self.vao_id[i] = primitive._vaid
            self.pose[i] = scene.get_pose(node_list[i])

            material = primitive.material
            tf = material.tex_flags
            self.textures[i, 0] = tf
            if tf & 1:
                self.textures[i, 1] = material.normalTexture._texid
            if tf & 2:
                self.textures[i, 2] = material.occlusiontexture._texid
            if tf & 4:
                self.textures[i, 3] = material.emissiveTexture._texid
            if tf & 8:
                self.textures[i, 4] = material.baseColorTexture._texid
            if tf & 16:
                self.textures[i, 5] = material.metallicRoughnessTexture._texid
            if tf & 32:
                self.textures[i, 6] = material.diffuseTexture._texid
            if tf & 64:
                self.textures[i, 7] = material.specularGlossinessTexture._texid

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

            self.render_flags[i, 0] = material.alphaMode == "BLEND"
            self.render_flags[i, 1] = material.wireframe
            self.render_flags[i, 2] = material.doubleSided
            self.render_flags[i, 3] = isinstance(material, MetallicRoughnessMaterial)
            self.render_flags[i, 4] = primitive.is_floor and not floor_existed
            self.render_flags[i, 5] = node_list[i].mesh.is_transparent
            self.render_flags[i, 6] = primitive.env_shared

            if primitive.is_floor:
                floor_existed = True

            self.mode[i] = primitive.mode
            self.n_instances[i] = len(primitive.poses) if primitive.poses is not None else 1
            self.n_indices[i] = primitive.indices.size if primitive.indices is not None else -len(primitive.positions)

    def load_programs(self, renderer, flags, program_flags):
        if (flags, program_flags) not in self.program_id:
            program_id = np.zeros_like(self.vao_id)
            for i, primitive in enumerate(self.primitive_list):
                program = renderer._get_primitive_program(primitive, flags, program_flags)
                program_id[i] = program._program_id
            self.program_id[(flags, program_flags)] = program_id

    def gen_func_ptr(self):
        self.gl = GLWrapper()

        IS_OPENGL_42_AVAILABLE = hasattr(self.gl.wrapper_instance, "glDrawElementsInstancedBaseInstance")
        OPENGL_42_ERROR_MSG = "Seperated env rendering not supported because OpenGL 4.2 not available on this machine."

        @njit(
            none(
                int32[:],
                int32[:],
                float32[:, :, :],
                int32[:, :],
                float32[:, :],
                float32[:, :],
                int8[:, :],
                int32[:],
                int32[:],
                int32[:],
                float32[:, :],
                int32[:],
                float32[:, :, :, :],
                float32[:],
                float32[:, :],
                float32[:, :],
                float32[:],
                int32,
                float32[:, :],
                float32[:, :],
                int32,
                float32[:],
                int32,
                self.gl.wrapper_type,
            ),
            cache=True,
        )
        def forward_pass(
            vao_id,
            program_id,
            pose,
            textures,
            pbr_mat,
            spec_mat,
            render_flags,
            mode,
            n_instances,
            n_indices,
            light,
            shadow_map,
            light_matrix,
            ambient_light,
            mat_V,
            mat_P,
            cam_pos,
            flags,
            color_list,
            reflection_mat,
            floor_tex,
            screen_size,
            env_idx,
            gl,
        ):
            det_reflection = np.linalg.det(reflection_mat)
            last_pid = -1
            lighting_texture = 0
            solid_idx = [i for i in range(len(vao_id)) if not render_flags[i, 5]]
            trans_idx = [i for i in range(len(vao_id)) if render_flags[i, 5]]
            idx = solid_idx + trans_idx
            for id in idx:
                if render_flags[id, 4] and (flags & RenderFlags_SKIP_FLOOR):
                    continue
                pid = program_id[id]
                if pid != last_pid:
                    gl.glUseProgram(pid)
                    if not (flags & RenderFlags_DEPTH_ONLY or flags & RenderFlags_SEG or flags & RenderFlags_FLAT):
                        lighting_texture = bind_lighting(pid, flags, light, shadow_map, light_matrix, ambient_light, gl)
                        set_uniform_3fv(pid, "cam_pos", cam_pos, gl)
                        set_uniform_matrix_4fv(pid, "reflection_mat", reflection_mat, gl)

                    set_uniform_matrix_4fv(pid, "V", mat_V, gl)
                    set_uniform_matrix_4fv(pid, "P", mat_P, gl)

                    last_pid = pid

                active_texture = lighting_texture

                if render_flags[id, 4] and (flags & RenderFlags_REFLECTIVE_FLOOR):
                    gl.glActiveTexture(GL_TEXTURE0 + active_texture)
                    gl.glBindTexture(GL_TEXTURE_2D, floor_tex)
                    set_uniform_1i(pid, "floor_tex", active_texture, gl)
                    set_uniform_1i(pid, "floor_flag", 1, gl)
                    set_uniform_2f(pid, "screen_size", screen_size[0], screen_size[1], gl)
                    active_texture += 1
                elif flags & RenderFlags_REFLECTIVE_FLOOR:
                    set_uniform_1i(pid, "floor_tex", 0, gl)
                    set_uniform_1i(pid, "floor_flag", 0, gl)

                set_uniform_matrix_4fv(pid, "M", pose[id], gl)
                gl.glBindVertexArray(vao_id[id])

                if not (flags & RenderFlags_DEPTH_ONLY or flags & RenderFlags_SEG or flags & RenderFlags_FLAT):
                    tf = textures[id, 0]
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
                            gl.glActiveTexture(GL_TEXTURE0 + active_texture)
                            gl.glBindTexture(GL_TEXTURE_2D, textures[id, i + 1])
                            set_uniform_1i(pid, "material." + texture_list[i], active_texture, gl)
                            active_texture += 1

                    if render_flags[id, 3]:
                        set_uniform_4fv(pid, "material.base_color_factor", pbr_mat[id, :4], gl)
                        set_uniform_1f(pid, "material.metallic_factor", pbr_mat[id, 4], gl)
                        set_uniform_1f(pid, "material.roughness_factor", pbr_mat[id, 5], gl)
                        set_uniform_3fv(pid, "material.emissive_factor", pbr_mat[id, 6:9], gl)
                    else:
                        set_uniform_4fv(pid, "material.diffuse_factor", spec_mat[id, :4], gl)
                        set_uniform_3fv(pid, "material.specular_factor", spec_mat[id, 4:7], gl)
                        set_uniform_1f(pid, "material.roughness_factor", spec_mat[id, 7], gl)
                        set_uniform_3fv(pid, "material.glossiness_factor", spec_mat[id, 8:11], gl)

                    if render_flags[id, 0]:
                        gl.glEnable(GL_BLEND)
                        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    else:
                        gl.glEnable(GL_BLEND)
                        gl.glBlendFunc(GL_ONE, GL_ZERO)

                    wf = render_flags[id, 1]
                    if flags & RenderFlags_FLIP_WIREFRAME:
                        wf = not wf
                    if (flags & RenderFlags_ALL_WIREFRAME) or wf:
                        gl.glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    else:
                        gl.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                    if render_flags[id, 2] or flags & RenderFlags_SKIP_CULL_FACES:
                        gl.glDisable(GL_CULL_FACE)
                    else:
                        gl.glEnable(GL_CULL_FACE)
                        gl.glCullFace(GL_BACK if det_reflection > 0 else GL_FRONT)
                else:
                    gl.glEnable(GL_CULL_FACE)
                    gl.glEnable(GL_BLEND)
                    gl.glCullFace(GL_BACK if det_reflection > 0 else GL_FRONT)
                    gl.glBlendFunc(GL_ONE, GL_ZERO)
                    gl.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                gl.glDisable(GL_PROGRAM_POINT_SIZE)

                if flags & RenderFlags_SEG:
                    if color_list[id, 0] < -1:
                        gl.glBindVertexArray(0)
                        continue
                    set_uniform_3fv(pid, "color", color_list[id], gl)

                if render_flags[id, 6] or env_idx == -1:
                    if n_indices[id] > 0:
                        gl.glDrawElementsInstanced(
                            mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), n_instances[id]
                        )
                    else:
                        gl.glDrawArraysInstanced(mode[id], 0, -n_indices[id], n_instances[id])
                elif IS_OPENGL_42_AVAILABLE:
                    if n_indices[id] > 0:
                        gl.glDrawElementsInstancedBaseInstance(
                            mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), 1, env_idx
                        )
                    else:
                        gl.glDrawArraysInstancedBaseInstance(mode[id], 0, -n_indices[id], 1, env_idx)
                else:
                    raise RuntimeError(OPENGL_42_ERROR_MSG)

                gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            gl.glFlush()

        @njit(
            none(
                int32[:],
                int32[:],
                float32[:, :, :],
                int32[:],
                int32[:],
                int32[:],
                float32[:, :],
                float32[:, :],
                int8[:, :],
                int32,
                self.gl.wrapper_type,
            ),
            cache=True,
        )
        def shadow_mapping_pass(
            vao_id, program_id, pose, mode, n_instances, n_indices, mat_V, mat_P, render_flags, env_idx, gl
        ):
            last_pid = -1
            for id in range(len(vao_id)):
                if render_flags[id, 5]:
                    continue
                pid = program_id[id]
                if pid != last_pid:
                    gl.glUseProgram(pid)

                    set_uniform_matrix_4fv(pid, "V", mat_V, gl)
                    set_uniform_matrix_4fv(pid, "P", mat_P, gl)

                    last_pid = pid

                set_uniform_matrix_4fv(pid, "M", pose[id], gl)
                gl.glBindVertexArray(vao_id[id])

                gl.glEnable(GL_CULL_FACE)
                gl.glEnable(GL_BLEND)
                gl.glCullFace(GL_BACK)
                gl.glBlendFunc(GL_ONE, GL_ZERO)
                gl.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                gl.glDisable(GL_PROGRAM_POINT_SIZE)

                if render_flags[id, 6] or env_idx == -1:
                    if n_indices[id] > 0:
                        gl.glDrawElementsInstanced(
                            mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), n_instances[id]
                        )
                    else:
                        gl.glDrawArraysInstanced(mode[id], 0, -n_indices[id], n_instances[id])
                elif IS_OPENGL_42_AVAILABLE:
                    if n_indices[id] > 0:
                        gl.glDrawElementsInstancedBaseInstance(
                            mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), 1, env_idx
                        )
                    else:
                        gl.glDrawArraysInstancedBaseInstance(mode[id], 0, -n_indices[id], 1, env_idx)
                else:
                    raise RuntimeError(OPENGL_42_ERROR_MSG)

                gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            gl.glFlush()

        @njit(
            none(
                int32[:],
                int32[:],
                float32[:, :, :],
                int32[:],
                int32[:],
                int32[:],
                float32[:, :, :],
                float32[:],
                int8[:, :],
                int32,
                self.gl.wrapper_type,
            ),
            cache=True,
        )
        def point_shadow_mapping_pass(
            vao_id, program_id, pose, mode, n_instances, n_indices, light_matrix, light_pos, render_flags, env_idx, gl
        ):
            last_pid = -1
            for id in range(len(vao_id)):
                if render_flags[id, 5]:
                    continue
                pid = program_id[id]
                if pid != last_pid:
                    gl.glUseProgram(pid)

                    for i in range(6):
                        set_uniform_matrix_4fv(pid, "light_matrix[" + str(i) + "]", light_matrix[i], gl)
                    set_uniform_3fv(pid, "light_pos", light_pos, gl)

                    last_pid = pid

                set_uniform_matrix_4fv(pid, "M", pose[id], gl)
                gl.glBindVertexArray(vao_id[id])

                gl.glEnable(GL_CULL_FACE)
                gl.glEnable(GL_BLEND)
                gl.glCullFace(GL_BACK)
                gl.glBlendFunc(GL_ONE, GL_ZERO)
                gl.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                gl.glDisable(GL_PROGRAM_POINT_SIZE)

                if render_flags[id, 6] or env_idx == -1:
                    if n_indices[id] > 0:
                        gl.glDrawElementsInstanced(
                            mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), n_instances[id]
                        )
                    else:
                        gl.glDrawArraysInstanced(mode[id], 0, -n_indices[id], n_instances[id])
                elif IS_OPENGL_42_AVAILABLE:
                    if n_indices[id] > 0:
                        gl.glDrawElementsInstancedBaseInstance(
                            mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), 1, env_idx
                        )
                    else:
                        gl.glDrawArraysInstancedBaseInstance(mode[id], 0, -n_indices[id], 1, env_idx)
                else:
                    raise RuntimeError(OPENGL_42_ERROR_MSG)

                gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            gl.glFlush()

        @njit(float32[:, :](int32, int32, float32, float32, self.gl.wrapper_type), cache=True)
        def read_depth_buf(width, height, z_near, z_far, gl):
            buf = np.zeros((height, width), np.float32)
            gl.glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, address_to_ptr(buf.ctypes.data))
            depth_im = buf[::-1, :] * 2 - 1
            if z_far < 0:
                depth_im = z_near / (1 - depth_im) * 2
            else:
                depth_im = (z_near * z_far) / (z_far + z_near - depth_im * (z_far - z_near)) * 2
            return depth_im

        @njit(uint8[:, :, :](int32, int32, int32, self.gl.wrapper_type), cache=True)
        def read_color_buf(width, height, rgba, gl):
            if rgba:
                buf = np.zeros((height, width, 4), np.uint8)
                gl.glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, address_to_ptr(buf.ctypes.data))
            else:
                buf = np.zeros((height, width, 3), np.uint8)
                gl.glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, address_to_ptr(buf.ctypes.data))
            return buf[::-1, :, :]

        @njit(float32[:, :](float32[:, :, :]), cache=True)
        def update_normal_flat(p):
            face_normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            vertex_normal = np.zeros((p.shape[0] * 3, 3), p.dtype)
            for f in range(face_normal.shape[0]):
                n = face_normal[f]
                n /= np.linalg.norm(n)
                vertex_normal[f * 3 + 0] = n
                vertex_normal[f * 3 + 1] = n
                vertex_normal[f * 3 + 2] = n
            return vertex_normal

        @njit(float32[:, :](float32[:, :], int32[:, :]), cache=True)
        def update_normal_smooth(p, idx):
            face_normal = np.cross(p[idx[:, 1]] - p[idx[:, 0]], p[idx[:, 2]] - p[idx[:, 0]])
            vertex_normal = np.zeros_like(p)
            for f in range(face_normal.shape[0]):
                vertex_normal[idx[f, 0]] += face_normal[f]
                vertex_normal[idx[f, 1]] += face_normal[f]
                vertex_normal[idx[f, 2]] += face_normal[f]
            for v in range(vertex_normal.shape[0]):
                vertex_normal[v] /= np.linalg.norm(vertex_normal[v])
            return vertex_normal

        @njit(none(int64[:, :], self.gl.wrapper_type), cache=True)
        def update_buffer(updates, gl):
            for i in range(updates.shape[0]):
                buffer_id = updates[i, 0]
                buffer_size = updates[i, 1]
                buffer_addr = updates[i, 2]
                if buffer_id >= 0:
                    gl.glBindBuffer(GL_ARRAY_BUFFER, buffer_id)

                    gl.glBufferData(GL_ARRAY_BUFFER, buffer_size, address_to_ptr(0), GL_STREAM_DRAW)
                    gl.glBufferSubData(GL_ARRAY_BUFFER, 0, buffer_size, address_to_ptr(buffer_addr))

                    gl.glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._forward_pass = forward_pass
        self._shadow_mapping_pass = shadow_mapping_pass
        self._point_shadow_mapping_pass = point_shadow_mapping_pass
        self._read_depth_buf = read_depth_buf
        self._read_color_buf = read_color_buf
        self._update_normal_flat = update_normal_flat
        self._update_normal_smooth = update_normal_smooth
        self._update_buffer = update_buffer

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
        reflection_mat=np.identity(4, np.float32),
        floor_tex=0,
        env_idx=-1,
    ):
        self.load_programs(renderer, flags, program_flags)
        if self._forward_pass is None:
            self.gen_func_ptr()
        # timer = time()
        if flags & RenderFlags.SEG:
            self._forward_pass(
                self.vao_id,
                self.program_id[(flags, program_flags)],
                self.pose,
                self.textures,
                self.pbr_mat,
                self.spec_mat,
                self.render_flags,
                self.mode,
                self.n_instances,
                self.n_indices,
                self.light,
                self.shadow_map,
                self.light_matrix,
                self.ambient_light,
                V.astype(np.float32),
                P.astype(np.float32),
                cam_pos.astype(np.float32),
                flags,
                color_list,
                reflection_mat,
                floor_tex,
                screen_size,
                env_idx,
                self.gl.wrapper_instance,
            )
        else:
            self._forward_pass(
                self.vao_id,
                self.program_id[(flags, program_flags)],
                self.pose,
                self.textures,
                self.pbr_mat,
                self.spec_mat,
                self.render_flags,
                self.mode,
                self.n_instances,
                self.n_indices,
                self.light,
                self.shadow_map,
                self.light_matrix,
                self.ambient_light,
                V.astype(np.float32),
                P.astype(np.float32),
                cam_pos.astype(np.float32),
                flags,
                self.pbr_mat,
                reflection_mat,
                floor_tex,
                screen_size,
                env_idx,
                self.gl.wrapper_instance,
            )
        # print(100.0/(time()-timer))

    def shadow_mapping_pass(self, renderer, V, P, flags, program_flags, env_idx=-1):
        self.load_programs(renderer, flags, program_flags)
        if self._shadow_mapping_pass is None:
            self.gen_func_ptr()
        self._shadow_mapping_pass(
            self.vao_id,
            self.program_id[(flags, program_flags)],
            self.pose,
            self.mode,
            self.n_instances,
            self.n_indices,
            V.astype(np.float32),
            P.astype(np.float32),
            self.render_flags,
            env_idx,
            self.gl.wrapper_instance,
        )

    def point_shadow_mapping_pass(self, renderer, light_matrix, light_pos, flags, program_flags, env_idx=-1):
        self.load_programs(renderer, flags, program_flags)
        if self._point_shadow_mapping_pass is None:
            self.gen_func_ptr()
        self._point_shadow_mapping_pass(
            self.vao_id,
            self.program_id[(flags, program_flags)],
            self.pose,
            self.mode,
            self.n_instances,
            self.n_indices,
            light_matrix.astype(np.float32),
            light_pos.astype(np.float32),
            self.render_flags,
            env_idx,
            self.gl.wrapper_instance,
        )

    def update_normal(self, node, vertices):
        primitive = node.mesh.primitives[0]
        if primitive.normals is None:
            return None
        if primitive.indices is not None:
            if self._update_normal_smooth is None:
                self.gen_func_ptr()
            return self._update_normal_smooth(vertices, primitive.indices)
        else:
            if self._update_normal_flat is None:
                self.gen_func_ptr()
            return self._update_normal_flat(vertices.reshape((-1, 3, 3)))

    def update_buffer(self, buffer_updates):
        updates = np.zeros((len(buffer_updates), 3), dtype=np.int64)
        flattened_list = []
        for idx, (id, data) in enumerate(buffer_updates.items()):
            flattened = data.astype(np.float32, order="C", copy=False).reshape((-1,))
            flattened_list.append(flattened)

            updates[idx, 0] = id
            updates[idx, 1] = 4 * len(flattened)
            updates[idx, 2] = flattened.ctypes.data

        if self._update_buffer is None:
            self.gen_func_ptr()
        self._update_buffer(updates, self.gl.wrapper_instance)

    def read_depth_buf(self, weight, height, z_near, z_far):
        if self._read_depth_buf is None:
            self.gen_func_ptr()
        return self._read_depth_buf(weight, height, z_near, z_far, self.gl.wrapper_instance)

    def read_color_buf(self, weight, height, rgba):
        if self._read_color_buf is None:
            self.gen_func_ptr()
        return self._read_color_buf(weight, height, rgba, self.gl.wrapper_instance)
