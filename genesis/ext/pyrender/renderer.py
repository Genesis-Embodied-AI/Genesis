"""PBR renderer for Python.

Author: Matthew Matl
"""

import sys
from time import time

import numpy as np
import PIL
from OpenGL.GL import *
import matplotlib.pyplot as plt

from .constants import (
    DEFAULT_Z_FAR,
    DEFAULT_Z_NEAR,
    GLTF,
    MAX_N_LIGHTS,
    SHADOW_TEX_SZ,
    BufFlags,
    ProgramFlags,
    RenderFlags,
    TexFlags,
    TextAlign,
)
from .font import FontCache
from .jit_render import JITRenderer
from .light import DirectionalLight, PointLight, SpotLight
from .material import MetallicRoughnessMaterial, SpecularGlossinessMaterial
from .shader_program import ShaderProgramCache
from .utils import format_color_vector
from .texture import Texture


class Renderer(object):
    """Class for handling all rendering operations on a scene.

    Note
    ----
    This renderer relies on the existence of an OpenGL context and
    does not create one on its own.

    Parameters
    ----------
    viewport_width : int
        Width of the viewport in pixels.
    viewport_height : int
        Width of the viewport height in pixels.
    point_size : float, optional
        Size of points in pixels. Defaults to 1.0.
    """

    def __init__(self, viewport_width, viewport_height, jit, point_size=1.0):
        self.dpscale = 1

        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.point_size = point_size

        # Optional framebuffer for offscreen renders
        self._main_fb = None
        self._main_cb = None
        self._main_db = None
        self._main_fb_ms = None
        self._main_cb_ms = None
        self._main_db_ms = None
        self._main_fb_dims = (None, None)
        self._shadow_fb = None
        self._floor_fb = None
        self._latest_znear = DEFAULT_Z_NEAR
        self._latest_zfar = DEFAULT_Z_FAR

        # Shader Program Cache
        self._program_cache = ShaderProgramCache()
        self._font_cache = FontCache()
        self._meshes = set()
        self._mesh_textures = set()
        self._shadow_textures = set()
        self._texture_alloc_idx = 0

        self._floor_texture_color = None
        self._floor_texture_depth = None

        self.jit = jit

    @property
    def viewport_width(self):
        """int : The width of the main viewport, in pixels."""
        return self._viewport_width

    @viewport_width.setter
    def viewport_width(self, value):
        self._viewport_width = self.dpscale * value

    @property
    def viewport_height(self):
        """int : The height of the main viewport, in pixels."""
        return self._viewport_height

    @viewport_height.setter
    def viewport_height(self, value):
        self._viewport_height = self.dpscale * value

    @property
    def point_size(self):
        """float : The size of screen-space points, in pixels."""
        return self._point_size

    @point_size.setter
    def point_size(self, value):
        self._point_size = float(value)

    def render(self, scene, flags, seg_node_map=None):
        """Render a scene with the given set of flags.

        Parameters
        ----------
        scene : :class:`Scene`
            A scene to render.
        flags : int
            A specification from :class:`.RenderFlags`.
        seg_node_map : dict
            A map from :class:`.Node` objects to (3,) colors for each.
            If specified along with flags set to :attr:`.RenderFlags.SEG`,
            the color image will be a segmentation image.

        Returns
        -------
        color_im : (h, w, 3) uint8 or (h, w, 4) uint8
            If :attr:`RenderFlags.OFFSCREEN` is set, the color buffer. This is
            normally an RGB buffer, but if :attr:`.RenderFlags.RGBA` is set,
            the buffer will be a full RGBA buffer.
        depth_im : (h, w) float32
            If :attr:`RenderFlags.OFFSCREEN` is set, the depth buffer
            in linear units.
        """
        # Update context with meshes and textures
        self._update_context(scene, flags)

        self.jit.update(scene)

        if bool(flags & RenderFlags.DEPTH_ONLY or flags & RenderFlags.SEG or flags & RenderFlags.FLAT):
            flags &= ~RenderFlags.REFLECTIVE_FLOOR

        if bool(flags & RenderFlags.ENV_SEPARATE) and bool(flags & RenderFlags.OFFSCREEN):
            n_envs = scene.n_envs
            use_env_idx = True
        else:
            n_envs = 1
            use_env_idx = False

        retval_list = None
        for i in range(n_envs):
            env_idx = i if use_env_idx else -1

            # Render necessary shadow maps
            if not bool(flags & RenderFlags.DEPTH_ONLY or flags & RenderFlags.SEG):
                for ln in scene.light_nodes:
                    take_pass = False
                    if isinstance(ln.light, DirectionalLight) and bool(flags & RenderFlags.SHADOWS_DIRECTIONAL):
                        take_pass = True
                    elif isinstance(ln.light, SpotLight) and bool(flags & RenderFlags.SHADOWS_SPOT):
                        take_pass = False
                    elif isinstance(ln.light, PointLight) and bool(flags & RenderFlags.SHADOWS_POINT):
                        take_pass = True
                    if take_pass:
                        if isinstance(ln.light, PointLight):
                            self._point_shadow_mapping_pass(scene, ln, flags, env_idx=env_idx)
                        else:
                            self._shadow_mapping_pass(scene, ln, flags, env_idx=env_idx)
                        glBindFramebuffer(GL_FRAMEBUFFER, 0)

            # Make forward pass
            # forward_pass_start = time()
            if flags & RenderFlags.REFLECTIVE_FLOOR:
                self._floor_pass(scene, flags, env_idx=env_idx)

            retval = self._forward_pass(scene, flags, seg_node_map=seg_node_map, env_idx=env_idx)
            if isinstance(retval, tuple):
                if retval_list is None:
                    retval_list = tuple([[val] for val in retval])
                else:
                    for idx, val in enumerate(retval):
                        retval_list[idx].append(val)
            elif retval is not None:
                if retval_list is None:
                    retval_list = [retval]
                else:
                    retval_list.append(retval)
            # retval = self._forward_pass_legacy(scene, flags, seg_node_map=seg_node_map)
            # print('render_forward', time()-forward_pass_start)

            # If necessary, make normals pass
            if flags & (RenderFlags.VERTEX_NORMALS | RenderFlags.FACE_NORMALS):
                self._normals_pass(scene, flags, env_idx=env_idx)

        if use_env_idx:
            if isinstance(retval_list, list):
                retval_list = np.stack(retval_list, axis=0)
            elif isinstance(retval_list, tuple):
                retval_list = tuple([np.stack(val_list, axis=0) for val_list in retval_list])
        else:
            if isinstance(retval_list, list):
                retval_list = retval_list[0]
            elif isinstance(retval_list, tuple):
                retval_list = tuple([val_list[0] for val_list in retval_list])

        # Update camera settings for retrieving depth buffers
        self._latest_znear = scene.main_camera_node.camera.znear
        self._latest_zfar = scene.main_camera_node.camera.zfar

        return retval_list

    def render_text(
        self, text, x, y, font_name="OpenSans-Regular", font_pt=40, color=None, scale=1.0, align=TextAlign.BOTTOM_LEFT
    ):
        """Render text into the current viewport.

        Note
        ----
        This cannot be done into an offscreen buffer.

        Parameters
        ----------
        text : str
            The text to render.
        x : int
            Horizontal pixel location of text.
        y : int
            Vertical pixel location of text.
        font_name : str
            Name of font, from the ``pyrender/fonts`` folder, or
            a path to a ``.ttf`` file.
        font_pt : int
            Height of the text, in font points.
        color : (4,) float
            The color of the text. Default is black.
        scale : int
            Scaling factor for text.
        align : int
            One of the :class:`TextAlign` options which specifies where the
            ``x`` and ``y`` parameters lie on the text. For example,
            :attr:`TextAlign.BOTTOM_LEFT` means that ``x`` and ``y`` indicate
            the position of the bottom-left corner of the textbox.
        """
        x *= self.dpscale
        y *= self.dpscale
        font_pt *= self.dpscale

        if color is None:
            color = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            color = format_color_vector(color, 4)

        # Set up viewport for render
        self._configure_forward_pass_viewport(0)

        # Load font
        font = self._font_cache.get_font(font_name, font_pt)
        if not font._in_context():
            font._add_to_context()

        # Load program
        program = self._get_text_program()
        program._bind()

        # Set uniforms
        p = np.eye(4)
        p[0, 0] = 2.0 / self.viewport_width
        p[0, 3] = -1.0
        p[1, 1] = 2.0 / self.viewport_height
        p[1, 3] = -1.0
        program.set_uniform("projection", p)
        program.set_uniform("text_color", color)

        # Draw text
        font.render_string(text, x, y, scale, align)

    def render_texts(self, texts, x, y, font_name="UbuntuMono-Regular", font_pt=40, color=None, scale=1.0):
        x *= self.dpscale
        y *= self.dpscale
        font_pt *= self.dpscale

        if color is None:
            color = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            color = format_color_vector(color, 4)

        # Set up viewport for render
        self._configure_forward_pass_viewport(0)

        # Load font
        font = self._font_cache.get_font(font_name, font_pt)
        if not font._in_context():
            font._add_to_context()

        # Load program
        program = self._get_text_program()
        program._bind()

        # Set uniforms
        p = np.eye(4)
        p[0, 0] = 2.0 / self.viewport_width
        p[0, 3] = -1.0
        p[1, 1] = 2.0 / self.viewport_height
        p[1, 3] = -1.0
        program.set_uniform("projection", p)
        program.set_uniform("text_color", color)

        # Draw text
        for i, text in enumerate(texts):
            font.render_string(text, x, int(y - i * font_pt * 1.1), scale, TextAlign.TOP_LEFT)

    def read_color_buf(self):
        """Read and return the current viewport's color buffer.

        Alpha cannot be computed for an on-screen buffer.

        Returns
        -------
        color_im : (h, w, 3) uint8
            The color buffer in RGB byte format.
        """
        # Extract color image from frame buffer
        width, height = self.viewport_width, self.viewport_height
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glReadBuffer(GL_FRONT)
        color_buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

        # Re-format them into numpy arrays
        color_im = np.frombuffer(color_buf, dtype=np.uint8)
        color_im = color_im.reshape((height, width, 3))
        color_im = np.flip(color_im, axis=0)

        # Resize for macos if needed
        if sys.platform == "darwin":
            color_im = self._resize_image(color_im, True)

        return color_im

    def get_tex_image(self, tex_id, bind_target=GL_TEXTURE_2D, read_target=GL_TEXTURE_2D, width=None, height=None):
        """Read and return the current viewport's color buffer.

        Returns
        -------
        depth_im : (h, w) float32
            The depth buffer in linear units.
        """
        # width, height = self.viewport_width, self.viewport_height
        # width, height = 8192, 8192
        # glBindFramebuffer(GL_READ_FRAMEBUFFER, self._shadow_fb)
        # glReadBuffer(GL_NONE)
        # depth_buf = glReadPixels(
        #     0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT
        # )

        glBindTexture(bind_target, tex_id)
        depth_buf = glGetTexImage(read_target, 0, GL_RGB, GL_FLOAT)

        depth_im = np.frombuffer(depth_buf, dtype=np.float32)
        if width is None:
            height = int(depth_im.size**0.5)
            width = height
        depth_im = depth_im.reshape((height, width, 3))
        depth_im = np.flip(depth_im, axis=0)

        return depth_im

        inf_inds = depth_im == 1.0
        depth_im = 2.0 * depth_im - 1.0
        z_near, z_far = self._latest_znear, self._latest_zfar
        noninf = np.logical_not(inf_inds)
        if z_far is None:
            depth_im[noninf] = 2 * z_near / (1.0 - depth_im[noninf])
        else:
            depth_im[noninf] = (2.0 * z_near * z_far) / (z_far + z_near - depth_im[noninf] * (z_far - z_near))
        depth_im[inf_inds] = 0.0

        # # Resize for macos if needed
        # if sys.platform == 'darwin':
        #     depth_im = self._resize_image(depth_im)

        return depth_im

    def delete(self):
        """Free all allocated OpenGL resources."""
        # Free shaders
        self._program_cache.clear()

        # Free fonts
        self._font_cache.clear()

        # Free meshes
        for mesh in self._meshes:
            for p in mesh.primitives:
                p.delete()

        # Free textures
        for mesh_texture in self._mesh_textures:
            mesh_texture.delete()

        for shadow_texture in self._shadow_textures:
            shadow_texture.delete()

        self._meshes = set()
        self._mesh_textures = set()
        self._shadow_textures = set()
        self._texture_alloc_idx = 0

        self._delete_main_framebuffer()
        self._delete_shadow_framebuffer()
        self._delete_floor_framebuffer()

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass

    def _forward_pass_legacy(self, scene, flags, seg_node_map=None, env_idx=-1):
        # Set up viewport for render
        self._configure_forward_pass_viewport(flags)

        # Clear it
        if bool(flags & RenderFlags.SEG):
            glClearColor(0.0, 0.0, 0.0, 1.0)
            if seg_node_map is None:
                seg_node_map = {}
        else:
            glClearColor(*scene.bg_color)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not bool(flags & RenderFlags.SEG):
            glEnable(GL_MULTISAMPLE)
        else:
            glDisable(GL_MULTISAMPLE)

        # Set up camera matrices
        V, P = self._get_camera_matrices(scene)

        program = None
        # Now, render each object in sorted order
        for node in scene.sorted_mesh_nodes():
            mesh = node.mesh

            # Skip the mesh if it's not visible
            if not mesh.is_visible:
                continue

            # If SEG, set color
            if bool(flags & RenderFlags.SEG):
                if node not in seg_node_map:
                    continue
                color = seg_node_map[node]
                if not isinstance(color, (list, tuple, np.ndarray)):
                    color = np.repeat(color, 3)
                else:
                    color = np.asanyarray(color)
                color = color / 255.0

            for primitive in mesh.primitives:

                # First, get and bind the appropriate program
                program = self._get_primitive_program(primitive, flags, ProgramFlags.USE_MATERIAL)
                program._bind()

                # Set the camera uniforms
                program.set_uniform("V", V)
                program.set_uniform("P", P)
                program.set_uniform("cam_pos", scene.get_pose(scene.main_camera_node)[:3, 3])
                if bool(flags & RenderFlags.SEG):
                    program.set_uniform("color", color)

                # Next, bind the lighting
                if not (flags & RenderFlags.DEPTH_ONLY or flags & RenderFlags.FLAT or flags & RenderFlags.SEG):
                    self._bind_lighting(scene, program, node, flags)

                # Finally, bind and draw the primitive
                self._bind_and_draw_primitive(
                    primitive=primitive,
                    pose=scene.get_pose(node),
                    program=program,
                    flags=flags,
                    env_idx=env_idx,
                )
                self._reset_active_textures()

        # Unbind the shader and flush the output
        if program is not None:
            program._unbind()
        glFlush()

        # If doing offscreen render, copy result from framebuffer and return
        if flags & RenderFlags.OFFSCREEN:
            return self._read_main_framebuffer(scene, flags)
        else:
            return

    ###########################################################################
    # Rendering passes
    ###########################################################################

    def _floor_pass(self, scene, flags, seg_node_map=None, env_idx=-1):
        self._configure_floor_pass_viewport(flags)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        V, P = self._get_camera_matrices(scene)

        cam_pos = scene.get_pose(scene.main_camera_node)[:3, 3]
        screen_size = np.array([self.viewport_width, self.viewport_height], np.float32)

        self.jit.forward_pass(
            self,
            V,
            P,
            cam_pos,
            flags | RenderFlags.SKIP_FLOOR,
            ProgramFlags.USE_MATERIAL,
            screen_size,
            reflection_mat=self.jit.reflection_mat,
            env_idx=env_idx,
        )

        # tmp = self.get_tex_image(self._floor_texture_color._texid, width=self.viewport_width, height=self.viewport_height)
        # plt.imshow(tmp)
        # plt.show()

    def _forward_pass(self, scene, flags, seg_node_map=None, env_idx=-1):
        # Set up viewport for render
        self._configure_forward_pass_viewport(flags)

        # Clear it
        if bool(flags & RenderFlags.SEG):
            glClearColor(0.0, 0.0, 0.0, 1.0)
            if seg_node_map is None:
                seg_node_map = {}
        else:
            glClearColor(*scene.bg_color)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not bool(flags & RenderFlags.SEG):
            glEnable(GL_MULTISAMPLE)
        else:
            glDisable(GL_MULTISAMPLE)

        # Set up camera matrices
        V, P = self._get_camera_matrices(scene)
        cam_pos = scene.get_pose(scene.main_camera_node)[:3, 3]

        floor_tex = self._floor_texture_color._texid if flags & RenderFlags.REFLECTIVE_FLOOR else 0
        screen_size = np.array([self.viewport_width, self.viewport_height], np.float32)

        if flags & RenderFlags.SEG:
            color_list = np.zeros((len(self.jit.node_list), 3), np.float32)
            for i, node in enumerate(self.jit.node_list):
                if node not in seg_node_map:
                    color_list[i, :] = -2.0
                else:
                    color_list[i] = seg_node_map[node] / 255.0
            self.jit.forward_pass(
                self,
                V,
                P,
                cam_pos,
                flags,
                ProgramFlags.USE_MATERIAL,
                screen_size,
                color_list=color_list,
                env_idx=env_idx,
            )
        else:
            self.jit.forward_pass(
                self, V, P, cam_pos, flags, ProgramFlags.USE_MATERIAL, screen_size, floor_tex=floor_tex, env_idx=env_idx
            )
            # self.jit.forward_pass(self, V, P, cam_pos, flags, ProgramFlags.USE_MATERIAL,
            #                       reflection_mat=np.diag(np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float32)))

        # If doing offscreen render, copy result from framebuffer and return
        if flags & RenderFlags.OFFSCREEN:
            return self._read_main_framebuffer(scene, flags)
        else:
            return

    def _point_shadow_mapping_pass(self, scene, light_node, flags, env_idx=-1):
        light = light_node.light
        position = scene.get_pose(light_node)[:3, 3]
        camera = light._get_shadow_camera(scene.scale)
        projection = camera.get_projection_matrix()
        view = light._get_view_matrices(position)
        light_matrix = projection @ view

        self._configure_point_shadow_mapping_viewport(light, flags)

        self.jit.point_shadow_mapping_pass(
            self, light_matrix, position, flags, ProgramFlags.POINT_SHADOW, env_idx=env_idx
        )

    def _point_shadow_mapping_pass_legacy(self, scene, light_node, flags, env_idx=-1):
        light = light_node.light
        position = scene.get_pose(light_node)[:3, 3]
        camera = light._get_shadow_camera(scene.scale)
        projection = camera.get_projection_matrix()
        view = light._get_view_matrices(position)
        light_matrix = projection @ view

        self._configure_point_shadow_mapping_viewport(light, flags)

        for node in scene.sorted_mesh_nodes():
            mesh = node.mesh

            # Skip the mesh if it's not visible
            if not mesh.is_visible:
                continue

            for primitive in mesh.primitives:

                # First, get and bind the appropriate program
                program = self._get_primitive_program(primitive, flags, ProgramFlags.POINT_SHADOW)
                program._bind()

                # Set the camera uniforms
                for i in range(6):
                    program.set_uniform("light_matrix[" + str(i) + "]", light_matrix[i])
                program.set_uniform("light_pos", position)
                # program.set_uniform(
                #     'cam_pos', scene.get_pose(scene.main_camera_node)[:3,3]
                # )

                # Finally, bind and draw the primitive
                self._bind_and_draw_primitive(
                    primitive=primitive,
                    pose=scene.get_pose(node),
                    program=program,
                    flags=RenderFlags.DEPTH_ONLY,
                    env_idx=env_idx,
                )
                self._reset_active_textures()

        # for i in range(6):
        #     dep = self.get_depth_image(light.shadow_texture._texid, GL_TEXTURE_CUBE_MAP, GL_TEXTURE_CUBE_MAP_POSITIVE_X+i)
        #     print(dep.min(), dep.max())
        #     plt.imshow(dep)
        #     plt.show()

        if program is not None:
            program._unbind()
        glFlush()

    def _shadow_mapping_pass(self, scene, light_node, flags, env_idx=-1):
        light = light_node.light

        # Set up viewport for render
        self._configure_shadow_mapping_viewport(light, flags)

        # Set up camera matrices
        V, P = self._get_light_cam_matrices(scene, light_node, flags)

        self.jit.shadow_mapping_pass(self, V, P, flags, ProgramFlags.NONE, env_idx=env_idx)

        # dep = self.get_depth_image(light.shadow_texture._texid)
        # plt.imshow(dep)
        # plt.show()
        # plt.savefig('tmp/tmp_dep.jpg')

    def _shadow_mapping_pass_legacy(self, scene, light_node, flags, env_idx=-1):
        light = light_node.light

        # Set up viewport for render
        self._configure_shadow_mapping_viewport(light, flags)

        # Set up camera matrices
        V, P = self._get_light_cam_matrices(scene, light_node, flags)

        # Now, render each object in sorted order
        for node in scene.sorted_mesh_nodes():
            mesh = node.mesh

            # Skip the mesh if it's not visible
            if not mesh.is_visible:
                continue

            for primitive in mesh.primitives:

                # First, get and bind the appropriate program
                program = self._get_primitive_program(primitive, flags, ProgramFlags.NONE)
                program._bind()

                # Set the camera uniforms
                program.set_uniform("V", V)
                program.set_uniform("P", P)
                # program.set_uniform(
                #     'cam_pos', scene.get_pose(scene.main_camera_node)[:3,3]
                # )

                # Finally, bind and draw the primitive
                self._bind_and_draw_primitive(
                    primitive=primitive,
                    pose=scene.get_pose(node),
                    program=program,
                    flags=RenderFlags.DEPTH_ONLY,
                    env_idx=env_idx,
                )
                self._reset_active_textures()

        # dep = self.get_depth_image(light.shadow_texture._texid)
        # plt.imshow(dep)
        # plt.savefig('tmp/tmp_dep.jpg')

        # Unbind the shader and flush the output
        if program is not None:
            program._unbind()
        glFlush()

    def _normals_pass(self, scene, flags, env_idx=-1):
        # Set up viewport for render
        self._configure_forward_pass_viewport(flags)
        program = None

        # Set up camera matrices
        V, P = self._get_camera_matrices(scene)

        # Now, render each object in sorted order
        for node in scene.sorted_mesh_nodes():
            mesh = node.mesh

            # Skip the mesh if it's not visible
            if not mesh.is_visible:
                continue

            for primitive in mesh.primitives:

                # Skip objects that don't have normals
                if not primitive.buf_flags & BufFlags.NORMAL:
                    continue

                # First, get and bind the appropriate program
                pf = ProgramFlags.NONE
                if flags & RenderFlags.VERTEX_NORMALS:
                    pf = pf | ProgramFlags.VERTEX_NORMALS
                if flags & RenderFlags.FACE_NORMALS:
                    pf = pf | ProgramFlags.FACE_NORMALS
                program = self._get_primitive_program(primitive, flags, pf)
                program._bind()

                # Set the camera uniforms
                program.set_uniform("V", V)
                program.set_uniform("P", P)
                program.set_uniform("normal_magnitude", 0.05 * primitive.scale)
                program.set_uniform("normal_color", np.array([0.1, 0.1, 1.0, 1.0]))

                # Finally, bind and draw the primitive
                self._bind_and_draw_primitive(
                    primitive=primitive,
                    pose=scene.get_pose(node),
                    program=program,
                    flags=RenderFlags.DEPTH_ONLY,
                    env_idx=env_idx,
                )
                self._reset_active_textures()

        # Unbind the shader and flush the output
        if program is not None:
            program._unbind()
        glFlush()

    ###########################################################################
    # Handlers for binding uniforms and drawing primitives
    ###########################################################################

    def _bind_and_draw_primitive(self, primitive, pose, program, flags, env_idx):
        # Set model pose matrix
        program.set_uniform("M", pose)

        # Bind mesh buffers
        primitive._bind()

        # Bind mesh material
        if not (flags & RenderFlags.DEPTH_ONLY or flags & RenderFlags.SEG):
            material = primitive.material

            # Bind textures
            tf = material.tex_flags
            if tf & TexFlags.NORMAL:
                self._bind_texture(material.normalTexture, "material.normal_texture", program)
            if tf & TexFlags.OCCLUSION:
                self._bind_texture(material.occlusionTexture, "material.occlusion_texture", program)
            if tf & TexFlags.EMISSIVE:
                self._bind_texture(material.emissiveTexture, "material.emissive_texture", program)
            if tf & TexFlags.BASE_COLOR:
                self._bind_texture(material.baseColorTexture, "material.base_color_texture", program)
            if tf & TexFlags.METALLIC_ROUGHNESS:
                self._bind_texture(material.metallicRoughnessTexture, "material.metallic_roughness_texture", program)
            if tf & TexFlags.DIFFUSE:
                self._bind_texture(material.diffuseTexture, "material.diffuse_texture", program)
            if tf & TexFlags.SPECULAR_GLOSSINESS:
                self._bind_texture(material.specularGlossinessTexture, "material.specular_glossiness_texture", program)

            # Bind other uniforms
            b = "material.{}"
            program.set_uniform(b.format("emissive_factor"), material.emissiveFactor)
            if isinstance(material, MetallicRoughnessMaterial):
                program.set_uniform(b.format("base_color_factor"), material.baseColorFactor)
                program.set_uniform(b.format("metallic_factor"), material.metallicFactor)
                program.set_uniform(b.format("roughness_factor"), material.roughnessFactor)
            elif isinstance(material, SpecularGlossinessMaterial):
                program.set_uniform(b.format("diffuse_factor"), material.diffuseFactor)
                program.set_uniform(b.format("specular_factor"), material.specularFactor)
                program.set_uniform(b.format("glossiness_factor"), material.glossinessFactor)

            # Set blending options
            if material.alphaMode == "BLEND":
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            else:
                glEnable(GL_BLEND)
                glBlendFunc(GL_ONE, GL_ZERO)

            # Set wireframe mode
            wf = material.wireframe
            if flags & RenderFlags.FLIP_WIREFRAME:
                wf = not wf
            if (flags & RenderFlags.ALL_WIREFRAME) or wf:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Set culling mode
            if material.doubleSided or flags & RenderFlags.SKIP_CULL_FACES:
                glDisable(GL_CULL_FACE)
            else:
                glEnable(GL_CULL_FACE)
                glCullFace(GL_BACK)
        else:
            glEnable(GL_CULL_FACE)
            glEnable(GL_BLEND)
            glCullFace(GL_BACK)
            glBlendFunc(GL_ONE, GL_ZERO)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Set point size if needed
        glDisable(GL_PROGRAM_POINT_SIZE)
        if primitive.mode == GLTF.POINTS:
            glEnable(GL_PROGRAM_POINT_SIZE)
            glPointSize(self.point_size)

        # Render mesh
        n_instances = 1
        if primitive.poses is not None:
            n_instances = len(primitive.poses)

        if primitive.env_shared or env_idx == -1:
            if primitive.indices is not None:
                glDrawElementsInstanced(
                    primitive.mode, primitive.indices.size, GL_UNSIGNED_INT, ctypes.c_void_p(0), n_instances
                )
            else:
                glDrawArraysInstanced(primitive.mode, 0, len(primitive.positions), n_instances)
        else:
            if primitive.indices is not None:
                glDrawElementsInstancedBaseInstance(
                    primitive.mode, primitive.indices.size, GL_UNSIGNED_INT, ctypes.c_void_p(0), 1, env_idx
                )
            else:
                glDrawArraysInstancedBaseInstance(primitive.mode, 0, len(primitive.positions), 1, env_idx)

        # Unbind mesh buffers
        primitive._unbind()

    def _bind_lighting(self, scene, program, node, flags):
        """Bind all lighting uniform values for a scene."""
        max_n_lights = self._compute_max_n_lights(flags)

        n_d = min(len(scene.directional_light_nodes), max_n_lights[0])
        n_s = min(len(scene.spot_light_nodes), max_n_lights[1])
        n_p = min(len(scene.point_light_nodes), max_n_lights[2])
        program.set_uniform("ambient_light", scene.ambient_light)
        program.set_uniform("n_directional_lights", n_d)
        program.set_uniform("n_spot_lights", n_s)
        program.set_uniform("n_point_lights", n_p)
        plc = 0
        slc = 0
        dlc = 0

        light_nodes = scene.light_nodes
        if (
            len(scene.directional_light_nodes) > max_n_lights[0]
            or len(scene.spot_light_nodes) > max_n_lights[1]
            or len(scene.point_light_nodes) > max_n_lights[2]
        ):
            light_nodes = self._sorted_nodes_by_distance(scene, scene.light_nodes, node)

        for n in light_nodes:
            light = n.light
            pose = scene.get_pose(n)
            position = pose[:3, 3]
            direction = -pose[:3, 2]

            if isinstance(light, PointLight):
                if plc == max_n_lights[2]:
                    continue
                b = "point_lights[{}].".format(plc)
                plc += 1
                shadow = bool(flags & RenderFlags.SHADOWS_POINT)
                program.set_uniform(b + "position", position)
            elif isinstance(light, SpotLight):
                if slc == max_n_lights[1]:
                    continue
                b = "spot_lights[{}].".format(slc)
                slc += 1
                shadow = bool(flags & RenderFlags.SHADOWS_SPOT)
                las = 1.0 / max(0.001, np.cos(light.innerConeAngle) - np.cos(light.outerConeAngle))
                lao = -np.cos(light.outerConeAngle) * las
                program.set_uniform(b + "direction", direction)
                program.set_uniform(b + "position", position)
                program.set_uniform(b + "light_angle_scale", las)
                program.set_uniform(b + "light_angle_offset", lao)
            else:
                if dlc == max_n_lights[0]:
                    continue
                b = "directional_lights[{}].".format(dlc)
                dlc += 1
                shadow = bool(flags & RenderFlags.SHADOWS_DIRECTIONAL)
                program.set_uniform(b + "direction", direction)

            program.set_uniform(b + "color", light.color)
            program.set_uniform(b + "intensity", light.intensity)
            # if light.range is not None:
            #     program.set_uniform(b + 'range', light.range)
            # else:
            #     program.set_uniform(b + 'range', 0)

            if shadow:
                self._bind_texture(light.shadow_texture, b + "shadow_map", program)
                if not isinstance(light, PointLight):
                    V, P = self._get_light_cam_matrices(scene, n, flags)
                    program.set_uniform(b + "light_matrix", P.dot(V))
                else:
                    raise NotImplementedError("Point light shadows not implemented")

    def _sorted_nodes_by_distance(self, scene, nodes, compare_node):
        nodes = list(nodes)
        if compare_node is not None:
            compare_posn = scene.get_pose(compare_node)[:3, 3]
            nodes.sort(key=lambda n: np.linalg.norm(scene.get_pose(n)[:3, 3] - compare_posn))
        return nodes

    ###########################################################################
    # Context Management
    ###########################################################################

    def _update_context(self, scene, flags):

        # Update meshes
        scene_meshes = scene.meshes

        # Add new meshes to context
        for mesh in scene_meshes - self._meshes:
            for p in mesh.primitives:
                p._add_to_context()

        # Remove old meshes from context
        for mesh in self._meshes - scene_meshes:
            for p in mesh.primitives:
                p.delete()

        self._meshes = scene_meshes.copy()

        # Update mesh textures
        mesh_textures = set()
        for m in scene_meshes:
            for p in m.primitives:
                mesh_textures |= p.material.textures

        # Add new textures to context
        for texture in mesh_textures - self._mesh_textures:
            texture._add_to_context()

        # Remove old textures from context
        for texture in self._mesh_textures - mesh_textures:
            texture.delete()

        self._mesh_textures = mesh_textures.copy()

        shadow_textures = set()
        for l in scene.lights:
            # Create if needed
            active = False
            if isinstance(l, DirectionalLight) and flags & RenderFlags.SHADOWS_DIRECTIONAL:
                active = True
            elif isinstance(l, PointLight) and flags & RenderFlags.SHADOWS_POINT:
                active = True
            elif isinstance(l, SpotLight) and flags & RenderFlags.SHADOWS_SPOT:
                active = True

            if active and l.shadow_texture is None:
                l._generate_shadow_texture()
            if l.shadow_texture is not None:
                shadow_textures.add(l.shadow_texture)

        # Add new textures to context
        for texture in shadow_textures - self._shadow_textures:
            texture._add_to_context()

        # Remove old textures from context
        for texture in self._shadow_textures - shadow_textures:
            texture.delete()

        self._shadow_textures = shadow_textures.copy()

    ###########################################################################
    # Texture Management
    ###########################################################################

    def _bind_texture(self, texture, uniform_name, program):
        """Bind a texture to an active texture unit and return
        the texture unit index that was used.
        """
        tex_id = self._get_next_active_texture()
        glActiveTexture(GL_TEXTURE0 + tex_id)
        texture._bind()
        program.set_uniform(uniform_name, tex_id)

    def _get_next_active_texture(self):
        val = self._texture_alloc_idx
        self._texture_alloc_idx += 1
        return val

    def _reset_active_textures(self):
        self._texture_alloc_idx = 0

    ###########################################################################
    # Camera Matrix Management
    ###########################################################################

    def _get_camera_matrices(self, scene):
        main_camera_node = scene.main_camera_node
        if main_camera_node is None:
            raise ValueError("Cannot render scene without a camera")
        P = main_camera_node.camera.get_projection_matrix(width=self.viewport_width, height=self.viewport_height)
        pose = scene.get_pose(main_camera_node)
        V = np.linalg.inv(pose)  # V maps from world to camera
        return V, P

    def _get_light_cam_matrices(self, scene, light_node, flags):
        light = light_node.light
        pose = scene.get_pose(light_node).copy()
        s = scene.scale
        camera = light._get_shadow_camera(s)
        P = camera.get_projection_matrix()
        if isinstance(light, DirectionalLight):
            direction = -pose[:3, 2]
            c = scene.centroid
            loc = c - direction * s
            pose[:3, 3] = loc
        V = np.linalg.inv(pose)  # V maps from world to camera
        return V, P

    ###########################################################################
    # Shader Program Management
    ###########################################################################

    def _get_text_program(self):
        program = self._program_cache.get_program(vertex_shader="text.vert", fragment_shader="text.frag")

        if not program._in_context():
            program._add_to_context()

        return program

    def _compute_max_n_lights(self, flags):
        max_n_lights = [MAX_N_LIGHTS, MAX_N_LIGHTS, MAX_N_LIGHTS]
        # n_tex_units = glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS)

        # Reserved texture units: 6
        #   Normal Map
        #   Occlusion Map
        #   Emissive Map
        #   Base Color or Diffuse Map
        #   MR or SG Map
        #   Environment cubemap

        # n_reserved_textures = 6
        # n_available_textures = n_tex_units - n_reserved_textures

        # if flags & RenderFlags.SHADOWS_DIRECTIONAL and n_available_textures < max_n_lights[0]:
        #     max_n_lights[0] = n_available_textures

        return max_n_lights

    def _get_primitive_program(self, primitive, flags, program_flags):
        vertex_shader = None
        fragment_shader = None
        geometry_shader = None
        defines = {}

        if program_flags & ProgramFlags.POINT_SHADOW:
            vertex_shader = "point_shadow.vert"
            fragment_shader = "point_shadow.frag"
            geometry_shader = "point_shadow.geom"
        elif (
            bool(program_flags & ProgramFlags.USE_MATERIAL)
            and not flags & RenderFlags.DEPTH_ONLY
            and not flags & RenderFlags.FLAT
            and not flags & RenderFlags.SEG
        ):
            vertex_shader = "mesh.vert"
            fragment_shader = "mesh.frag"
            if primitive.double_sided:
                geometry_shader = "mesh_double_sided.geom"
                defines["DOUBLE_SIDED"] = 1
        elif bool(program_flags & (ProgramFlags.VERTEX_NORMALS | ProgramFlags.FACE_NORMALS)):
            vertex_shader = "vertex_normals.vert"
            if primitive.mode == GLTF.POINTS:
                geometry_shader = "vertex_normals_pc.geom"
            else:
                geometry_shader = "vertex_normals.geom"
            fragment_shader = "vertex_normals.frag"
        elif flags & RenderFlags.FLAT:
            vertex_shader = "flat.vert"
            fragment_shader = "flat.frag"
        elif flags & RenderFlags.SEG:
            vertex_shader = "segmentation.vert"
            fragment_shader = "segmentation.frag"
        else:
            vertex_shader = "mesh_depth.vert"
            fragment_shader = "mesh_depth.frag"

        # Set up vertex buffer DEFINES
        bf = primitive.buf_flags
        buf_idx = 1
        if bf & BufFlags.NORMAL:
            defines["NORMAL_LOC"] = buf_idx
            buf_idx += 1
        if bf & BufFlags.TANGENT:
            defines["TANGENT_LOC"] = buf_idx
            buf_idx += 1
        if bf & BufFlags.TEXCOORD_0:
            defines["TEXCOORD_0_LOC"] = buf_idx
            buf_idx += 1
        if bf & BufFlags.TEXCOORD_1:
            defines["TEXCOORD_1_LOC"] = buf_idx
            buf_idx += 1
        if bf & BufFlags.COLOR_0:
            defines["COLOR_0_LOC"] = buf_idx
            buf_idx += 1
        if bf & BufFlags.JOINTS_0:
            defines["JOINTS_0_LOC"] = buf_idx
            buf_idx += 1
        if bf & BufFlags.WEIGHTS_0:
            defines["WEIGHTS_0_LOC"] = buf_idx
            buf_idx += 1
        defines["INST_M_LOC"] = buf_idx

        # Set up shadow mapping defines
        if flags & RenderFlags.SHADOWS_DIRECTIONAL:
            defines["DIRECTIONAL_LIGHT_SHADOWS"] = 1
        if flags & RenderFlags.SHADOWS_SPOT:
            defines["SPOT_LIGHT_SHADOWS"] = 1
        if flags & RenderFlags.SHADOWS_POINT:
            defines["POINT_LIGHT_SHADOWS"] = 1
        max_n_lights = self._compute_max_n_lights(flags)
        defines["MAX_DIRECTIONAL_LIGHTS"] = max_n_lights[0]
        defines["MAX_SPOT_LIGHTS"] = max_n_lights[1]
        defines["MAX_POINT_LIGHTS"] = max_n_lights[2]

        # Set up vertex normal defines
        if program_flags & ProgramFlags.VERTEX_NORMALS:
            defines["VERTEX_NORMALS"] = 1
        if program_flags & ProgramFlags.FACE_NORMALS:
            defines["FACE_NORMALS"] = 1

        # Set up material texture defines
        if bool(program_flags & ProgramFlags.USE_MATERIAL):
            tf = primitive.material.tex_flags
            if tf & TexFlags.NORMAL:
                defines["HAS_NORMAL_TEX"] = 1
            if tf & TexFlags.OCCLUSION:
                defines["HAS_OCCLUSION_TEX"] = 1
            if tf & TexFlags.EMISSIVE:
                defines["HAS_EMISSIVE_TEX"] = 1
            if tf & TexFlags.BASE_COLOR:
                defines["HAS_BASE_COLOR_TEX"] = 1
            if tf & TexFlags.METALLIC_ROUGHNESS:
                defines["HAS_METALLIC_ROUGHNESS_TEX"] = 1
            if tf & TexFlags.DIFFUSE:
                defines["HAS_DIFFUSE_TEX"] = 1
            if tf & TexFlags.SPECULAR_GLOSSINESS:
                defines["HAS_SPECULAR_GLOSSINESS_TEX"] = 1
            if isinstance(primitive.material, MetallicRoughnessMaterial):
                defines["USE_METALLIC_MATERIAL"] = 1
            elif isinstance(primitive.material, SpecularGlossinessMaterial):
                defines["USE_GLOSSY_MATERIAL"] = 1

        program = self._program_cache.get_program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
            geometry_shader=geometry_shader,
            defines=defines,
        )

        if not program._in_context():
            program._add_to_context()

        return program

    ###########################################################################
    # Viewport Management
    ###########################################################################

    def _configure_forward_pass_viewport(self, flags):

        # If using offscreen render, bind main framebuffer
        if flags & RenderFlags.OFFSCREEN:
            self._configure_main_framebuffer()
            if flags & RenderFlags.SEG:
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            else:
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb_ms)
        else:
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)

        glViewport(0, 0, self.viewport_width, self.viewport_height)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)

    def _configure_floor_pass_viewport(self, flags):
        self._configure_floor_framebuffer()
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._floor_fb)

        glViewport(0, 0, self.viewport_width, self.viewport_height)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)

    def _configure_point_shadow_mapping_viewport(self, light, flags):
        self._configure_shadow_framebuffer()
        glBindFramebuffer(GL_FRAMEBUFFER, self._shadow_fb)
        light.shadow_texture._bind_as_depth_attachment()
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        glClear(GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, SHADOW_TEX_SZ, SHADOW_TEX_SZ)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)

    def _configure_shadow_mapping_viewport(self, light, flags):
        self._configure_shadow_framebuffer()
        glBindFramebuffer(GL_FRAMEBUFFER, self._shadow_fb)
        light.shadow_texture._bind_as_depth_attachment()
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        glClear(GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, SHADOW_TEX_SZ, SHADOW_TEX_SZ)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)

    ###########################################################################
    # Framebuffer Management
    ###########################################################################

    def _configure_floor_framebuffer(self):
        if self._floor_texture_depth is None:
            self._floor_texture_depth = Texture(
                width=self.viewport_width, height=self.viewport_height, source_channels="D", data_format=GL_FLOAT
            )
            self._floor_texture_depth._add_to_context()
        if self._floor_texture_color is None:
            self._floor_texture_color = Texture(
                width=self.viewport_width, height=self.viewport_height, source_channels="RGB", data_format=GL_FLOAT
            )
            self._floor_texture_color._add_to_context()
        if self._floor_fb is None:
            self._floor_fb = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._floor_fb)
            self._floor_texture_color._bind_as_color_attachment()
            self._floor_texture_depth._bind_as_depth_attachment()

    def _delete_floor_framebuffer(self):
        if self._floor_fb is not None:
            glDeleteFramebuffers(1, [self._floor_fb])
            self._floor_fb = None

        if self._floor_texture_color is not None:
            self._floor_texture_color.delete()
            self._floor_texture_color = None

        if self._floor_texture_depth is not None:
            self._floor_texture_depth.delete()
            self._floor_texture_depth = None

    def _configure_shadow_framebuffer(self):
        if self._shadow_fb is None:
            self._shadow_fb = glGenFramebuffers(1)

    def _delete_shadow_framebuffer(self):
        if self._shadow_fb is not None:
            glDeleteFramebuffers(1, [self._shadow_fb])
            self._shadow_fb = None

    def _configure_main_framebuffer(self):
        # If mismatch with prior framebuffer, delete it
        if (
            self._main_fb is not None
            and self.viewport_width != self._main_fb_dims[0]
            or self.viewport_height != self._main_fb_dims[1]
        ):
            self._delete_main_framebuffer()

        # If framebuffer doesn't exist, create it
        if self._main_fb is None:
            # Generate standard buffer
            self._main_cb, self._main_db = glGenRenderbuffers(2)

            glBindRenderbuffer(GL_RENDERBUFFER, self._main_cb)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, self.viewport_width, self.viewport_height)

            glBindRenderbuffer(GL_RENDERBUFFER, self._main_db)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.viewport_width, self.viewport_height)

            self._main_fb = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self._main_cb)
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._main_db)

            # Generate multisample buffer
            self._main_cb_ms, self._main_db_ms = glGenRenderbuffers(2)
            glBindRenderbuffer(GL_RENDERBUFFER, self._main_cb_ms)
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGBA, self.viewport_width, self.viewport_height)
            glBindRenderbuffer(GL_RENDERBUFFER, self._main_db_ms)
            glRenderbufferStorageMultisample(
                GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT24, self.viewport_width, self.viewport_height
            )
            self._main_fb_ms = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb_ms)
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self._main_cb_ms)
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._main_db_ms)

            self._main_fb_dims = (self.viewport_width, self.viewport_height)

    def _delete_main_framebuffer(self):
        if self._main_fb is not None:
            glDeleteFramebuffers(2, [self._main_fb, self._main_fb_ms])
        if self._main_cb is not None:
            glDeleteRenderbuffers(2, [self._main_cb, self._main_cb_ms])
        if self._main_db is not None:
            glDeleteRenderbuffers(2, [self._main_db, self._main_db_ms])

        self._main_fb = None
        self._main_cb = None
        self._main_db = None
        self._main_fb_ms = None
        self._main_cb_ms = None
        self._main_db_ms = None
        self._main_fb_dims = (None, None)

    def _read_main_framebuffer(self, scene, flags):
        width, height = self._main_fb_dims[0], self._main_fb_dims[1]

        if not (flags & RenderFlags.SEG):
            # Bind framebuffer and blit buffers
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb_ms)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR)
            glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb)

        # # Read depth
        z_near = scene.main_camera_node.camera.znear
        z_far = scene.main_camera_node.camera.zfar
        if z_far is None:
            z_far = -1.0
        # depth_buf = glReadPixels(
        #     0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT
        # )
        # depth_im = np.frombuffer(depth_buf, dtype=np.float32)
        # depth_im = depth_im.reshape((height, width))
        # depth_im = np.flip(depth_im, axis=0)
        # inf_inds = (depth_im == 1.0)
        # depth_im = 2.0 * depth_im - 1.0
        # noninf = np.logical_not(inf_inds)
        # if z_far is None:
        #     depth_im[noninf] = 2 * z_near / (1.0 - depth_im[noninf])
        # else:
        #     depth_im[noninf] = ((2.0 * z_near * z_far) /
        #                         (z_far + z_near - depth_im[noninf] *
        #                         (z_far - z_near)))
        # depth_im[inf_inds] = 0.0

        if flags & RenderFlags.RET_DEPTH:
            depth_im = self.jit.read_depth_buf(width, height, z_near, z_far)

            # Resize for macos if needed
            if sys.platform == "darwin":
                depth_im = self._resize_image(depth_im)
        else:
            depth_im = None

        if flags & RenderFlags.DEPTH_ONLY:
            return depth_im

        # Read color
        color_im = self.jit.read_color_buf(width, height, flags & RenderFlags.RGBA)

        # Resize for macos if needed
        if sys.platform == "darwin":
            color_im = self._resize_image(color_im, not (flags & RenderFlags.SEG))

        return color_im, depth_im

    def _resize_image(self, value, antialias=False):
        """If needed, rescale the render for MacOS."""
        img = PIL.Image.fromarray(value)
        resample = PIL.Image.NEAREST
        if antialias:
            resample = PIL.Image.BILINEAR
        size = (self.viewport_width // self.dpscale, self.viewport_height // self.dpscale)
        img = img.resize(size, resample=resample)
        return np.array(img)

    ###########################################################################
    # Shadowmap Debugging
    ###########################################################################

    def _forward_pass_no_reset(self, scene, flags, env_idx=-1):
        # Set up camera matrices
        V, P = self._get_camera_matrices(scene)

        # Now, render each object in sorted order
        for node in scene.sorted_mesh_nodes():
            mesh = node.mesh

            # Skip the mesh if it's not visible
            if not mesh.is_visible:
                continue

            for primitive in mesh.primitives:

                # First, get and bind the appropriate program
                program = self._get_primitive_program(primitive, flags, ProgramFlags.USE_MATERIAL)
                program._bind()

                # Set the camera uniforms
                program.set_uniform("V", V)
                program.set_uniform("P", P)
                program.set_uniform("cam_pos", scene.get_pose(scene.main_camera_node)[:3, 3])

                # Next, bind the lighting
                if not flags & RenderFlags.DEPTH_ONLY and not flags & RenderFlags.FLAT:
                    self._bind_lighting(scene, program, node, flags)

                # Finally, bind and draw the primitive
                self._bind_and_draw_primitive(
                    primitive=primitive,
                    pose=scene.get_pose(node),
                    program=program,
                    flags=flags,
                    env_idx=env_idx,
                )
                self._reset_active_textures()

        # Unbind the shader and flush the output
        if program is not None:
            program._unbind()
        glFlush()

    def _render_light_shadowmaps(self, scene, light_nodes, flags, tile=False, env_idx=-1):
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glClearColor(*scene.bg_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)

        w = self.viewport_width
        h = self.viewport_height

        num_nodes = len(light_nodes)
        viewport_dims = {
            (0, 2): [0, h // 2, w // 2, h],
            (1, 2): [w // 2, h // 2, w, h],
            (0, 3): [0, h // 2, w // 2, h],
            (1, 3): [w // 2, h // 2, w, h],
            (2, 3): [0, 0, w // 2, h // 2],
            (0, 4): [0, h // 2, w // 2, h],
            (1, 4): [w // 2, h // 2, w, h],
            (2, 4): [0, 0, w // 2, h // 2],
            (3, 4): [w // 2, 0, w, h // 2],
        }

        if tile:
            for i, ln in enumerate(light_nodes):
                light = ln.light

                if light.shadow_texture is None:
                    raise ValueError("Light does not have a shadow texture")

                glViewport(*viewport_dims[(i, num_nodes + 1)])

                program = self._get_debug_quad_program()
                program._bind()
                self._bind_texture(light.shadow_texture, "depthMap", program)
                self._render_debug_quad()
                self._reset_active_textures()
                glFlush()
            i += 1
            glViewport(*viewport_dims[(i, num_nodes + 1)])
            self._forward_pass_no_reset(scene, flags, env_idx=env_idx)
        else:
            for i, ln in enumerate(light_nodes):
                light = ln.light

                if light.shadow_texture is None:
                    raise ValueError("Light does not have a shadow texture")

                glViewport(0, 0, self.viewport_width, self.viewport_height)

                program = self._get_debug_quad_program()
                program._bind()
                self._bind_texture(light.shadow_texture, "depthMap", program)
                self._render_debug_quad()
                self._reset_active_textures()
                glFlush()
                return

    def _get_debug_quad_program(self):
        program = self._program_cache.get_program(vertex_shader="debug_quad.vert", fragment_shader="debug_quad.frag")
        if not program._in_context():
            program._add_to_context()
        return program

    def _render_debug_quad(self):
        x = glGenVertexArrays(1)
        glBindVertexArray(x)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glDeleteVertexArrays(1, [x])

    def reload_program(self):
        self._program_cache.clear()
        self.jit.program_id.clear()
