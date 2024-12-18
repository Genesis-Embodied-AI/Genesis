"""
widget.py
-------------

A widget which can visualize trimesh.Scene objects in a glooey window.

Check out an example in `examples/widget.py`
"""

import glooey
import numpy as np
import pyglet
from pyglet import gl

from .. import rendering
from .trackball import Trackball
from .windowed import geometry_hash
from .windowed import SceneViewer


class SceneGroup(pyglet.graphics.Group):

    def __init__(
        self,
        rect,
        scene,
        background=None,
        pixel_per_point=(1, 1),
        parent=None,
    ):
        super().__init__(parent)
        self.rect = rect
        self.scene = scene

        if background is None:
            background = [0.99, 0.99, 0.99, 1.0]
        self._background = background

        self._pixel_per_point = pixel_per_point

    def _set_view(self):
        left = int(self._pixel_per_point[0] * self.rect.left)
        bottom = int(self._pixel_per_point[1] * self.rect.bottom)
        width = int(self._pixel_per_point[0] * self.rect.width)
        height = int(self._pixel_per_point[1] * self.rect.height)

        gl.glPushAttrib(gl.GL_ENABLE_BIT)
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(left, bottom, width, height)

        self._mode = (gl.GLint)()
        gl.glGetIntegerv(gl.GL_MATRIX_MODE, self._mode)
        self._viewport = (gl.GLint * 4)()
        gl.glGetIntegerv(gl.GL_VIEWPORT, self._viewport)

        gl.glViewport(left, bottom, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        near = 0.01
        far = 1000.0
        gl.gluPerspective(self.scene.camera.fov[1], width / height, near, far)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def _unset_view(self):
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(self._mode.value)
        gl.glViewport(
            self._viewport[0],
            self._viewport[1],
            self._viewport[2],
            self._viewport[3],
        )

        gl.glPopAttrib()

    def set_state(self):
        self._set_view()

        SceneViewer._gl_set_background(self._background)
        SceneViewer._gl_enable_depth(self.scene.camera)
        SceneViewer._gl_enable_color_material()
        SceneViewer._gl_enable_blending()
        SceneViewer._gl_enable_smooth_lines()
        SceneViewer._gl_enable_lighting(self.scene)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glMultMatrixf(rendering.matrix_to_gl(np.linalg.inv(self.scene.camera_transform)))

    def unset_state(self):
        gl.glPopMatrix()

        SceneViewer._gl_unset_background()

        self._unset_view()


class MeshGroup(pyglet.graphics.Group):

    def __init__(self, transform=None, texture=None, parent=None):
        super().__init__(parent)
        if transform is None:
            transform = np.eye(4)
        self.transform = transform
        self.texture = texture

    def set_state(self):
        gl.glPushMatrix()
        gl.glMultMatrixf(rendering.matrix_to_gl(self.transform))

        if self.texture:
            gl.glEnable(self.texture.target)
            gl.glBindTexture(self.texture.target, self.texture.id)

    def unset_state(self):
        if self.texture:
            gl.glDisable(self.texture.target)

        gl.glPopMatrix()


class SceneWidget(glooey.Widget):

    def __init__(self, scene, **kwargs):
        super().__init__()
        self.scene = scene
        self._scene_group = None

        # key is node_name
        self.mesh_group = {}

        # key is geometry_name
        self.vertex_list = {}
        self.vertex_list_hash = {}
        self.textures = {}

        self._initial_camera_transform = self.scene.camera_transform.copy()
        self.reset_view()

        self._background = kwargs.pop("background", None)
        self._smooth = kwargs.pop("smooth", True)
        if kwargs:
            raise TypeError("unexpected kwargs: {}".format(kwargs))

    @property
    def scene_group(self):
        if self._scene_group is None:
            pixel_per_point = np.array(self.window.get_viewport_size()) / np.array(self.window.get_size())
            self._scene_group = SceneGroup(
                rect=self.rect,
                scene=self.scene,
                background=self._background,
                pixel_per_point=pixel_per_point,
                parent=self.group,
            )
        return self._scene_group

    def clear(self):
        self._scene_group = None
        self.mesh_group = {}
        while self.vertex_list:
            _, vertex = self.vertex_list.popitem()
            vertex.delete()
        self.vertex_list_hash = {}
        self.textures = {}

    def reset_view(self):
        self.view = {
            "ball": Trackball(
                pose=self._initial_camera_transform,
                size=self.scene.camera.resolution,
                scale=self.scene.scale,
                target=self.scene.centroid,
            )
        }
        self.scene.camera_transform = self.view["ball"].pose

    def do_claim(self):
        return 0, 0

    def do_regroup(self):
        if not self.vertex_list:
            return

        node_names = self.scene.graph.nodes_geometry
        for node_name in node_names:
            transform, geometry_name = self.scene.graph[node_name]
            if geometry_name not in self.vertex_list:
                continue
            vertex_list = self.vertex_list[geometry_name]

            if node_name in self.mesh_group:
                mesh_group = self.mesh_group[node_name]
            else:
                mesh_group = MeshGroup(
                    transform=transform, texture=self.textures.get(geometry_name), parent=self.scene_group
                )
                self.mesh_group[node_name] = mesh_group
            self.batch.migrate(vertex_list, gl.GL_TRIANGLES, mesh_group, self.batch)

    def do_draw(self):
        resolution = (self.rect.width, self.rect.height)
        if not (resolution == self.scene.camera.resolution).all():
            self.scene.camera.resolution = resolution

        node_names = self.scene.graph.nodes_geometry
        for node_name in node_names:
            transform, geometry_name = self.scene.graph[node_name]
            geometry = self.scene.geometry[geometry_name]
            self._update_node(node_name, geometry_name, geometry, transform)

    def do_undraw(self):
        if not self.vertex_list:
            return
        for vertex_list in self.vertex_list.values():
            vertex_list.delete()
        self._scene_group = None
        self.mesh_group = {}
        self.vertex_list = {}
        self.vertex_list_hash = {}
        self.textures = {}

    def on_mouse_press(self, x, y, buttons, modifiers):
        SceneViewer.on_mouse_press(self, x, y, buttons, modifiers)
        self._draw()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # detect a drag across widgets
        x_prev = x - dx
        y_prev = y - dy
        left, bottom = self.rect.left, self.rect.bottom
        width, height = self.rect.width, self.rect.height
        if not (left < x_prev <= left + width) or not (bottom < y_prev <= bottom + height):
            self.view["ball"].down(np.array([x, y]))

        SceneViewer.on_mouse_drag(self, x, y, dx, dy, buttons, modifiers)
        self._draw()

    def on_mouse_scroll(self, x, y, dx, dy):
        SceneViewer.on_mouse_scroll(self, x, y, dx, dy)
        self._draw()

    def _update_node(self, node_name, geometry_name, geometry, transform):
        geometry_hash_new = geometry_hash(geometry)
        if self.vertex_list_hash.get(geometry_name) != geometry_hash_new:
            # if geometry has texture defined convert it to opengl form
            if hasattr(geometry, "visual") and hasattr(geometry.visual, "material"):
                tex = rendering.material_to_texture(geometry.visual.material)
                if tex is not None:
                    self.textures[geometry_name] = tex

        if node_name in self.mesh_group:
            mesh_group = self.mesh_group[node_name]
            mesh_group.transform = transform
            mesh_group.texture = self.textures.get(geometry_name)
        else:
            mesh_group = MeshGroup(
                transform=transform, texture=self.textures.get(geometry_name), parent=self.scene_group
            )
            self.mesh_group[node_name] = mesh_group

        if self.vertex_list_hash.get(geometry_name) != geometry_hash_new:
            if geometry_name in self.vertex_list:
                self.vertex_list[geometry_name].delete()

            # convert geometry to constructor args
            args = rendering.convert_to_vertexlist(geometry, group=mesh_group, smooth=self._smooth)
            # create the indexed vertex list
            self.vertex_list[geometry_name] = self.batch.add_indexed(*args)
            # save the MD5 of the geometry
            self.vertex_list_hash[geometry_name] = geometry_hash_new
