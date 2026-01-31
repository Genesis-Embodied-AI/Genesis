import os
from typing import TYPE_CHECKING

import genesis as gs
from genesis.vis.keybindings import Key, Keybind

from ..viewer_plugin import ViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


class DefaultControlsPlugin(ViewerPlugin):
    """
    Default keyboard controls for the Genesis viewer.

    This plugin handles the standard viewer keyboard shortcuts for recording, changing render modes, etc.
    """

    def __init__(self):
        super().__init__()

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)

        self.viewer.register_keybinds(
            Keybind("record_video", Key.R, callback=self._toggle_record_video),
            Keybind("save_image", Key.S, callback=self._save_image),
            Keybind("reset_camera", Key.Z, callback=self._reset_camera),
            Keybind("camera_rotation", Key.A, callback=self._toggle_cam_rotation),
            Keybind("shadow", Key.H, callback=self._toggle_shadow),
            Keybind("face_normals", Key.F, callback=self._toggle_face_normals),
            Keybind("vertex_normals", Key.V, callback=self._toggle_vertex_normals),
            Keybind("world_frame", Key.W, callback=self._toggle_world_frame),
            Keybind("link_frame", Key.L, callback=self._toggle_link_frame),
            Keybind("wireframe", Key.D, callback=self._toggle_wireframe),
            Keybind("camera_frustum", Key.C, callback=self._toggle_camera_frustum),
            Keybind("reload_shader", Key.P, callback=self._reload_shader),
            Keybind("fullscreen_mode", Key.F11, callback=self._toggle_fullscreen),
        )

    def _toggle_cam_rotation(self):
        self.viewer.viewer_flags["rotate"] = not self.viewer.viewer_flags["rotate"]
        if self.viewer.viewer_flags["rotate"]:
            self.viewer.set_message_text("Rotation On")
        else:
            self.viewer.set_message_text("Rotation Off")

    def _toggle_fullscreen(self):
        self.viewer.viewer_flags["fullscreen"] = not self.viewer.viewer_flags["fullscreen"]
        self.viewer.set_fullscreen(self.viewer.viewer_flags["fullscreen"])
        self.viewer.activate()
        if self.viewer.viewer_flags["fullscreen"]:
            self.viewer.set_message_text("Fullscreen On")
        else:
            self.viewer.set_message_text("Fullscreen Off")

    def _toggle_shadow(self):
        self.viewer.render_flags["shadows"] = not self.viewer.render_flags["shadows"]
        if self.viewer.render_flags["shadows"]:
            self.viewer.set_message_text("Shadows On")
        else:
            self.viewer.set_message_text("Shadows Off")

    def _toggle_world_frame(self):
        if not self.viewer.gs_context.world_frame_shown:
            self.viewer.gs_context.on_world_frame()
            self.viewer.set_message_text("World Frame On")
        else:
            self.viewer.gs_context.off_world_frame()
            self.viewer.set_message_text("World Frame Off")

    def _toggle_link_frame(self):
        if not self.viewer.gs_context.link_frame_shown:
            self.viewer.gs_context.on_link_frame()
            self.viewer.set_message_text("Link Frame On")
        else:
            self.viewer.gs_context.off_link_frame()
            self.viewer.set_message_text("Link Frame Off")

    def _toggle_camera_frustum(self):
        if not self.viewer.gs_context.camera_frustum_shown:
            self.viewer.gs_context.on_camera_frustum()
            self.viewer.set_message_text("Camera Frustum On")
        else:
            self.viewer.gs_context.off_camera_frustum()
            self.viewer.set_message_text("Camera Frustum Off")

    def _toggle_face_normals(self):
        self.viewer.render_flags["face_normals"] = not self.viewer.render_flags["face_normals"]
        if self.viewer.render_flags["face_normals"]:
            self.viewer.set_message_text("Face Normals On")
        else:
            self.viewer.set_message_text("Face Normals Off")

    def _toggle_vertex_normals(self):
        self.viewer.render_flags["vertex_normals"] = not self.viewer.render_flags["vertex_normals"]
        if self.viewer.render_flags["vertex_normals"]:
            self.viewer.set_message_text("Vert Normals On")
        else:
            self.viewer.set_message_text("Vert Normals Off")

    def _toggle_record_video(self):
        if self.viewer.viewer_flags["record"]:
            self.viewer.save_video()
            self.viewer.set_caption(self.viewer.viewer_flags["window_title"])
        else:
            # Importing moviepy is very slow and not used very often. Let's delay import.
            from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

            self.viewer._video_recorder = FFMPEG_VideoWriter(
                filename=os.path.join(gs.utils.misc.get_cache_dir(), "tmp_video.mp4"),
                fps=self.viewer.viewer_flags["refresh_rate"],
                size=self.viewer.viewport_size,
            )
            self.viewer.set_caption("{} (RECORDING)".format(self.viewer.viewer_flags["window_title"]))
        self.viewer.viewer_flags["record"] = not self.viewer.viewer_flags["record"]

    def _save_image(self):
        self.viewer._save_image()

    def _toggle_wireframe(self):
        if self.viewer.render_flags["flip_wireframe"]:
            self.viewer.render_flags["flip_wireframe"] = False
            self.viewer.render_flags["all_wireframe"] = True
            self.viewer.render_flags["all_solid"] = False
            self.viewer.set_message_text("All Wireframe")
        elif self.viewer.render_flags["all_wireframe"]:
            self.viewer.render_flags["flip_wireframe"] = False
            self.viewer.render_flags["all_wireframe"] = False
            self.viewer.render_flags["all_solid"] = True
            self.viewer.set_message_text("All Solid")
        elif self.viewer.render_flags["all_solid"]:
            self.viewer.render_flags["flip_wireframe"] = False
            self.viewer.render_flags["all_wireframe"] = False
            self.viewer.render_flags["all_solid"] = False
            self.viewer.set_message_text("Default Wireframe")
        else:
            self.viewer.render_flags["flip_wireframe"] = True
            self.viewer.render_flags["all_wireframe"] = False
            self.viewer.render_flags["all_solid"] = False
            self.viewer.set_message_text("Flip Wireframe")

    def _reset_camera(self):
        self.viewer._reset_view()

    def _reload_shader(self):
        self.viewer._renderer.reload_program()
