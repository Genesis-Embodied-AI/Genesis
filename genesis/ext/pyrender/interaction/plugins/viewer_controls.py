import os
from typing import TYPE_CHECKING

import numpy as np
import pyglet
from typing_extensions import override

import genesis as gs
from genesis.options.viewer_interactions import ViewerDefaultControls as ViewerDefaultControlsOptions

from ...constants import TEXT_PADDING
from ..base_interaction import EVENT_HANDLE_STATE, BaseViewerInteraction, register_viewer_plugin
from ..keybindings import Keybind, get_keycode_string

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node

INSTR_KEYBIND_NAME = "toggle_instructions"

@register_viewer_plugin(ViewerDefaultControlsOptions)
class ViewerDefaultControls(BaseViewerInteraction):
    """
    Default keyboard controls for the Genesis viewer.
    
    This plugin handles the standard viewer keyboard shortcuts for recording, changing render modes, etc.
    """

    def __init__(
        self,
        viewer,
        options=None,
        camera: "Node" = None,
        scene: "Scene" = None,
        viewport_size: tuple[int, int] = None,
    ):
        super().__init__(viewer, options, camera, scene, viewport_size)

        self.viewer.register_keybinds((
            Keybind(key_code=pyglet.window.key.I, name=INSTR_KEYBIND_NAME, callback_func=self.toggle_instructions),
            Keybind(key_code=pyglet.window.key.R, name="record_video", callback_func=self.toggle_record_video),
            Keybind(key_code=pyglet.window.key.S, name="save_image", callback_func=self.save_image),
            Keybind(key_code=pyglet.window.key.Z, name="reset_camera", callback_func=self.reset_camera),
            Keybind(key_code=pyglet.window.key.A, name="camera_rotation", callback_func=self.toggle_camera_rotation),
            Keybind(key_code=pyglet.window.key.H, name="shadow", callback_func=self.toggle_shadow),
            Keybind(key_code=pyglet.window.key.F, name="face_normals", callback_func=self.toggle_face_normals),
            Keybind(key_code=pyglet.window.key.V, name="vertex_normals", callback_func=self.toggle_vertex_normals),
            Keybind(key_code=pyglet.window.key.W, name="world_frame", callback_func=self.toggle_world_frame),
            Keybind(key_code=pyglet.window.key.L, name="link_frame", callback_func=self.toggle_link_frame),
            Keybind(key_code=pyglet.window.key.D, name="wireframe", callback_func=self.toggle_wireframe),
            Keybind(key_code=pyglet.window.key.C, name="camera_frustum", callback_func=self.toggle_camera_frustum),
            Keybind(key_code=pyglet.window.key.P, name="reload_shader", callback_func=self.reload_shader),
            Keybind(key_code=pyglet.window.key.F11, name="fullscreen_mode", callback_func=self.toggle_fullscreen),
        ))
        self._display_instr = False
        self._instr_texts: tuple[list[str], list[str]] = ([], [])
        self._update_instr_texts()

    def _update_instr_texts(self):
        self.instr_key_str = get_keycode_string(self.viewer._keybindings.get_by_name(INSTR_KEYBIND_NAME).key_code)
        kb_texts = [
            f"{'[' + get_keycode_string(kb.key_code):>{7}}]: " +
            kb.name.replace('_', ' ') for kb in self.viewer._keybindings.keybinds if kb.name != INSTR_KEYBIND_NAME
        ]
        self._instr_texts = (
            [f"> [{self.instr_key_str}]: show keyboard instructions"],
            [f"< [{self.instr_key_str}]: hide keyboard instructions"] + kb_texts
        )

    def toggle_instructions(self):
        self._display_instr = not self._display_instr
        self._update_instr_texts()
    
    def toggle_camera_rotation(self):
        self.viewer.viewer_flags["rotate"] = not self.viewer.viewer_flags["rotate"]
        if self.viewer.viewer_flags["rotate"]:
            self.viewer._message_text = "Rotation On"
        else:
            self.viewer._message_text = "Rotation Off"
    
    def toggle_fullscreen(self):
        self.viewer.viewer_flags["fullscreen"] = not self.viewer.viewer_flags["fullscreen"]
        self.viewer.set_fullscreen(self.viewer.viewer_flags["fullscreen"])
        self.viewer.activate()
        if self.viewer.viewer_flags["fullscreen"]:
            self.viewer._message_text = "Fullscreen On"
        else:
            self.viewer._message_text = "Fullscreen Off"
    
    def toggle_shadow(self):
        self.viewer.render_flags["shadows"] = not self.viewer.render_flags["shadows"]
        if self.viewer.render_flags["shadows"]:
            self.viewer._message_text = "Shadows On"
        else:
            self.viewer._message_text = "Shadows Off"
    
    def toggle_world_frame(self):
        if not self.viewer.gs_context.world_frame_shown:
            self.viewer.gs_context.on_world_frame()
            self.viewer._message_text = "World Frame On"
        else:
            self.viewer.gs_context.off_world_frame()
            self.viewer._message_text = "World Frame Off"
    
    def toggle_link_frame(self):
        if not self.viewer.gs_context.link_frame_shown:
            self.viewer.gs_context.on_link_frame()
            self.viewer._message_text = "Link Frame On"
        else:
            self.viewer.gs_context.off_link_frame()
            self.viewer._message_text = "Link Frame Off"
    
    def toggle_camera_frustum(self):
        if not self.viewer.gs_context.camera_frustum_shown:
            self.viewer.gs_context.on_camera_frustum()
            self.viewer._message_text = "Camera Frustrum On"
        else:
            self.viewer.gs_context.off_camera_frustum()
            self.viewer._message_text = "Camera Frustrum Off"
    
    def toggle_face_normals(self):
        self.viewer.render_flags["face_normals"] = not self.viewer.render_flags["face_normals"]
        if self.viewer.render_flags["face_normals"]:
            self.viewer._message_text = "Face Normals On"
        else:
            self.viewer._message_text = "Face Normals Off"
    
    def toggle_vertex_normals(self):
        self.viewer.render_flags["vertex_normals"] = not self.viewer.render_flags["vertex_normals"]
        if self.viewer.render_flags["vertex_normals"]:
            self.viewer._message_text = "Vert Normals On"
        else:
            self.viewer._message_text = "Vert Normals Off"
    
    def toggle_record_video(self):
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
    
    def save_image(self):
        self.viewer._save_image()
    
    def toggle_wireframe(self):
        if self.viewer.render_flags["flip_wireframe"]:
            self.viewer.render_flags["flip_wireframe"] = False
            self.viewer.render_flags["all_wireframe"] = True
            self.viewer.render_flags["all_solid"] = False
            self.viewer._message_text = "All Wireframe"
        elif self.viewer.render_flags["all_wireframe"]:
            self.viewer.render_flags["flip_wireframe"] = False
            self.viewer.render_flags["all_wireframe"] = False
            self.viewer.render_flags["all_solid"] = True
            self.viewer._message_text = "All Solid"
        elif self.viewer.render_flags["all_solid"]:
            self.viewer.render_flags["flip_wireframe"] = False
            self.viewer.render_flags["all_wireframe"] = False
            self.viewer.render_flags["all_solid"] = False
            self.viewer._message_text = "Default Wireframe"
        else:
            self.viewer.render_flags["flip_wireframe"] = True
            self.viewer.render_flags["all_wireframe"] = False
            self.viewer.render_flags["all_solid"] = False
            self.viewer._message_text = "Flip Wireframe"
    
    def reset_camera(self):
        self.viewer._reset_view()
    
    def reload_shader(self):
        self.viewer._renderer.reload_program()
    
    @override
    def on_key_press(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self.viewer is None:
            return None

        # Reset message text and check for keybinding
        self.viewer._message_text = None
        super().on_key_press(symbol, modifiers)
        
        if self.viewer._message_text is not None:
            self.viewer._message_opac = 1.0 + self.viewer._ticks_till_fade

        return None
    
    @override
    def on_draw(self):
        """Render keyboard instructions."""
        if self.viewer is None:
            return
        
        if self._display_instr:
            self.viewer._renderer.render_texts(
                self._instr_texts[1],
                TEXT_PADDING,
                self.viewer.viewport_size[1] - TEXT_PADDING,
                font_pt=26,
                color=np.array([1.0, 1.0, 1.0, 0.85]),
            )
        else:
            self.viewer._renderer.render_texts(
                self._instr_texts[0],
                TEXT_PADDING,
                self.viewer.viewport_size[1] - TEXT_PADDING,
                font_pt=26,
                color=np.array([1.0, 1.0, 1.0, 0.85]),
            )
