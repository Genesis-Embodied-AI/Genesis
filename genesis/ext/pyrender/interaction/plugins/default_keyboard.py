import os
from typing import TYPE_CHECKING

import numpy as np
import pyglet
from typing_extensions import override

import genesis as gs

from ...constants import TEXT_PADDING
from ..base_interaction import EVENT_HANDLE_STATE, BaseViewerInteraction

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


class ViewerControls(BaseViewerInteraction):
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
        
        # Instruction display state
        self._display_instr = False
        self._instr_texts = [
            ["> [i]: show keyboard instructions"],
            [
                "< [i]: hide keyboard instructions",
                "     [r]: record video",
                "     [s]: save image",
                "     [z]: reset camera",
                "     [a]: camera rotation",
                "     [h]: shadow",
                "     [f]: face normal",
                "     [v]: vertex normal",
                "     [w]: world frame",
                "     [l]: link frame",
                "     [d]: wireframe",
                "     [c]: camera & frustrum",
                "   [F11]: full-screen mode",
            ],
        ]

    @override
    def on_key_press(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self.viewer is None:
            return None

        # A causes the frame to rotate
        self.viewer._message_text = None
        if symbol == pyglet.window.key.A:
            self.viewer.viewer_flags["rotate"] = not self.viewer.viewer_flags["rotate"]
            if self.viewer.viewer_flags["rotate"]:
                self.viewer._message_text = "Rotation On"
            else:
                self.viewer._message_text = "Rotation Off"

        # F11 toggles fullscreen
        elif symbol == pyglet.window.key.F11:
            self.viewer.viewer_flags["fullscreen"] = not self.viewer.viewer_flags["fullscreen"]
            self.viewer.set_fullscreen(self.viewer.viewer_flags["fullscreen"])
            self.viewer.activate()
            if self.viewer.viewer_flags["fullscreen"]:
                self.viewer._message_text = "Fullscreen On"
            else:
                self.viewer._message_text = "Fullscreen Off"

        # H toggles shadows
        elif symbol == pyglet.window.key.H:
            self.viewer.render_flags["shadows"] = not self.viewer.render_flags["shadows"]
            if self.viewer.render_flags["shadows"]:
                self.viewer._message_text = "Shadows On"
            else:
                self.viewer._message_text = "Shadows Off"

        # W toggles world frame
        elif symbol == pyglet.window.key.W:
            if not self.viewer.gs_context.world_frame_shown:
                self.viewer.gs_context.on_world_frame()
                self.viewer._message_text = "World Frame On"
            else:
                self.viewer.gs_context.off_world_frame()
                self.viewer._message_text = "World Frame Off"

        # L toggles link frame
        elif symbol == pyglet.window.key.L:
            if not self.viewer.gs_context.link_frame_shown:
                self.viewer.gs_context.on_link_frame()
                self.viewer._message_text = "Link Frame On"
            else:
                self.viewer.gs_context.off_link_frame()
                self.viewer._message_text = "Link Frame Off"

        # C toggles camera frustum
        elif symbol == pyglet.window.key.C:
            if not self.viewer.gs_context.camera_frustum_shown:
                self.viewer.gs_context.on_camera_frustum()
                self.viewer._message_text = "Camera Frustrum On"
            else:
                self.viewer.gs_context.off_camera_frustum()
                self.viewer._message_text = "Camera Frustrum Off"

        # F toggles face normals
        elif symbol == pyglet.window.key.F:
            self.viewer.render_flags["face_normals"] = not self.viewer.render_flags["face_normals"]
            if self.viewer.render_flags["face_normals"]:
                self.viewer._message_text = "Face Normals On"
            else:
                self.viewer._message_text = "Face Normals Off"

        # V toggles vertex normals
        elif symbol == pyglet.window.key.V:
            self.viewer.render_flags["vertex_normals"] = not self.viewer.render_flags["vertex_normals"]
            if self.viewer.render_flags["vertex_normals"]:
                self.viewer._message_text = "Vert Normals On"
            else:
                self.viewer._message_text = "Vert Normals Off"

        # R starts recording frames
        elif symbol == pyglet.window.key.R:
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

        # S saves the current frame as an image
        elif symbol == pyglet.window.key.S:
            self.viewer._save_image()

        # D toggles through wireframe modes
        elif symbol == pyglet.window.key.D:
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

        # Z resets the camera viewpoint
        elif symbol == pyglet.window.key.Z:
            self.viewer._reset_view()

        # I toggles instruction display
        elif symbol == pyglet.window.key.I:
            self._display_instr = not self._display_instr

        # P reloads shader program
        elif symbol == pyglet.window.key.P:
            self.viewer._renderer.reload_program()

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
