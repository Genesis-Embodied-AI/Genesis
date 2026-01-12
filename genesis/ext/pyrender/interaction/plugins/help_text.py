from typing import TYPE_CHECKING

import numpy as np
import pyglet
from typing_extensions import override

from genesis.options.viewer_plugins import HelpTextPlugin as HelpTextPluginOptions

from ...constants import TEXT_PADDING, TextAlign
from ..keybindings import Keybind, get_keycode_string
from ..viewer_plugin import ViewerPlugin, register_viewer_plugin

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node

INSTR_KEYBIND_NAME = "toggle_instructions"


@register_viewer_plugin(HelpTextPluginOptions)
class HelpTextPlugin(ViewerPlugin):
    """
    Default keyboard controls for the Genesis viewer.
    """

    def __init__(
        self,
        viewer,
        options: HelpTextPluginOptions,
        camera: "Node" = None,
        scene: "Scene" = None,
    ):
        super().__init__(viewer, options, camera, scene)

        if self.options.display_instructions:
            self.viewer.register_keybinds(
                (Keybind(key_code=pyglet.window.key.I, name=INSTR_KEYBIND_NAME, callback=self._toggle_instructions),)
            )
            self._collapse_instructions = True
            self._instr_texts: tuple[list[str], list[str]] = ([], [])
            self._update_instr_texts()

        self._message_text = None
        self._ticks_till_fade = 2.0 / 3.0 * self.viewer.viewer_flags["refresh_rate"]
        self._message_opac = 1.0 + self._ticks_till_fade

    def _update_instr_texts(self):
        if len(self.viewer._keybindings) != len(self._instr_texts[1]) - 1:
            self.instr_key_str = get_keycode_string(self.viewer._keybindings.get_by_name(INSTR_KEYBIND_NAME).key_code)
            kb_texts = [
                f"{'[' + get_keycode_string(kb.key_code):>{7}}]: " + kb.name.replace("_", " ")
                for kb in self.viewer._keybindings.keybinds
                if kb.name != INSTR_KEYBIND_NAME
            ]
            self._instr_texts = (
                [f"> [{self.instr_key_str}]: show keyboard instructions"],
                [f"< [{self.instr_key_str}]: hide keyboard instructions"] + kb_texts,
            )

    def _toggle_instructions(self):
        if not self.options.display_instructions:
            raise RuntimeError("Instructions display is disabled by options.")
        self._collapse_instructions = not self._collapse_instructions
        self._update_instr_texts()

    def set_message_text(self, text: str):
        self._message_text = text
        self._message_opac = 1.0 + self._ticks_till_fade

    @override
    def on_draw(self):
        if self._message_text is not None:
            self.viewer._renderer.render_text(
                self._message_text,
                self.viewer._viewport_size[0] - TEXT_PADDING,
                TEXT_PADDING,
                font_pt=self.options.font_size,
                color=np.array([0.1, 0.7, 0.2, np.clip(self._message_opac, 0.0, 1.0)]),
                align=TextAlign.BOTTOM_RIGHT,
            )

            if self._message_opac > 1.0:
                self._message_opac -= 1.0
            else:
                self._message_opac *= 0.90

            if self._message_opac < 0.05:
                self._message_opac = 1.0 + self._ticks_till_fade
                self._message_text = None

        if self.options.display_instructions:
            if self._collapse_instructions:
                self.viewer._renderer.render_texts(
                    self._instr_texts[0],
                    TEXT_PADDING,
                    self.viewer._viewport_size[1] - TEXT_PADDING,
                    font_pt=self.options.font_size,
                    color=np.array([1.0, 1.0, 1.0, 0.85]),
                )
            else:
                self.viewer._renderer.render_texts(
                    self._instr_texts[1],
                    TEXT_PADDING,
                    self.viewer._viewport_size[1] - TEXT_PADDING,
                    font_pt=self.options.font_size,
                    color=np.array([1.0, 1.0, 1.0, 0.85]),
                )
