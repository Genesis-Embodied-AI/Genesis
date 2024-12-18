"""Font texture loader and processor.

Author: Matthew Matl
"""

import freetype
import numpy as np
import os

import OpenGL
from OpenGL.GL import *

from .constants import TextAlign, FLOAT_SZ
from .texture import Texture
from .sampler import Sampler


class FontCache(object):
    """A cache for fonts."""

    def __init__(self, font_dir=None):
        self._font_cache = {}
        self.font_dir = font_dir
        if self.font_dir is None:
            base_dir, _ = os.path.split(os.path.realpath(__file__))
            self.font_dir = os.path.join(base_dir, "fonts")

    def get_font(self, font_name, font_pt):
        # If it's a file, load it directly, else, try to load from font dir.
        if os.path.isfile(font_name):
            font_filename = font_name
            _, font_name = os.path.split(font_name)
            font_name, _ = os.path.split(font_name)
        else:
            font_filename = os.path.join(self.font_dir, font_name) + ".ttf"

        cid = OpenGL.contextdata.getContext()
        key = (cid, font_name, int(font_pt))

        if key not in self._font_cache:
            self._font_cache[key] = Font(font_filename, font_pt)
        return self._font_cache[key]

    def clear(self):
        for key in self._font_cache:
            self._font_cache[key].delete()
        self._font_cache = {}


class Character(object):
    """A single character, with its texture and attributes."""

    def __init__(self, texture, size, bearing, advance):
        self.texture = texture
        self.size = size
        self.bearing = bearing
        self.advance = advance


class Font(object):
    """A font object.

    Parameters
    ----------
    font_file : str
        The file to load the font from.
    font_pt : int
        The height of the font in pixels.
    """

    def __init__(self, font_file, font_pt=40):
        self.font_file = font_file
        self.font_pt = int(font_pt)
        self._face = freetype.Face(font_file)
        self._face.set_pixel_sizes(0, font_pt)
        self._character_map = {}

        for i in range(0, 128):

            # Generate texture
            face = self._face
            face.load_char(chr(i))
            buf = face.glyph.bitmap.buffer
            src = (np.array(buf) / 255.0).astype(np.float32)
            src = src.reshape((face.glyph.bitmap.rows, face.glyph.bitmap.width))
            tex = Texture(
                sampler=Sampler(
                    magFilter=GL_LINEAR, minFilter=GL_LINEAR, wrapS=GL_CLAMP_TO_EDGE, wrapT=GL_CLAMP_TO_EDGE
                ),
                source=src,
                source_channels="R",
            )
            character = Character(
                texture=tex,
                size=np.array([face.glyph.bitmap.width, face.glyph.bitmap.rows]),
                bearing=np.array([face.glyph.bitmap_left, face.glyph.bitmap_top]),
                advance=face.glyph.advance.x,
            )
            self._character_map[chr(i)] = character

        self._vbo = None
        self._vao = None

    @property
    def font_file(self):
        """str : The file the font was loaded from."""
        return self._font_file

    @font_file.setter
    def font_file(self, value):
        self._font_file = value

    @property
    def font_pt(self):
        """int : The height of the font in pixels."""
        return self._font_pt

    @font_pt.setter
    def font_pt(self, value):
        self._font_pt = int(value)

    def _add_to_context(self):

        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)
        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, FLOAT_SZ * 6 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * FLOAT_SZ, ctypes.c_void_p(0))
        glBindVertexArray(0)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        for c in self._character_map:
            ch = self._character_map[c]
            if not ch.texture._in_context():
                ch.texture._add_to_context()

    def _remove_from_context(self):
        for c in self._character_map:
            ch = self._character_map[c]
            ch.texture.delete()
        if self._vao is not None:
            glDeleteVertexArrays(1, [self._vao])
            glDeleteBuffers(1, [self._vbo])
            self._vao = None
            self._vbo = None

    def _in_context(self):
        return self._vao is not None

    def _bind(self):
        glBindVertexArray(self._vao)

    def _unbind(self):
        glBindVertexArray(0)

    def delete(self):
        self._unbind()
        self._remove_from_context()

    def render_string(self, text, x, y, scale=1.0, align=TextAlign.BOTTOM_LEFT):
        """Render a string to the current view buffer.

        Note
        ----
        Assumes correct shader program already bound w/ uniforms set.

        Parameters
        ----------
        text : str
            The text to render.
        x : int
            Horizontal pixel location of text.
        y : int
            Vertical pixel location of text.
        scale : int
            Scaling factor for text.
        align : int
            One of the TextAlign options which specifies where the ``x``
            and ``y`` parameters lie on the text. For example,
            :attr:`.TextAlign.BOTTOM_LEFT` means that ``x`` and ``y`` indicate
            the position of the bottom-left corner of the textbox.
        """
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self._bind()

        # Determine width and height of text relative to x, y
        width = 0.0
        height = 0.0
        for c in text:
            ch = self._character_map[c]
            height = max(height, ch.bearing[1] * scale)
            width += (ch.advance >> 6) * scale

        # Determine offsets based on alignments
        xoff = 0
        yoff = 0
        if align == TextAlign.BOTTOM_RIGHT:
            xoff = -width
        elif align == TextAlign.BOTTOM_CENTER:
            xoff = -width / 2.0
        elif align == TextAlign.TOP_LEFT:
            yoff = -height
        elif align == TextAlign.TOP_RIGHT:
            yoff = -height
            xoff = -width
        elif align == TextAlign.TOP_CENTER:
            yoff = -height
            xoff = -width / 2.0
        elif align == TextAlign.CENTER:
            xoff = -width / 2.0
            yoff = -height / 2.0
        elif align == TextAlign.CENTER_LEFT:
            yoff = -height / 2.0
        elif align == TextAlign.CENTER_RIGHT:
            xoff = -width
            yoff = -height / 2.0

        x += xoff
        y += yoff

        ch = None
        for c in text:
            ch = self._character_map[c]
            xpos = x + ch.bearing[0] * scale
            ypos = y - (ch.size[1] - ch.bearing[1]) * scale
            w = ch.size[0] * scale
            h = ch.size[1] * scale

            vertices = np.array(
                [
                    [xpos, ypos, 0.0, 0.0],
                    [xpos + w, ypos, 1.0, 0.0],
                    [xpos + w, ypos + h, 1.0, 1.0],
                    [xpos + w, ypos + h, 1.0, 1.0],
                    [xpos, ypos + h, 0.0, 1.0],
                    [xpos, ypos, 0.0, 0.0],
                ],
                dtype=np.float32,
            )

            ch.texture._bind()

            glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
            glBufferData(GL_ARRAY_BUFFER, FLOAT_SZ * 6 * 4, vertices, GL_DYNAMIC_DRAW)
            # TODO MAKE THIS MORE EFFICIENT, lgBufferSubData is broken
            # glBufferSubData(
            #     GL_ARRAY_BUFFER, 0, 6 * 4 * FLOAT_SZ,
            #     np.ascontiguousarray(vertices.flatten)
            # )
            glDrawArrays(GL_TRIANGLES, 0, 6)
            x += (ch.advance >> 6) * scale

        self._unbind()
        if ch:
            ch.texture._unbind()
