from .base import Platform


__all__ = ["OSMesaPlatform"]


class OSMesaPlatform(Platform):
    """Renders into a software buffer using OSMesa. Requires special versions
    of OSMesa to be installed, plus PyOpenGL upgrade.
    """

    def __init__(self, viewport_width, viewport_height):
        super(OSMesaPlatform, self).__init__(viewport_width, viewport_height)
        self._context = None
        self._buffer = None

    def init_context(self):
        from OpenGL import arrays
        from OpenGL.osmesa import (
            OSMesaCreateContextAttribs,
            OSMESA_FORMAT,
            OSMESA_RGBA,
            OSMESA_PROFILE,
            OSMESA_CORE_PROFILE,
            OSMESA_CONTEXT_MAJOR_VERSION,
            OSMESA_CONTEXT_MINOR_VERSION,
            OSMESA_DEPTH_BITS,
        )

        attrs = arrays.GLintArray.asArray(
            [
                OSMESA_FORMAT,
                OSMESA_RGBA,
                OSMESA_DEPTH_BITS,
                24,
                OSMESA_PROFILE,
                OSMESA_CORE_PROFILE,
                OSMESA_CONTEXT_MAJOR_VERSION,
                3,
                OSMESA_CONTEXT_MINOR_VERSION,
                3,
                0,
            ]
        )
        self._context = OSMesaCreateContextAttribs(attrs, None)
        self._buffer = arrays.GLubyteArray.zeros((self.viewport_height, self.viewport_width, 4))

    def make_current(self):
        from OpenGL import GL as gl
        from OpenGL.osmesa import OSMesaMakeCurrent

        assert OSMesaMakeCurrent(
            self._context, self._buffer, gl.GL_UNSIGNED_BYTE, self.viewport_width, self.viewport_height
        )

    def make_uncurrent(self):
        """Make the OpenGL context uncurrent."""
        pass

    def delete_context(self):
        from OpenGL.osmesa import OSMesaDestroyContext

        OSMesaDestroyContext(self._context)
        self._context = None
        self._buffer = None

    def supports_framebuffers(self):
        return False
