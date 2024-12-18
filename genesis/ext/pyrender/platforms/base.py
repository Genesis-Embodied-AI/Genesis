import abc

import six


@six.add_metaclass(abc.ABCMeta)
class Platform(object):
    """Base class for all OpenGL platforms.

    Parameters
    ----------
    viewport_width : int
        The width of the main viewport, in pixels.
    viewport_height : int
        The height of the main viewport, in pixels
    """

    def __init__(self, viewport_width, viewport_height):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

    @property
    def viewport_width(self):
        """int : The width of the main viewport, in pixels."""
        return self._viewport_width

    @viewport_width.setter
    def viewport_width(self, value):
        self._viewport_width = value

    @property
    def viewport_height(self):
        """int : The height of the main viewport, in pixels."""
        return self._viewport_height

    @viewport_height.setter
    def viewport_height(self, value):
        self._viewport_height = value

    @abc.abstractmethod
    def init_context(self):
        """Create an OpenGL context."""
        pass

    @abc.abstractmethod
    def make_current(self):
        """Make the OpenGL context current."""
        pass

    @abc.abstractmethod
    def make_uncurrent(self):
        """Make the OpenGL context uncurrent."""
        pass

    @abc.abstractmethod
    def delete_context(self):
        """Delete the OpenGL context."""
        pass

    @abc.abstractmethod
    def supports_framebuffers(self):
        """Returns True if the method supports framebuffer rendering."""
        pass

    def __del__(self):
        try:
            self.delete_context()
        except Exception:
            pass
