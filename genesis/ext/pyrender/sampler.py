"""Samplers, conforming to the glTF 2.0 standards as specified in
https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#reference-sampler

Author: Matthew Matl
"""

from .constants import GLTF


class Sampler(object):
    """Texture sampler properties for filtering and wrapping modes.

    Parameters
    ----------
    name : str, optional
        The user-defined name of this object.
    magFilter : int, optional
        Magnification filter. Valid values:
            - :attr:`.GLTF.NEAREST`
            - :attr:`.GLTF.LINEAR`
    minFilter : int, optional
        Minification filter. Valid values:
            - :attr:`.GLTF.NEAREST`
            - :attr:`.GLTF.LINEAR`
            - :attr:`.GLTF.NEAREST_MIPMAP_NEAREST`
            - :attr:`.GLTF.LINEAR_MIPMAP_NEAREST`
            - :attr:`.GLTF.NEAREST_MIPMAP_LINEAR`
            - :attr:`.GLTF.LINEAR_MIPMAP_LINEAR`
    wrapS : int, optional
        S (U) wrapping mode. Valid values:
            - :attr:`.GLTF.CLAMP_TO_EDGE`
            - :attr:`.GLTF.MIRRORED_REPEAT`
            - :attr:`.GLTF.REPEAT`
    wrapT : int, optional
        T (V) wrapping mode. Valid values:
            - :attr:`.GLTF.CLAMP_TO_EDGE`
            - :attr:`.GLTF.MIRRORED_REPEAT`
            - :attr:`.GLTF.REPEAT`
    """

    def __init__(self, name=None, magFilter=None, minFilter=None, wrapS=GLTF.REPEAT, wrapT=GLTF.REPEAT):
        self.name = name
        self.magFilter = magFilter
        self.minFilter = minFilter
        self.wrapS = wrapS
        self.wrapT = wrapT

    @property
    def name(self):
        """str : The user-defined name of this object."""
        return self._name

    @name.setter
    def name(self, value):
        if value is not None:
            value = str(value)
        self._name = value

    @property
    def magFilter(self):
        """int : Magnification filter type."""
        return self._magFilter

    @magFilter.setter
    def magFilter(self, value):
        self._magFilter = value

    @property
    def minFilter(self):
        """int : Minification filter type."""
        return self._minFilter

    @minFilter.setter
    def minFilter(self, value):
        self._minFilter = value

    @property
    def wrapS(self):
        """int : S (U) wrapping mode."""
        return self._wrapS

    @wrapS.setter
    def wrapS(self, value):
        self._wrapS = value

    @property
    def wrapT(self):
        """int : T (V) wrapping mode."""
        return self._wrapT

    @wrapT.setter
    def wrapT(self, value):
        self._wrapT = value
