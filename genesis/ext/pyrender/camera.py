"""Virtual cameras compliant with the glTF 2.0 specification as described at
https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#reference-camera

Author: Matthew Matl
"""

import abc
import numpy as np
import six
import sys

from .constants import DEFAULT_Z_NEAR, DEFAULT_Z_FAR


@six.add_metaclass(abc.ABCMeta)
class Camera(object):
    """Abstract base class for all cameras.

    Note
    ----
    Camera poses are specified in the OpenGL format,
    where the z axis points away from the view direction and the
    x and y axes point to the right and up in the image plane, respectively.

    Parameters
    ----------
    znear : float
        The floating-point distance to the near clipping plane.
    zfar : float
        The floating-point distance to the far clipping plane.
        ``zfar`` must be greater than ``znear``.
    name : str, optional
        The user-defined name of this object.
    """

    def __init__(self, znear=DEFAULT_Z_NEAR, zfar=DEFAULT_Z_FAR, name=None):
        self.name = name
        self.znear = znear
        self.zfar = zfar

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
    def znear(self):
        """float : The distance to the near clipping plane."""
        return self._znear

    @znear.setter
    def znear(self, value):
        value = float(value)
        if value < 0:
            raise ValueError("z-near must be >= 0.0")
        self._znear = value

    @property
    def zfar(self):
        """float : The distance to the far clipping plane."""
        return self._zfar

    @zfar.setter
    def zfar(self, value):
        value = float(value)
        if value <= 0 or value <= self.znear:
            raise ValueError("zfar must be >0 and >znear")
        self._zfar = value

    @abc.abstractmethod
    def get_projection_matrix(self, width=None, height=None):
        """Return the OpenGL projection matrix for this camera.

        Parameters
        ----------
        width : int
            Width of the current viewport, in pixels.
        height : int
            Height of the current viewport, in pixels.
        """
        pass


class PerspectiveCamera(Camera):
    """A perspective camera for perspective projection.

    Parameters
    ----------
    yfov : float
        The floating-point vertical field of view in radians.
    znear : float
        The floating-point distance to the near clipping plane.
        If not specified, defaults to 0.05.
    zfar : float, optional
        The floating-point distance to the far clipping plane.
        ``zfar`` must be greater than ``znear``.
        If None, the camera uses an infinite projection matrix.
    aspectRatio : float, optional
        The floating-point aspect ratio of the field of view.
        If not specified, the camera uses the viewport's aspect ratio.
    name : str, optional
        The user-defined name of this object.
    """

    def __init__(self, yfov, znear=DEFAULT_Z_NEAR, zfar=None, aspectRatio=None, name=None):
        super(PerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )

        self.yfov = yfov
        self.aspectRatio = aspectRatio

    @property
    def yfov(self):
        """float : The vertical field of view in radians."""
        return self._yfov

    @yfov.setter
    def yfov(self, value):
        value = float(value)
        if value <= 0.0:
            raise ValueError("Field of view must be positive")
        self._yfov = value

    @property
    def zfar(self):
        """float : The distance to the far clipping plane."""
        return self._zfar

    @zfar.setter
    def zfar(self, value):
        if value is not None:
            value = float(value)
            if value <= 0 or value <= self.znear:
                raise ValueError("zfar must be >0 and >znear")
        self._zfar = value

    @property
    def aspectRatio(self):
        """float : The ratio of the width to the height of the field of view."""
        return self._aspectRatio

    @aspectRatio.setter
    def aspectRatio(self, value):
        if value is not None:
            value = float(value)
            if value <= 0.0:
                raise ValueError("Aspect ratio must be positive")
        self._aspectRatio = value

    def get_projection_matrix(self, width=None, height=None):
        """Return the OpenGL projection matrix for this camera.

        Parameters
        ----------
        width : int
            Width of the current viewport, in pixels.
        height : int
            Height of the current viewport, in pixels.
        """
        aspect_ratio = self.aspectRatio
        if aspect_ratio is None:
            if width is None or height is None:
                raise ValueError("Aspect ratio of camera must be defined")
            aspect_ratio = float(width) / float(height)

        a = aspect_ratio
        t = np.tan(self.yfov / 2.0)
        n = self.znear
        f = self.zfar

        P = np.zeros((4, 4))
        P[0][0] = 1.0 / (a * t)
        P[1][1] = 1.0 / t
        P[3][2] = -1.0

        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P


class OrthographicCamera(Camera):
    """A perspective camera for perspective projection.

    Parameters
    ----------
    xmag : float
        The floating-point horizontal magnification of the view.
    ymag : float
        The floating-point vertical magnification of the view.
    znear : float
        The floating-point distance to the near clipping plane.
        If not specified, defaults to 0.05.
    zfar : float
        The floating-point distance to the far clipping plane.
        ``zfar`` must be greater than ``znear``.
        If not specified, defaults to 100.0.
    name : str, optional
        The user-defined name of this object.
    """

    def __init__(self, xmag, ymag, znear=DEFAULT_Z_NEAR, zfar=DEFAULT_Z_FAR, name=None):
        super(OrthographicCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )

        self.xmag = xmag
        self.ymag = ymag

    @property
    def xmag(self):
        """float : The horizontal magnification of the view."""
        return self._xmag

    @xmag.setter
    def xmag(self, value):
        value = float(value)
        if value <= 0.0:
            raise ValueError("X magnification must be positive")
        self._xmag = value

    @property
    def ymag(self):
        """float : The vertical magnification of the view."""
        return self._ymag

    @ymag.setter
    def ymag(self, value):
        value = float(value)
        if value <= 0.0:
            raise ValueError("Y magnification must be positive")
        self._ymag = value

    @property
    def znear(self):
        """float : The distance to the near clipping plane."""
        return self._znear

    @znear.setter
    def znear(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("z-near must be > 0.0")
        self._znear = value

    def get_projection_matrix(self, width=None, height=None):
        """Return the OpenGL projection matrix for this camera.

        Parameters
        ----------
        width : int
            Width of the current viewport, in pixels.
            Unused in this function.
        height : int
            Height of the current viewport, in pixels.
            Unused in this function.
        """
        xmag = self.xmag
        ymag = self.ymag

        # If screen width/height defined, rescale xmag
        if width is not None and height is not None:
            xmag = width / height * ymag

        n = self.znear
        f = self.zfar
        P = np.zeros((4, 4))
        P[0][0] = 1.0 / xmag
        P[1][1] = 1.0 / ymag
        P[2][2] = 2.0 / (n - f)
        P[2][3] = (f + n) / (n - f)
        P[3][3] = 1.0
        return P


class IntrinsicsCamera(Camera):
    """A perspective camera with custom intrinsics.

    Parameters
    ----------
    fx : float
        X-axis focal length in pixels.
    fy : float
        Y-axis focal length in pixels.
    cx : float
        X-axis optical center in pixels.
    cy : float
        Y-axis optical center in pixels.
    znear : float
        The floating-point distance to the near clipping plane.
        If not specified, defaults to 0.05.
    zfar : float
        The floating-point distance to the far clipping plane.
        ``zfar`` must be greater than ``znear``.
        If not specified, defaults to 100.0.
    name : str, optional
        The user-defined name of this object.
    """

    def __init__(self, fx, fy, cx, cy, znear=DEFAULT_Z_NEAR, zfar=DEFAULT_Z_FAR, name=None):
        super(IntrinsicsCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @property
    def fx(self):
        """float : X-axis focal length in meters."""
        return self._fx

    @fx.setter
    def fx(self, value):
        self._fx = float(value)

    @property
    def fy(self):
        """float : Y-axis focal length in meters."""
        return self._fy

    @fy.setter
    def fy(self, value):
        self._fy = float(value)

    @property
    def cx(self):
        """float : X-axis optical center in pixels."""
        return self._cx

    @cx.setter
    def cx(self, value):
        self._cx = float(value)

    @property
    def cy(self):
        """float : Y-axis optical center in pixels."""
        return self._cy

    @cy.setter
    def cy(self, value):
        self._cy = float(value)

    def get_projection_matrix(self, width, height):
        """Return the OpenGL projection matrix for this camera.

        Parameters
        ----------
        width : int
            Width of the current viewport, in pixels.
        height : int
            Height of the current viewport, in pixels.
        """
        width = float(width)
        height = float(height)

        cx, cy = self.cx, self.cy
        fx, fy = self.fx, self.fy
        if sys.platform == "darwin":
            cx = self.cx * 2.0
            cy = self.cy * 2.0
            fx = self.fx * 2.0
            fy = self.fy * 2.0

        P = np.zeros((4, 4))
        P[0][0] = 2.0 * fx / width
        P[1][1] = 2.0 * fy / height
        P[0][2] = 1.0 - 2.0 * cx / width
        P[1][2] = 2.0 * cy / height - 1.0
        P[3][2] = -1.0

        n = self.znear
        f = self.zfar
        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P


__all__ = ["Camera", "PerspectiveCamera", "OrthographicCamera", "IntrinsicsCamera"]
