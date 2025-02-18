"""Trackball class for 3D manipulation of viewpoints."""

import numpy as np

import genesis.ext.trimesh.transformations as transformations


class Trackball(object):
    """A trackball class for creating camera transforms from mouse movements."""

    STATE_ROTATE = 0
    STATE_PAN = 1
    STATE_ROLL = 2
    STATE_ZOOM = 3

    def __init__(self, pose, size, scale, target=np.array([0.0, 0.0, 0.0])):
        """Initialize a trackball with an initial camera-to-world pose
        and the given parameters.

        Parameters
        ----------
        pose : [4,4]
            An initial camera-to-world pose for the trackball.

        size : (float, float)
            The width and height of the camera image in pixels.

        scale : float
            The diagonal of the scene's bounding box --
            used for ensuring translation motions are sufficiently
            fast for differently-sized scenes.

        target : (3,) float
            The center of the scene in world coordinates.
            The trackball will revolve around this point.
        """
        self._size = np.array(size)
        self._scale = float(scale)

        self._pose = pose
        self._n_pose = pose

        self._target = target
        self._n_target = target

        self._state = Trackball.STATE_ROTATE

        self._pdown = np.array([0.0, 0.0], dtype=np.float32)

    @property
    def pose(self):
        """autolab_core.RigidTransform : The current camera-to-world pose."""
        return self._n_pose

    def set_state(self, state):
        """Set the state of the trackball in order to change the effect of
        dragging motions.

        Parameters
        ----------
        state : int
            One of Trackball.STATE_ROTATE, Trackball.STATE_PAN,
            Trackball.STATE_ROLL, and Trackball.STATE_ZOOM.
        """
        self._state = state

    def resize(self, size):
        """Resize the window.

        Parameters
        ----------
        size : (float, float)
            The new width and height of the camera image in pixels.
        """
        self._size = np.array(size)

    def down(self, point):
        """Record an initial mouse press at a given point.

        Parameters
        ----------
        point : (2,) int
            The x and y pixel coordinates of the mouse press.
        """
        self._pdown = np.array(point, dtype=np.float32)
        self._pose = self._n_pose
        self._target = self._n_target

    def drag(self, point):
        """Update the trackball during a drag.

        Parameters
        ----------
        point : (2,) int
            The current x and y pixel coordinates of the mouse during a drag.
            This will compute a movement for the trackball with the relative
            motion between this point and the one marked by down().
        """
        point = np.array(point, dtype=np.float32)
        dx, dy = point - self._pdown
        mindim = 0.3 * np.min(self._size)

        target = self._target
        x_axis = self._pose[:3, 0].flatten()
        y_axis = self._pose[:3, 1].flatten()
        z_axis = self._pose[:3, 2].flatten()
        eye = self._pose[:3, 3].flatten()

        # Interpret drag as a rotation
        if self._state == Trackball.STATE_ROTATE:
            x_angle = -dx / mindim
            x_rot_mat = transformations.rotation_matrix(x_angle, y_axis, target)

            y_angle = dy / mindim
            y_rot_mat = transformations.rotation_matrix(y_angle, x_axis, target)

            self._n_pose = y_rot_mat.dot(x_rot_mat.dot(self._pose))

        # Interpret drag as a roll about the camera axis
        elif self._state == Trackball.STATE_ROLL:
            center = self._size / 2.0
            v_init = self._pdown - center
            v_curr = point - center
            v_init = v_init / np.linalg.norm(v_init)
            v_curr = v_curr / np.linalg.norm(v_curr)

            theta = -np.arctan2(v_curr[1], v_curr[0]) + np.arctan2(v_init[1], v_init[0])

            rot_mat = transformations.rotation_matrix(theta, z_axis, target)

            self._n_pose = rot_mat.dot(self._pose)

        # Interpret drag as a camera pan in view plane
        elif self._state == Trackball.STATE_PAN:
            dx = -dx / (5.0 * mindim) * self._scale
            dy = -dy / (5.0 * mindim) * self._scale

            translation = dx * x_axis + dy * y_axis
            self._n_target = self._target + translation
            t_tf = np.eye(4)
            t_tf[:3, 3] = translation
            self._n_pose = t_tf.dot(self._pose)

        # Interpret drag as a zoom motion
        elif self._state == Trackball.STATE_ZOOM:
            radius = np.linalg.norm(eye - target)
            ratio = 0.0
            if dy > 0:
                ratio = np.exp(abs(dy) / (0.5 * self._size[1])) - 1.0
            elif dy < 0:
                ratio = 1.0 - np.exp(dy / (0.5 * (self._size[1])))
            translation = -np.sign(dy) * ratio * radius * z_axis
            t_tf = np.eye(4)
            t_tf[:3, 3] = translation
            self._n_pose = t_tf.dot(self._pose)

    def scroll(self, clicks):
        """Zoom using a mouse scroll wheel motion.

        Parameters
        ----------
        clicks : int
            The number of clicks. Positive numbers indicate forward wheel
            movement.
        """
        target = self._target
        ratio = 0.90

        mult = 1.0
        if clicks > 0:
            mult = ratio**clicks
        elif clicks < 0:
            mult = (1.0 / ratio) ** abs(clicks)

        z_axis = self._n_pose[:3, 2].flatten()
        eye = self._n_pose[:3, 3].flatten()
        radius = np.linalg.norm(eye - target)
        translation = (mult * radius - radius) * z_axis
        t_tf = np.eye(4)
        t_tf[:3, 3] = translation
        self._n_pose = t_tf.dot(self._n_pose)

        z_axis = self._pose[:3, 2].flatten()
        eye = self._pose[:3, 3].flatten()
        radius = np.linalg.norm(eye - target)
        translation = (mult * radius - radius) * z_axis
        t_tf = np.eye(4)
        t_tf[:3, 3] = translation
        self._pose = t_tf.dot(self._pose)

    def rotate(self, azimuth, axis=None):
        """Rotate the trackball about the "Up" axis by azimuth radians.

        Parameters
        ----------
        azimuth : float
            The number of radians to rotate.
        """
        target = self._target

        y_axis = self._n_pose[:3, 1].flatten()
        if axis is not None:
            y_axis = axis
        x_rot_mat = transformations.rotation_matrix(azimuth, y_axis, target)
        self._n_pose = x_rot_mat.dot(self._n_pose)

        y_axis = self._pose[:3, 1].flatten()
        if axis is not None:
            y_axis = axis
        x_rot_mat = transformations.rotation_matrix(azimuth, y_axis, target)
        self._pose = x_rot_mat.dot(self._pose)

    def set_camera_pose(self, pose):
        """Set the camera pose.

        Parameters
        ----------
        pose : [4,4]
            The camera pose.
        """
        self._n_pose = pose
