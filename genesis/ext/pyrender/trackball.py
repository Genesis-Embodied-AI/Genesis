"""Trackball class for 3D manipulation of viewpoints."""

import numpy as np

import trimesh.transformations as transformations


EPSILON = np.finfo(np.float32).eps


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
        x_axis = self._pose[:3, 0]
        y_axis = self._pose[:3, 1]
        z_axis = self._pose[:3, 2]
        eye = self._pose[:3, 3]

        # Interpret drag as a rotation
        if self._state == Trackball.STATE_ROTATE:
            # Compute updated azimut directly. No fancy math here because this angle can controlled freely.
            roll_angle = np.arctan2(self._pose[2, 1], self._pose[2, 2])
            world_up_axis = np.array([0.0, 0.0, np.sign(roll_angle)])
            azimuth_angle = -dx / mindim
            azimuth_transform = transformations.rotation_matrix(azimuth_angle, world_up_axis, target)

            # Compute current elevation angle
            pose_after_azimuth = azimuth_transform.dot(self._pose)
            eye_after_azimuth = pose_after_azimuth[:3, 3]
            view_dir = target - eye_after_azimuth
            current_elevation_angle = -np.arctan2(view_dir[2], np.linalg.norm(view_dir[:2]))

            # Update elevation angle based on mouse motion
            desired_elevation_angle = current_elevation_angle - dy / mindim
            clamped_elevation_angle = np.clip(desired_elevation_angle, np.radians(-89.0), np.radians(89.0))
            delta_elevation_angle = desired_elevation_angle - current_elevation_angle

            # Compute the elevation axis
            norm_view_dir = np.linalg.norm(view_dir)
            if norm_view_dir < EPSILON:
                 elevation_axis = pose_after_azimuth[:3, 0]
                 delta_elevation_angle = 0.0
            else:
                view_dir_normalized = view_dir / norm_view_dir
                elevation_axis = np.cross(world_up_axis, view_dir_normalized)
                elevation_axis_norm = np.linalg.norm(elevation_axis)
                if elevation_axis_norm < EPSILON:
                    elevation_axis = pose_after_azimuth[:3, 0]
                else:
                    elevation_axis = elevation_axis / elevation_axis_norm

            # Apply the elevation rotation
            elevation_rotation = transformations.rotation_matrix(delta_elevation_angle, elevation_axis, target)
            self._n_pose = elevation_rotation.dot(pose_after_azimuth)

            # Prevent locking the camera in the up/down direction
            self._pdown[1] -= (desired_elevation_angle - clamped_elevation_angle) * mindim

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
