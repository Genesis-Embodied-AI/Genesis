import inspect
import math
import os
import time
from functools import cached_property

import cv2
import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.constants import IMAGE_TYPE
from genesis.repr_base import RBC
from genesis.utils.misc import tensor_to_array
from genesis.utils.image_exporter import as_grayscale_image


class Camera(RBC):
    """
    A camera which can be used to render RGB, depth, and segmentation images.
    Supports either rasterizer or raytracer for rendering, specified by `scene.renderer`.

    Parameters
    ----------
    visualizer : genesis.Visualizer
        The visualizer object that the camera is associated with.
    idx : int
        The index of the camera.
    model : str
        Specifies the camera model. Options are 'pinhole' or 'thinlens'.
    res : tuple of int, shape (2,)
        The resolution of the camera, specified as a tuple (width, height).
    pos : tuple of float, shape (3,)
        The position of the camera in the scene, specified as (x, y, z).
    lookat : tuple of float, shape (3,)
        The point in the scene that the camera is looking at, specified as (x, y, z).
    up : tuple of float, shape (3,)
        The up vector of the camera, defining its orientation, specified as (x, y, z).
    fov : float
        The vertical field of view of the camera in degrees.
    aperture : float
        The aperture size of the camera, controlling depth of field.
    focus_dist : float | None
        The focus distance of the camera. If None, it will be auto-computed using `pos` and `lookat`.
    GUI : bool
        Whether to display the camera's rendered image in a separate GUI window.
    spp : int, optional
        Samples per pixel. Only available when using the RayTracer renderer. Defaults to 256.
    denoise : bool
        Whether to denoise the camera's rendered image. Only available when using the RayTracer renderer.
        Defaults to True.  If OptiX denoiser is not available on your platform, consider enabling the OIDN denoiser
        option when building RayTracer.
    near : float
        Distance from camera center to near plane in meters.
        Only available when using rasterizer in Rasterizer and BatchRender renderer. Defaults to 0.05.
    far : float
        Distance from camera center to far plane in meters.
        Only available when using rasterizer in Rasterizer and BatchRender renderer. Defaults to 100.0.
    transform : np.ndarray, shape (4, 4), optional
        The transform matrix of the camera.
    env_idx : int, optional
        The index of the environment to track the camera.
    debug : bool, optional
        Whether to use the debug camera. It enables to create cameras that can used to monitor / debug the
        simulation without being part of the "sensors". Their output is rendered by the usual simple Rasterizer
        systematically, no matter if BatchRayTracer is enabled. This way, it is possible to record the
        simulation with arbitrary resolution and camera pose, without interfering with what robots can perceive
        from their environment. Defaults to False.
    """

    def __init__(
        self,
        visualizer,
        idx=0,
        model="pinhole",  # pinhole or thinlens
        res=(320, 320),
        pos=(0.5, 2.5, 3.5),
        lookat=(0.5, 0.5, 0.5),
        up=(0.0, 0.0, 1.0),
        fov=30,
        aperture=2.8,
        focus_dist=None,
        GUI=False,
        spp=256,
        denoise=True,
        near=0.05,
        far=100.0,
        transform=None,
        env_idx=None,
        debug=False,
    ):
        self._idx = idx
        self._uid = gs.UID()
        self._model = model
        self._res = res
        self._fov = fov
        self._aperture = aperture
        self._focus_dist = focus_dist
        self._GUI = GUI
        self._spp = spp
        self._denoise = denoise
        self._near = near
        self._far = far
        self._initial_pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)
        self._initial_lookat = torch.as_tensor(lookat, dtype=gs.tc_float, device=gs.device)
        self._initial_up = torch.as_tensor(up, dtype=gs.tc_float, device=gs.device)
        self._initial_transform = None
        if transform is not None:
            self._initial_transform = torch.as_tensor(transform, dtype=gs.tc_float, device=gs.device)
        self._aspect_ratio = self._res[0] / self._res[1]  # width / height
        self._visualizer = visualizer
        self._debug = debug

        self._is_built = False

        self._rasterizer = None
        self._raytracer = None
        self._batch_renderer = None

        self._env_idx = int(env_idx) if env_idx is not None else None
        self._envs_offset = None

        self._in_recording = False
        self._recorded_t_prev = -1
        self._recorded_imgs = []

        self._attached_link = None
        self._attached_offset_T = None

        self._followed_entity = None
        self._follow_fixed_axis = None
        self._follow_smoothing = None
        self._follow_fix_orientation = None

        if self._model not in ["pinhole", "thinlens"]:
            gs.raise_exception(f"Invalid camera model: {self._model}")

        if self._focus_dist is None:
            self._focus_dist = np.linalg.norm(np.asarray(lookat) - np.asarray(pos))

    def build(self):
        self._rasterizer = self._visualizer.rasterizer
        if not self._debug:
            self._raytracer = self._visualizer.raytracer
            self._batch_renderer = self._visualizer.batch_renderer

        if self._batch_renderer is not None:
            self._is_batched = True
            if self._env_idx is not None:
                gs.raise_exception("Binding a camera to one specific environment index not supported by BatchRender.")
        else:
            if self._raytracer is not None:
                self._raytracer.add_camera(self)
                self._is_batched = False
            else:
                self._rasterizer.add_camera(self)
                self._is_batched = self._visualizer.scene.n_envs > 0 and self._visualizer._context.env_separate_rigid
                if self._is_batched:
                    gs.logger.warning(
                        "Batched rendering via 'VisOptions.env_separate_rigid=True' is only partially supported by "
                        "Rasterizer for now. The same camera transform will be used for all the environments."
                    )
            if self._env_idx is None:
                self._env_idx = int(self._visualizer._context.rendered_envs_idx[0])
                if self._visualizer.scene.n_envs > 0:
                    gs.logger.info(
                        "Raytracer and Rasterizer requires binding to the camera with a specific environment index. "
                        "Defaulting to 'rendered_envs_idx[0]'. Please specify 'env_idx' if necessary."
                    )
            if self._env_idx not in self._visualizer._context.rendered_envs_idx:
                gs.raise_exception("Environment index bound to the camera not in 'VisOptions.rendered_envs_idx'.")

        if self._is_batched and self._env_idx is None:
            batch_size = (len(self._visualizer._context.rendered_envs_idx),)
        else:
            batch_size = ()
        self._pos = torch.empty((*batch_size, 3), dtype=gs.tc_float, device=gs.device)
        self._lookat = torch.empty((*batch_size, 3), dtype=gs.tc_float, device=gs.device)
        self._up = torch.empty((*batch_size, 3), dtype=gs.tc_float, device=gs.device)
        self._transform = torch.empty((*batch_size, 4, 4), dtype=gs.tc_float, device=gs.device)
        self._quat = torch.empty((*batch_size, 4), dtype=gs.tc_float, device=gs.device)

        self._envs_offset = torch.as_tensor(
            self._visualizer._scene.envs_offset[() if self._env_idx is None else self._env_idx],
            dtype=gs.tc_float,
            device=gs.device,
        )

        # Must consider the building process done before setting initial pose, otherwise it will fail
        self._is_built = True
        self.set_pose(
            transform=self._initial_transform,
            pos=self._initial_pos,
            lookat=self._initial_lookat,
            up=self._initial_up,
        )

        # FIXME: For some reason, it is necessary to update the camera twice...
        if self._raytracer is not None:
            self._raytracer.update_camera(self)

    def attach(self, rigid_link, offset_T):
        """
        Attach the camera to a rigid link in the scene.

        Once attached, the camera's position and orientation can be updated relative to the attached link using
        `move_to_attach()`. This is useful for mounting the camera to dynamic entities like robots or articulated
        objects.

        Parameters
        ----------
        rigid_link : genesis.RigidLink
            The rigid link to which the camera should be attached.
        offset_T : np.ndarray, shape (4, 4)
            The transformation matrix specifying the camera's pose relative to the rigid link.
        """
        if self._visualizer._context.env_separate_rigid:
            gs.raise_exception("This method is not supported by Rasterizer when 'VisOptions.env_separate_rigid=True'.")

        if self._followed_entity is not None:
            gs.raise_exception("Impossible to attach a camera that is already following an entity.")

        self._attached_link = rigid_link
        self._attached_offset_T = torch.as_tensor(offset_T, dtype=gs.tc_float, device=gs.device)

    def detach(self):
        """
        Detach the camera from the currently attached rigid link.

        After detachment, the camera will stop following the motion of the rigid link and maintain its current world
        pose. Calling this method has no effect if the camera is not currently attached.
        """
        self._attached_link = None
        self._attached_offset_T = None

    def move_to_attach(self):
        """
        Move the camera to follow the currently attached rigid link.

        This method updates the camera's pose using the transform of the attached rigid link combined with the
        specified offset. It should only be called after `attach()` has been used.

        Raises
        ------
        Exception
            If the camera has not been attached to a rigid link.
        """
        if self._attached_link is None:
            gs.raise_exception("Camera not attached to any rigid link.")

        link_pos = self._attached_link.get_pos(self._env_idx)
        link_quat = self._attached_link.get_quat(self._env_idx)
        if self._env_idx is not None and self._visualizer.scene.n_envs > 0:
            link_pos, link_quat = link_pos[0], link_quat[0]
        link_T = gu.trans_quat_to_T(link_pos, link_quat)
        transform = torch.matmul(link_T, self._attached_offset_T)
        self.set_pose(transform=transform)

    def follow_entity(self, entity, fixed_axis=(None, None, None), smoothing=None, fix_orientation=False):
        """
        Set the camera to follow a specified rigid entity.

        Parameters
        ----------
        entity : genesis.Entity
            The entity to follow.
        fixed_axis : (float, float, float), optional
            The fixed axis for the camera's movement. For each axis, if None, the camera will move freely. If a float,
            the viewer will be fixed on at that value. For example, [None, None, None] will allow the camera to move
            freely while following, [None, None, 0.5] will fix the viewer's z-axis at 0.5.
        smoothing : float, optional
            The smoothing factor for the camera's movement. If None, no smoothing will be applied.
        fix_orientation : bool, optional
            If True, the camera will maintain its orientation relative to the world. If False, the camera will look at
            the base link of the entity.
        """
        if self._visualizer._context.env_separate_rigid:
            gs.raise_exception("This method is not supported by Rasterizer when 'VisOptions.env_separate_rigid=True'.")

        if self._attached_link is not None:
            gs.raise_exception("Impossible to following an entity with a camera that is already attached.")

        self._followed_entity = entity
        self._follow_fixed_axis = fixed_axis
        self._follow_smoothing = smoothing
        self._follow_fix_orientation = fix_orientation

    def unfollow_entity(self):
        """
        Stop following any rigid entity with the camera.

        Calling this method has no effect if the camera is not currently following any entity.
        """
        self._followed_entity = None
        self._follow_fixed_axis = None
        self._follow_smoothing = None
        self._follow_fix_orientation = None

    @gs.assert_built
    def update_following(self):
        """
        Update the camera position to follow the specified rigid entity.
        """
        if self._followed_entity is None:
            gs.raise_exception("Camera not following any rigid entity.")

        # Keep the camera orientation fixed by overriding the lookat point if requested
        env_idx = self._env_idx if self._is_batched and self._env_idx is not None else ()
        if self._follow_fix_orientation:
            camera_transform = self._transform[env_idx].clone()
            camera_lookat = None
            camera_pos = camera_transform[..., :3, 3]
        else:
            camera_lookat = self._lookat[env_idx].clone()
            camera_pos = self._pos[env_idx].clone()

        # Smooth camera movement with a low-pass filter, in particular Exponential Moving Average (EMA) if requested
        entity_pos = self._followed_entity.get_pos(self._env_idx, unsafe=True)
        camera_pos -= self._initial_pos
        if self._follow_smoothing is not None:
            camera_pos[:] = self._follow_smoothing * camera_pos + (1.0 - self._follow_smoothing) * entity_pos
            if not self._follow_fix_orientation:
                camera_lookat[:] = self._follow_smoothing * camera_lookat + (1.0 - self._follow_smoothing) * entity_pos
        else:
            camera_pos[:] = entity_pos
        camera_pos += self._initial_pos

        # Fix the camera's position along the specified axis if requested
        for i_a, fixed_axis in enumerate(self._follow_fixed_axis):
            if fixed_axis is not None:
                camera_pos[..., i_a] = fixed_axis

        # Update the pose of all camera at once
        if self._follow_fix_orientation:
            self.set_pose(transform=camera_transform)
        else:
            self.set_pose(pos=camera_pos, lookat=camera_lookat)

    @gs.assert_built
    def _batch_render(
        self,
        rgb=True,
        depth=False,
        segmentation=False,
        normal=False,
        antialiasing=True,
        force_render=False,
    ):
        """
        Render the camera view with batch renderer.
        """
        assert self._visualizer._batch_renderer is not None

        # Render all cameras at once no matter what
        buffers = list(self._batch_renderer.render(rgb, depth, segmentation, normal, antialiasing, force_render))

        # Only return current camera data
        for i, buffer in enumerate(buffers):
            if buffer is not None:
                buffers[i] = buffer[self.idx]

        return tuple(buffers)

    @gs.assert_built
    def render(
        self,
        rgb=True,
        depth=False,
        segmentation=False,
        colorize_seg=False,
        normal=False,
        antialiasing=False,
        force_render=False,
    ):
        """
        Render the camera view.

        Note
        ----
        The segmentation mask can be colorized, and if not colorized, it will store an object index in each pixel based
        on the segmentation level specified in `VisOptions.segmentation_level`.
        For example, if `segmentation_level='link'`, the segmentation mask will store `link_idx`, which can then be
        used to retrieve the actual link objects using `scene.rigid_solver.links[link_idx]`.

        Note
        ----
        If `env_separate_rigid` in `VisOptions` is set to True, each component will return a stack of images, with the
        number of images equal to `len(rendered_envs_idx)`.

        Parameters
        ----------
        rgb : bool, optional
            Whether to render RGB image(s).
        depth : bool, optional
            Whether to render depth image(s).
        segmentation : bool, optional
            Whether to render the segmentation mask(s).
        colorize_seg : bool, optional
            If True, the segmentation mask will be colorized.
        normal : bool, optional
            Whether to render the surface normal.
        antialiasing : bool, optional
            Whether to apply anti-aliasing. Only supported by 'BatchRenderer' for now.
        force_render : bool, optional
            Whether to force rendering even if the scene has not changed.

        Returns
        -------
        rgb_arr : np.ndarray
            The rendered RGB image(s).
        depth_arr : np.ndarray
            The rendered depth image(s).
        seg_arr : np.ndarray
            The rendered segmentation mask(s).
        normal_arr : np.ndarray
            The rendered surface normal(s).
        """
        # Enforce RGB rendering if recording is enabled and the current frame is missing
        is_recording = self._in_recording and self._recorded_t_prev != self._visualizer.scene._t
        rgb_ = rgb or is_recording

        # Render the current frame
        rgb_arr, depth_arr, seg_arr, seg_color_arr, seg_idxc_arr, normal_arr = None, None, None, None, None, None
        if self._batch_renderer is not None:
            rgb_arr, depth_arr, seg_idxc_arr, normal_arr = self._batch_render(
                rgb_, depth, segmentation, normal, antialiasing, force_render
            )
        elif self._raytracer is not None:
            if rgb_:
                self._raytracer.update_scene()
                rgb_arr = self._raytracer.render_camera(self)

            if depth or segmentation or normal:
                self._rasterizer.update_scene()
                _, depth_arr, seg_idxc_arr, normal_arr = self._rasterizer.render_camera(
                    self, False, depth, segmentation, normal=normal
                )
        else:
            self._rasterizer.update_scene()
            rgb_arr, depth_arr, seg_idxc_arr, normal_arr = self._rasterizer.render_camera(
                self, rgb_, depth, segmentation, normal=normal
            )

        # Colorize the segmentation map is necessary
        if seg_idxc_arr is not None:
            if colorize_seg or (self._GUI and self._visualizer.has_display):
                seg_color_arr = self._visualizer.colorize_seg_idxc_arr(seg_idxc_arr)
            seg_arr = seg_color_arr if colorize_seg else seg_idxc_arr

        # Display images if requested and supported
        if self._GUI and self._visualizer.has_display:
            title = f"Genesis - Camera {self._idx}"
            if self._debug:
                title += " (debug)"
            if self._is_batched:
                title += f" - Environment {self._visualizer._context.rendered_envs_idx[0]}"
            for img_type, (flag, buffer) in enumerate(
                ((rgb, rgb_arr), (depth, depth_arr), (segmentation, seg_color_arr), (normal, normal_arr))
            ):
                if flag:
                    if self._is_batched:
                        buffer = buffer[0]
                    buffer = tensor_to_array(buffer)
                    if img_type == IMAGE_TYPE.DEPTH:
                        buffer = as_grayscale_image(buffer, black_to_white=False)
                    else:
                        buffer = np.flip(buffer, axis=-1)
                    cv2.imshow(f"{title} [{IMAGE_TYPE(img_type)}]", buffer)
            cv2.waitKey(1)

        # Store the current frame for video recording
        if is_recording:
            if not (self._recorded_t_prev < 0 or self._recorded_t_prev == self._visualizer.scene._t - 1):
                gs.raise_exception(
                    "Missing frames in recording. Please call 'camera.render()' after 'every scene.step()'."
                )
            self._recorded_t_prev == self._visualizer.scene._t
            self._recorded_imgs.append(tensor_to_array(rgb_arr))

        return rgb_arr if rgb else None, depth_arr, seg_arr, normal_arr

    def distance_center_to_plane(self, center_dis):
        width, height = self.res
        fx = fy = self.f
        cx = self.cx
        cy = self.cy

        if isinstance(center_dis, np.ndarray):
            v, u = np.meshgrid(np.arange(height, dtype=np.int32), np.arange(width, dtype=np.int32), indexing="ij")
            xd = (u + 0.5 - cx) / fx
            yd = (v + 0.5 - cy) / fy
            scale_inv = 1.0 / np.sqrt(xd**2 + yd**2 + 1.0)
        else:  # torch.Tensor
            v, u = torch.meshgrid(
                torch.arange(height, dtype=torch.int32, device=gs.device),
                torch.arange(width, dtype=torch.int32, device=gs.device),
                indexing="ij",
            )
            xd = (u + 0.5 - cx) / fx
            yd = (v + 0.5 - cy) / fy
            scale_inv = torch.rsqrt(xd**2 + yd**2 + 1.0)
        return center_dis * scale_inv

    @gs.assert_built
    def render_pointcloud(self, world_frame=True):
        """
        Render a partial point cloud from the camera view.

        Parameters
        ----------
        world_frame : bool, optional
            Whether the point cloud is on camera frame or world frame.

        Returns
        -------
        pc : np.ndarray
            Numpy array of shape (res[0], res[1], 3) or (N, res[0], res[1], 3).
            Represents the point cloud in each pixel.
        mask_arr : np.ndarray
            The valid depth mask. boolean array of same shape as depth_arr
        """
        # Compute the (denormalized) depth map
        if self._batch_renderer is not None:
            _, depth_arr, _, _ = self._batch_render(rgb=False, depth=True, segmentation=False, normal=False)
            # FIXME: Avoid converting to numpy
            depth_arr = tensor_to_array(depth_arr)
        else:
            self._rasterizer.update_scene()
            _, depth_arr, _, _ = self._rasterizer.render_camera(
                self, rgb=False, depth=True, segmentation=False, normal=False
            )

        # Convert OpenGL projection matrix to camera intrinsics
        width, height = self.res
        fx = fy = self.f
        cx = self.cx
        cy = self.cy

        # Mask out invalid depth
        mask = (self.near < depth_arr) & (depth_arr < self.far * (1.0 - 1e-3))

        # Compute normalized pixel coordinates
        v, u = np.meshgrid(np.arange(height, dtype=np.int32), np.arange(width, dtype=np.int32), indexing="ij")
        u = u.reshape((-1,))
        v = v.reshape((-1,))

        # Convert to world coordinates
        depth_grid = depth_arr[..., v, u]
        world_x = depth_grid * (u + 0.5 - cx) / fx
        world_y = depth_grid * (v + 0.5 - cy) / fy
        world_z = depth_grid

        point_cloud = np.stack((world_x, world_y, world_z, np.ones_like(world_z)), axis=-1)
        if world_frame:
            T_OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
            cam_pose = self.transform @ T_OPENGL_TO_OPENCV
            point_cloud = point_cloud @ cam_pose.swapaxes(-1, -2)

        point_cloud = point_cloud[..., :3].reshape((*depth_arr.shape, 3))
        return point_cloud, mask

    def set_pose(self, transform=None, pos=None, lookat=None, up=None, envs_idx=None):
        """
        Set the pose of the camera.
        Note that `transform` has a higher priority than `pos`, `lookat`, and `up`.
        If `transform` is provided, the camera pose will be set based on the transform matrix.
        Otherwise, the camera pose will be set based on `pos`, `lookat`, and `up`.

        Parameters
        ----------
        transform : np.ndarray, shape (4, 4) or (N, 4, 4), optional
            The transform matrix of the camera.
        pos : array-like, shape (3,) or (N, 3), optional
            The position of the camera.
        lookat : array-like, shape (3,) or (N, 3), optional
            The lookat point of the camera.
        up : array-like, shape (3,) or (N, 3), optional
            The up vector of the camera.
        envs_idx : array of indices in integers, optional
            The environment indices for which to update the pose. If not provided, the camera pose will be set for the
            specific environment bound to the camera if any, all the environments otherwise.
        """
        # Early return if nothing to do
        if pos is None and lookat is None and up is None and transform is None:
            return

        # Sanitize 'envs_idx' input argument
        if envs_idx is not None:
            if not self._is_batched:
                gs.raise_exception("Camera does not support batching. Impossible to specify 'envs_idx'.")
            if self._env_idx is not None:
                gs.raise_exception("Camera already bound to a specific environment. Impossible to specify 'envs_idx'.")
        if self._is_batched:
            if envs_idx is None:
                envs_idx = (self._env_idx,) if self._env_idx is not None else ()
                n_envs = len(self._visualizer._context.rendered_envs_idx)
            else:
                envs_idx = self._visualizer._scene._sanitize_envs_idx(envs_idx)
                if set(envs_idx).issubset(self._visualizer._context.rendered_envs_idx):
                    gs.raise_exception("Environment index not in 'VisOptions.rendered_envs_idx'.")
                n_envs = len(envs_idx)
        else:
            envs_idx = ()

        # Sanitize 'pos', 'lookat', 'up', 'transform' input arguments
        if pos is not None:
            pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)
            if pos.ndim > (1 + self._is_batched) or pos.shape[-1] != 3:
                gs.raise_exception(f"Pos shape {pos.shape} does not match (N, 3)")
            if self._is_batched and pos.ndim == 1:
                pos = pos.expand((n_envs, 3))
        if lookat is not None:
            lookat = torch.as_tensor(lookat, dtype=gs.tc_float, device=gs.device)
            if lookat.ndim > (1 + self._is_batched) or lookat.shape[-1] != 3:
                gs.raise_exception(f"Lookat shape {lookat.shape} does not match (N, 3)")
            if self._is_batched and lookat.ndim == 1:
                lookat = lookat.expand((n_envs, 3))
        if up is not None:
            up = torch.as_tensor(up, dtype=gs.tc_float, device=gs.device)
            if up.ndim > (1 + self._is_batched) or up.shape[-1] != 3:
                gs.raise_exception(f"Up shape {up.shape} does not match (N, 3)")
            if self._is_batched and up.ndim == 1:
                up = up.expand((n_envs, 3))
        if transform is not None:
            if any(data is not None for data in (pos, lookat, up)):
                gs.raise_exception("Must specify either 'transform', or ('pos', 'lookat', 'up').")
            transform = torch.as_tensor(transform, dtype=gs.tc_float, device=gs.device)
            if transform.ndim > (2 + self._is_batched) or transform.shape[-2:] != (4, 4):
                gs.raise_exception(f"Transform shape {transform.shape} does not match (N, 4, 4)")
            if self._is_batched and transform.ndim == 2:
                transform = transform.expand((n_envs, 4, 4))
        if self._is_batched:
            for data in (transform, pos, lookat, up):
                if data is not None and len(data) != n_envs:
                    gs.raise_exception(f"Input data inconsistent with 'envs_idx'.")

        # Compute redundant quantities
        if transform is None:
            pos_ = pos if pos is not None else self._pos[envs_idx]
            lookat_ = lookat if lookat is not None else self._lookat[envs_idx]
            up_ = up if up is not None else self._up[envs_idx]
            transform = gu.pos_lookat_up_to_T(pos_, lookat_, up_)
        else:
            pos, lookat, up = gu.T_to_pos_lookat_up(transform)

        # Update camera transform
        if pos is not None:
            self._pos[envs_idx] = pos
        if lookat is not None:
            self._lookat[envs_idx] = lookat
        if up is not None:
            self._up[envs_idx] = up
        self._transform[envs_idx] = transform
        self._quat[envs_idx] = gu.R_to_quat(transform[..., :3, :3])

        # Refresh rendering backend to taken into account updated camera pose
        if self._raytracer is not None:
            self._raytracer.update_camera(self)
        elif self._batch_renderer is None:
            self._rasterizer.update_camera(self)

    @gs.assert_built
    def start_recording(self):
        """
        Start recording on the camera. After recording is started, all the rgb images rendered by `camera.render()`
        will be stored, and saved to a video file when `camera.stop_recording()` is called.
        """
        self._in_recording = True

    @gs.assert_built
    def pause_recording(self):
        """
        Pause recording on the camera. After recording is paused, the rgb images rendered by `camera.render()` will
        not be stored. Recording can be resumed by calling `camera.start_recording()` again.
        """
        if not self._in_recording:
            gs.raise_exception("Recording not started.")
        self._in_recording = False

    @gs.assert_built
    def stop_recording(self, save_to_filename=None, fps=60):
        """
        Stop recording on the camera. Once this is called, all the rgb images stored so far will be saved to a video
        file. If `save_to_filename` is None, the video file will be saved with the name
        '{caller_file_name}_cam_{camera.idx}.mp4'.

        If `env_separate_rigid` in `VisOptions` is set to True, each environment will record and save a video
        separately. The filenames will be identified by the indices of the environments.

        Parameters
        ----------
        save_to_filename : str, optional
            Name of the output video file. If not provided, the name will be default to the name of the caller file,
            with camera idx, a timestamp and '.mp4' extension.
        fps : int, optional
            The frames per second of the video file.
        """

        if not self._in_recording:
            gs.raise_exception("Recording not started.")

        if save_to_filename is None:
            caller_file = inspect.stack()[-1].filename
            save_to_filename = (
                os.path.splitext(os.path.basename(caller_file))[0]
                + f'_cam_{self.idx}_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
            )

        if self._is_batched:
            for env_idx in self._visualizer._context.rendered_envs_idx:
                env_imgs = [imgs[env_idx] for imgs in self._recorded_imgs]
                env_name, env_ext = os.path.splitext(save_to_filename)
                gs.tools.animate(env_imgs, f"{env_name}_{env_idx}{env_ext}", fps)
        else:
            gs.tools.animate(self._recorded_imgs, save_to_filename, fps)

        self._recorded_t_prev = -1
        self._recorded_imgs.clear()
        self._in_recording = False

    def get_pos(self, envs_idx=None):
        """The current position of the camera."""
        assert self._env_idx is None or envs_idx is None
        envs_idx = () if envs_idx is None else envs_idx
        pos = self._pos[envs_idx]
        if self._batch_renderer is None:
            pos = pos + self._envs_offset[envs_idx]
        return pos

    def get_lookat(self, envs_idx=None):
        """The current lookat point of the camera."""
        assert self._env_idx is None or envs_idx is None
        envs_idx = () if envs_idx is None else envs_idx
        lookat = self._lookat[envs_idx]
        if self._batch_renderer is None:
            lookat = lookat + self._envs_offset[envs_idx]
        return lookat

    def get_up(self, envs_idx=None):
        """The current up vector of the camera."""
        assert self._env_idx is None or envs_idx is None
        envs_idx = () if envs_idx is None else envs_idx
        return self._up[envs_idx]

    def get_quat(self, envs_idx=None):
        """The current quaternion of the camera."""
        assert self._env_idx is None or envs_idx is None
        envs_idx = () if envs_idx is None else envs_idx
        return self._quat[envs_idx]

    def get_transform(self, envs_idx=None):
        """
        The current transform matrix of the camera.
        """
        assert self._env_idx is None or envs_idx is None
        envs_idx = () if envs_idx is None else envs_idx
        transform = self._transform[envs_idx]
        if self._batch_renderer is None:
            transform = transform.clone()
            transform[..., :3, 3] += self._envs_offset[envs_idx]
        return transform

    def _repr_brief(self):
        return f"{self._repr_type()}: idx: {self._idx}, pos: {self.pos}, lookat: {self.lookat}"

    @property
    def is_built(self):
        """Whether the camera is built."""
        return self._is_built

    @property
    def idx(self):
        """The global integer index of the camera."""
        return self._idx

    @property
    def uid(self):
        """The unique ID of the camera"""
        return self._uid

    @property
    def model(self):
        """The camera model: `pinhole` or `thinlens`."""
        return self._model

    @property
    def res(self):
        """The resolution of the camera."""
        return self._res

    @property
    def fov(self):
        """The field of view of the camera."""
        return self._fov

    @property
    def aperture(self):
        """The aperture of the camera."""
        return self._aperture

    @property
    def focal_len(self):
        """The focal length for thinlens camera. Returns -1 for pinhole camera."""
        tan_half_fov = np.tan(np.deg2rad(self._fov / 2))
        if self.model == "thinlens":
            if self._res[0] > self._res[1]:
                projected_pixel_size = min(0.036 / self._res[0], 0.024 / self._res[1])
            else:
                projected_pixel_size = min(0.036 / self._res[1], 0.024 / self._res[0])
            image_dist = self._res[1] * projected_pixel_size / (2 * tan_half_fov)
            return 1.0 / (1.0 / image_dist + 1.0 / self._focus_dist)
        elif self.model == "pinhole":
            return self._res[0] / (2.0 * tan_half_fov)

    @property
    def focus_dist(self):
        """The focus distance of the camera."""
        return self._focus_dist

    @property
    def GUI(self):
        """Whether the camera will display the rendered images in a separate window."""
        return self._GUI

    @GUI.setter
    def GUI(self, value):
        self._GUI = value

    @property
    def spp(self):
        """Samples per pixel of the camera."""
        return self._spp

    @property
    def denoise(self):
        """Whether the camera will denoise the rendered image in raytracer."""
        return self._denoise

    @property
    def near(self):
        """The near plane of the camera."""
        return self._near

    @property
    def far(self):
        """The far plane of the camera."""
        return self._far

    @property
    def aspect_ratio(self):
        """The aspect ratio of the camera."""
        return self._aspect_ratio

    @property
    def env_idx(self):
        """Index of the environment bound to the camera, if any."""
        return self._env_idx

    @property
    def debug(self):
        """Whether the camera is a debug camera."""
        return self._debug

    @property
    def pos(self):
        """The current position of the camera for the tracked environment."""
        envs_idx = self._env_idx if self._is_batched else None
        return tensor_to_array(self.get_pos(envs_idx), dtype=np.float32)

    @property
    def lookat(self):
        """The current lookat point of the camera for the tracked environment."""
        envs_idx = self._env_idx if self._is_batched else None
        return tensor_to_array(self.get_lookat(envs_idx), dtype=np.float32)

    @property
    def up(self):
        """The current up vector of the camera for the tracked environment."""
        envs_idx = self._env_idx if self._is_batched else None
        return tensor_to_array(self.get_up(envs_idx), dtype=np.float32)

    @property
    def transform(self):
        """The current transform matrix of the camera for the tracked environment."""
        envs_idx = self._env_idx if self._is_batched else None
        return tensor_to_array(self.get_transform(envs_idx), dtype=np.float32)

    @cached_property
    def extrinsics(self):
        """The current extrinsics matrix of the camera."""
        res = self.transform.copy()
        res[..., :3, 1:3] *= -1
        res.flags.writeable = False
        return np.linalg.inv(res)

    @cached_property
    def intrinsics(self):
        """The current intrinsics matrix of the camera."""
        res = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]])
        res.flags.writeable = False
        return res

    @cached_property
    def projection_matrix(self):
        """Return the projection matrix for this camera."""
        a = self._aspect_ratio
        t = np.tan(np.deg2rad(0.5 * self._fov))
        n = self.near
        f = self.far
        res = np.array(
            [
                [1.0 / (a * t), 0.0, 0.0, 0.0],
                [0.0, 1.0 / t, 0.0, 0.0],
                [0.0, 0.0, (f + n) / (n - f), (2 * f * n) / (n - f)],
                [0.0, 0.0, -1.0, 0.0],
            ]
        )
        res.flags.writeable = False
        return res

    @cached_property
    def f(self):
        return 0.5 * self._res[1] / np.tan(np.deg2rad(0.5 * self._fov))

    @cached_property
    def cx(self):
        return 0.5 * self._res[0]

    @cached_property
    def cy(self):
        return 0.5 * self._res[1]
