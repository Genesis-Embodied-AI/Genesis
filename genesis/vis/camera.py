import inspect
import math
import os
import time

import cv2
import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.repr_base import RBC
from genesis.utils.misc import tensor_to_array


# quat for Madrona needs to be transformed to y-forward
def _T_to_quat_for_madrona(T):
    if not isinstance(T, torch.Tensor):
        gs.raise_exception(f"the input must be torch.Tensor. got: {type(T)=}")

    R = T[..., :3, :3].contiguous()
    quat = gu.R_to_quat(R)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return torch.stack([x + w, x - w, y - z, y + z], dim=1) / math.sqrt(2.0)


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
        The near plane of the camera.
    far : float
        The far plane of the camera.
    transform : np.ndarray, shape (4, 4), optional
        The transform matrix of the camera.
    env_idx : int, optional
        The index of the environment to track the camera.
    debug : bool, optional
        Whether to use the debug camera. It enables to create cameras that can used to monitor / debug the
        simulation without being part of the "sensors". Their output is rendered by the usual simple Rasterizer
        systematically, no matter if BatchRender and RayTracer is enabled. This way, it is possible to record the
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
        self._aspect_ratio = self._res[0] / self._res[1]
        self._visualizer = visualizer
        self._is_built = False
        self._attached_link = None
        self._attached_offset_T = None
        self._debug = debug

        self._env_idx = env_idx
        self._envs_offset = None

        self._in_recording = False
        self._recorded_imgs = []

        self._followed_entity = None
        self._follow_fixed_axis = None
        self._follow_smoothing = None
        self._follow_fix_orientation = None

        if self._model not in ["pinhole", "thinlens"]:
            gs.raise_exception(f"Invalid camera model: {self._model}")

        if self._focus_dist is None:
            self._focus_dist = np.linalg.norm(np.asarray(lookat) - np.asarray(pos))

    def build(self):
        n_envs = max(self._visualizer.scene.n_envs, 1)
        self._multi_env_pos_tensor = torch.empty((n_envs, 3), dtype=gs.tc_float, device=gs.device)
        self._multi_env_lookat_tensor = torch.empty((n_envs, 3), dtype=gs.tc_float, device=gs.device)
        self._multi_env_up_tensor = torch.empty((n_envs, 3), dtype=gs.tc_float, device=gs.device)
        self._multi_env_transform_tensor = torch.empty((n_envs, 4, 4), dtype=gs.tc_float, device=gs.device)
        self._multi_env_quat_tensor = torch.empty((n_envs, 4), dtype=gs.tc_float, device=gs.device)

        self._envs_offset = torch.as_tensor(self._visualizer._scene.envs_offset, dtype=gs.tc_float, device=gs.device)

        self._rasterizer = self._visualizer.rasterizer
        self._raytracer = self._visualizer.raytracer if not self._debug else None
        self._batch_renderer = self._visualizer.batch_renderer if not self._debug else None

        if self._batch_renderer is not None:
            self._rgb_stacked = True
            self._other_stacked = True
        else:
            if self._env_idx is None:
                self._env_idx = 0
            elif not isinstance(self._env_idx, int) or self._env_idx >= max(self._visualizer.scene.n_envs, 1):
                gs.raise_exception("Tracked environment index out-of-bounds")
            if self._raytracer is not None:
                self._raytracer.add_camera(self)
                self._rgb_stacked = False
                self._other_stacked = False
            else:
                self._rasterizer.add_camera(self)
                self._rgb_stacked = self._visualizer._context.env_separate_rigid
                self._other_stacked = self._visualizer._context.env_separate_rigid

        self._is_built = True
        self.set_pose(
            transform=self._initial_transform, pos=self._initial_pos, lookat=self._initial_lookat, up=self._initial_up
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
        # move_to_attach can be called from update_visual_states(), which could be called either before or after build(),
        # but set_pose() is only allowed after build(), so we need to check if the camera is built here, and early out if not.
        if not self._is_built:
            return

        if self._attached_link is None:
            gs.raise_exception("Camera not attached")

        link_pos = self._attached_link.get_pos(self._env_idx)
        link_quat = self._attached_link.get_quat(self._env_idx)
        link_T = gu.trans_quat_to_T(link_pos, link_quat)
        transform = torch.matmul(link_T, self._attached_offset_T)
        self.set_pose(transform=transform, env_idx=self._env_idx)

    def follow_entity(self, entity, fixed_axis=(None, None, None), smoothing=None, fix_orientation=False):
        """
        Set the camera to follow a specified entity.

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
        if self._attached_link is not None:
            gs.raise_exception("Impossible to following an entity with a camera that is already attached.")

        self._followed_entity = entity
        self._follow_fixed_axis = fixed_axis
        self._follow_smoothing = smoothing
        self._follow_fix_orientation = fix_orientation

    def unfollow_entity(self):
        """
        Stop following any entity with the camera.

        Calling this method has no effect if the camera is not currently following any entity.
        """
        self._followed_entity = None
        self._follow_fixed_axis = None
        self._follow_smoothing = None
        self._follow_fix_orientation = None

    @gs.assert_built
    def update_following(self):
        """
        Update the camera position to follow the specified entity.
        """
        if self._followed_entity is None:
            gs.raise_exception("No entity to follow. Please call `camera.follow_entity(entity)` first.")

        # Keep the camera orientation fixed by overriding the lookat point if requested
        if self._follow_fix_orientation:
            camera_transform = self._multi_env_transform_tensor.clone()
            camera_lookat = None
            camera_pos = camera_transform[:, :3, 3]
        else:
            camera_lookat = self._multi_env_lookat_tensor.clone()
            camera_pos = self._multi_env_pos_tensor.clone()

        # Smooth camera movement with a low-pass filter, in particular Exponential Moving Average (EMA) if requested
        entity_pos = self._followed_entity.get_pos()
        camera_pos -= self._initial_pos
        if self._follow_smoothing is not None:
            camera_pos[:] = self._follow_smoothing * camera_pos + (1.0 - self._follow_smoothing) * entity_pos
            if not self._follow_fix_orientation:
                camera_lookat[:] = self._follow_smoothing * camera_lookat + (1.0 - self._follow_smoothing) * entity_pos
        else:
            camera_pos[:] = entity_pos
        camera_pos += self._initial_pos

        # Fix the camera's position along the specified axis if requested
        for i, fixed_axis in enumerate(self._follow_fixed_axis):
            if fixed_axis is not None:
                camera_pos[:, i] = fixed_axis

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
        rgb_arr, depth_arr, seg_arr, normal_arr = self._batch_renderer.render(
            rgb, depth, segmentation, normal, force_render, antialiasing
        )
        # The first dimension of the array is camera.
        # If n_envs > 0, the second dimension of the output is env.
        # If n_envs == 0, the second dimension of the output is camera.
        # Only return the current camera's image
        if rgb_arr:
            rgb_arr = rgb_arr[self.idx]
        if depth:
            depth_arr = depth_arr[self.idx]
        if segmentation:
            seg_arr = seg_arr[self.idx]
        if normal:
            normal_arr = normal_arr[self.idx]
        return rgb_arr, depth_arr, seg_arr, normal_arr

    @gs.assert_built
    def render(
        self,
        rgb=True,
        depth=False,
        segmentation=False,
        colorize_seg=False,
        normal=False,
        force_render=False,
        antialiasing=True,
    ):
        """
        Render the camera view.

        Note
        ----
        The segmentation mask can be colorized, and if not colorized, it will store an object index in each pixel based
        on the segmentation level specified in `VisOptions.segmentation_level`.
        For example, if `segmentation_level='link'`, the segmentation mask will store `link_idx`, which can then be
        used to retrieve the actual link objects using `scene.rigid_solver.links[link_idx]`. If `env_separate_rigid`
        in `VisOptions` is set to True, each component will return a stack of images, with the number of images equal
        to `len(rendered_envs_idx)`.

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
        force_render : bool, optional
            Whether to force rendering even if the scene has not changed.
        antialiasing : bool, optional
            Whether to apply anti-aliasing. Only supported by 'BatchRenderer' for now.

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
        rgb_arr, depth_arr, seg_arr, seg_color_arr, seg_idxc_arr, normal_arr = None, None, None, None, None, None

        if self._batch_renderer is not None:
            rgb_arr, depth_arr, seg_idxc_arr, normal_arr = self._batch_render(
                rgb, depth, segmentation, normal, force_render, antialiasing
            )
        elif self._raytracer is not None:
            if rgb:
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
                self, rgb, depth, segmentation, normal=normal
            )

        if seg_idxc_arr is not None:
            if colorize_seg or (self._GUI and self._visualizer.has_display):
                seg_color_arr = self._rasterizer._context.colorize_seg_idxc_arr(seg_idxc_arr)
            seg_arr = seg_color_arr if colorize_seg else seg_idxc_arr

        if self._in_recording or self._GUI and self._visualizer.has_display:
            rgb_np = rgb_arr if rgb_arr is None else tensor_to_array(rgb_arr)

        # succeed rendering, and display image
        if self._GUI and self._visualizer.has_display:
            depth_np, seg_color_np, normal_np = map(
                lambda e: e if e is None else tensor_to_array(e), (depth_arr, seg_color_arr, normal_arr)
            )

            title = f"Genesis - Camera {self._idx}"
            if rgb:
                # FIXME: Check whether it always render RGB or RGBA ?
                rgb_img = np.flip(rgb_np, axis=-1)
                rgb_env = ""
                if self._rgb_stacked:
                    rgb_img = rgb_img[0]
                    rgb_env = " Environment 0"
                cv2.imshow(f"{title + rgb_env} [RGB]", rgb_img)

            other_env = " Environment 0" if self._other_stacked else ""
            if depth:
                depth_min = depth_np.min()
                depth_max = depth_np.max()
                if depth_max - depth_min > gs.EPS:
                    depth_normalized = (depth_max - depth_np) / (depth_max - depth_min)
                    depth_img = (depth_normalized * 255).astype(np.uint8)
                else:
                    depth_img = np.zeros_like(depth_arr, dtype=np.uint8)
                if self._other_stacked:
                    depth_img = depth_img[0]
                cv2.imshow(f"{title + other_env} [Depth]", depth_img)

            if segmentation:
                seg_img = np.flip(seg_color_np, axis=-1)
                if self._other_stacked:
                    seg_img = seg_img[0]
                cv2.imshow(f"{title + other_env} [Segmentation]", seg_img)

            if normal:
                normal_img = np.flip(normal_np, axis=-1)
                if self._other_stacked:
                    normal_img = normal_img[0]
                cv2.imshow(f"{title + other_env} [Normal]", normal_img)

            cv2.waitKey(1)

        if self._in_recording and rgb_np is not None:
            self._recorded_imgs.append(rgb_np)

        return rgb_arr, depth_arr, seg_arr, normal_arr

    @gs.assert_built
    def get_segmentation_idx_dict(self):
        """
        Returns a dictionary mapping segmentation indices to scene entities.

        In the segmentation map:
        - Index 0 corresponds to the background (-1).
        - Indices > 0 correspond to scene elements, which may be represented as:
            - `entity_id`
            - `(entity_id, link_id)`
            - `(entity_id, link_id, geom_id)`
          depending on the material type and the configured segmentation level.
        """
        return self._rasterizer._context.seg_idxc_map

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
            Numpy array of shape (res[0], res[1], 3) representing the point cloud in each pixel.
        mask_arr : np.ndarray
            The valid depth mask.
        """
        # Compute the (denormalized) depth map using PyRender systematically.
        # TODO: Add support of BatchRendered (requires access to projection matrix)
        self._rasterizer.update_scene()
        rgb_arr, depth_arr, seg_idxc_arr, normal_arr = self._rasterizer.render_camera(
            self, rgb=False, depth=True, segmentation=False, normal=False
        )

        # Convert OpenGL projection matrix to camera intrinsics
        P = self._rasterizer._camera_nodes[self.uid].camera.get_projection_matrix()
        height, width = depth_arr.shape
        fx = P[0, 0] * width / 2.0
        fy = P[1, 1] * height / 2.0
        cx = (1.0 - P[0, 2]) * width / 2.0
        cy = (1.0 + P[1, 2]) * height / 2.0

        # Extract camera pose if needed
        if world_frame:
            T_OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
            cam_pose = self._rasterizer._camera_nodes[self.uid].matrix @ T_OPENGL_TO_OPENCV

        # Mask out invalid depth
        mask = np.where((self.near < depth_arr) & (depth_arr < self.far * (1.0 - 1e-3)))

        # Compute normalized pixel coordinates
        v, u = np.meshgrid(np.arange(height, dtype=np.int32), np.arange(width, dtype=np.int32), indexing="ij")
        u = u.reshape((-1,))
        v = v.reshape((-1,))

        # Convert to world coordinates
        depth_grid = depth_arr[v, u]
        world_x = depth_grid * (u + 0.5 - cx) / fx
        world_y = depth_grid * (v + 0.5 - cy) / fy
        world_z = depth_grid

        point_cloud = np.stack((world_x, world_y, world_z, np.ones((depth_arr.size,), dtype=np.float32)), axis=-1)
        if world_frame:
            point_cloud = point_cloud @ cam_pose.T
        point_cloud = point_cloud[:, :3].reshape((*depth_arr.shape, 3))
        return point_cloud, mask

    def set_pose(self, transform=None, pos=None, lookat=None, up=None, env_idx=None):
        """
        Set the pose of the camera.
        Note that `transform` has a higher priority than `pos`, `lookat`, and `up`.
        If `transform` is provided, the camera pose will be set based on the transform matrix.
        Otherwise, the camera pose will be set based on `pos`, `lookat`, and `up`.

        Parameters
        ----------
        transform : np.ndarray, shape (4, 4), optional
            The transform matrix of the camera.
        pos : array-like, shape (3,), optional
            The position of the camera.
        lookat : array-like, shape (3,), optional
            The lookat point of the camera.
        up : array-like, shape (3,), optional
            The up vector of the camera.
        env_idx : array of indices in integers, optional
            The environment indices. If not provided, the camera pose will be set for all environments.
        """
        n_envs = max(self._visualizer.scene.n_envs, 1)
        env_idx = self._visualizer._scene._sanitize_envs_idx(env_idx)

        if pos is None and lookat is None and up is None and transform is None:
            return

        if pos is not None:
            pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)
            if pos.shape[-1] != 3:
                gs.raise_exception(f"Pos shape {pos.shape} does not match (n_envs, 3)")
            if pos.ndim == 1:
                pos = pos.expand((n_envs, 3))
        if lookat is not None:
            lookat = torch.as_tensor(lookat, dtype=gs.tc_float, device=gs.device)
            if lookat.shape[-1] != 3:
                gs.raise_exception(f"Lookat shape {lookat.shape} does not match (n_envs, 3)")
            if lookat.ndim == 1:
                lookat = lookat.expand((n_envs, 3))
        if up is not None:
            up = torch.as_tensor(up, dtype=gs.tc_float, device=gs.device)
            if up.shape[-1] != 3:
                gs.raise_exception(f"Up shape {up.shape} does not match (n_envs, 3)")
            if up.ndim == 1:
                up = up.expand((n_envs, 3))
        if transform is not None:
            if any(data is not None for data in (pos, lookat, up)):
                gs.raise_exception("Must specify either 'transform', or ('pos', 'lookat', 'up').")
            transform = torch.as_tensor(transform, dtype=gs.tc_float, device=gs.device)
            if transform.shape[-2:] != (4, 4):
                gs.raise_exception(f"Transform shape {transform.shape} does not match (4, 4)")
            if transform.ndim == 2:
                transform = transform.expand((n_envs, 4, 4))

        for data in (transform, pos, lookat, up):
            if data is not None and len(data) != len(env_idx):
                gs.raise_exception(f"Input data inconsistent with env_idx.")

        if transform is None:
            pos_ = pos if pos is not None else self._multi_env_pos_tensor
            lookat_ = lookat if lookat is not None else self._multi_env_lookat_tensor
            up_ = up if up is not None else self._multi_env_up_tensor
            transform = gu.pos_lookat_up_to_T(pos_, lookat_, up_)
        else:
            pos, lookat, up = gu.T_to_pos_lookat_up(transform)

        if pos is not None:
            self._multi_env_pos_tensor[env_idx] = pos
        if lookat is not None:
            self._multi_env_lookat_tensor[env_idx] = lookat
        if up is not None:
            self._multi_env_up_tensor[env_idx] = up
        self._multi_env_transform_tensor[env_idx] = transform
        self._multi_env_quat_tensor[env_idx] = _T_to_quat_for_madrona(transform)

        if self._raytracer is not None:
            self._raytracer.update_camera(self)
        elif self._batch_renderer is None:
            self._rasterizer.update_camera(self)

    @gs.assert_built
    def set_params(self, fov=None, aperture=None, focus_dist=None, intrinsics=None):
        """
        Update the camera parameters.

        Parameters
        ----------
        fov: float, optional
            The vertical field of view of the camera.
        aperture : float, optional
            The aperture of the camera. Only supports 'thinlens' camera model.
        focus_dist : float, optional
            The focus distance of the camera. Only supports 'thinlens' camera model.
        intrinsics : np.ndarray, shape (3, 3), optional
            The intrinsics matrix of the camera. If provided, it should be consistent with the specified 'fov'.
        """
        if self.model != "thinlens" and (aperture is not None or focus_dist is not None):
            gs.logger.warning("Only `thinlens` camera model supports parameter update.")

        if aperture is not None:
            if self.model != "thinlens":
                gs.logger.warning("Only `thinlens` camera model supports `aperture`.")
            self._aperture = aperture
        if focus_dist is not None:
            if self.model != "thinlens":
                gs.logger.warning("Only `thinlens` camera model supports `focus_dist`.")
            self._focus_dist = focus_dist

        if fov is not None:
            self._fov = fov

        if intrinsics is not None:
            intrinsics_fov = 2 * np.rad2deg(np.arctan(0.5 * self._res[1] / intrinsics[0, 0]))
            if fov is not None:
                if abs(intrinsics_fov - fov) > 1e-4:
                    gs.raise_exception("The camera's intrinsic values and fov do not match.")
            else:
                self._fov = intrinsics_fov

        if self._raytracer is not None:
            self._raytracer.update_camera(self)
        elif self._batch_renderer is None:
            self._rasterizer.update_camera(self)

    @gs.assert_built
    def start_recording(self):
        """
        Start recording on the camera. After recording is started, all the rgb images rendered by `camera.render()` will be stored, and saved to a video file when `camera.stop_recording()` is called.
        """
        self._in_recording = True

    @gs.assert_built
    def pause_recording(self):
        """
        Pause recording on the camera. After recording is paused, the rgb images rendered by `camera.render()` will not be stored. Recording can be resumed by calling `camera.start_recording()` again.
        """
        if not self._in_recording:
            gs.raise_exception("Recording not started.")
        self._in_recording = False

    @gs.assert_built
    def stop_recording(self, save_to_filename=None, fps=60):
        """
        Stop recording on the camera. Once this is called, all the rgb images stored so far will be saved to a video file. If `save_to_filename` is None, the video file will be saved with the name '{caller_file_name}_cam_{camera.idx}.mp4'.
        If `env_separate_rigid` in `VisOptions` is set to True, each environment will record and save a video separately. The filenames will be identified by the indices of the environments.

        Parameters
        ----------
        save_to_filename : str, optional
            Name of the output video file. If not provided, the name will be default to the name of the caller file, with camera idx, a timestamp and '.mp4' extension.
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

        if self._rgb_stacked:
            for env_idx in self._visualizer._context.rendered_envs_idx:
                env_imgs = [imgs[env_idx] for imgs in self._recorded_imgs]
                env_name, env_ext = os.path.splitext(save_to_filename)
                gs.tools.animate(env_imgs, f"{env_name}_{env_idx}{env_ext}", fps)
        else:
            gs.tools.animate(self._recorded_imgs, save_to_filename, fps)

        self._recorded_imgs.clear()
        self._in_recording = False

    def get_pos(self, env_idx=None):
        """The current position of the camera."""
        env_idx = () if env_idx is None else env_idx
        pos = self._multi_env_pos_tensor[env_idx]
        if self._batch_renderer is None:
            pos = pos + self._envs_offset[env_idx]
        return pos

    def get_lookat(self, env_idx=None):
        """The current lookat point of the camera."""
        env_idx = () if env_idx is None else env_idx
        lookat = self._multi_env_lookat_tensor[env_idx]
        if self._batch_renderer is None:
            lookat = lookat + self._envs_offset[env_idx]
        return lookat

    def get_up(self, env_idx=None):
        """The current up vector of the camera."""
        env_idx = () if env_idx is None else env_idx
        return self._multi_env_up_tensor[env_idx]

    def get_quat(self, env_idx=None):
        """The current quaternion of the camera."""
        env_idx = () if env_idx is None else env_idx
        return self._multi_env_quat_tensor[env_idx]

    def get_transform(self, env_idx=None):
        """
        The current transform matrix of the camera.
        """
        env_idx = () if env_idx is None else env_idx
        transform = self._multi_env_transform_tensor[env_idx]
        if self._batch_renderer is None:
            transform = transform.clone()
            transform[..., :3, 3] += self._envs_offset[env_idx]
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
        """Index of the environment being tracked by the camera."""
        return self._env_idx

    @property
    def debug(self):
        """Whether the camera is a debug camera."""
        return self._debug

    @property
    def pos(self):
        """The current position of the camera for the tracked environment."""
        return tensor_to_array(self.get_pos(self._env_idx), dtype=np.float32)

    @property
    def lookat(self):
        """The current lookat point of the camera for the tracked environment."""
        return tensor_to_array(self.get_lookat(self._env_idx), dtype=np.float32)

    @property
    def up(self):
        """The current up vector of the camera for the tracked environment."""
        return tensor_to_array(self.get_up(self._env_idx), dtype=np.float32)

    @property
    def transform(self):
        """The current transform matrix of the camera for the tracked environment."""
        return tensor_to_array(self.get_transform(self._env_idx), dtype=np.float32)

    @property
    def extrinsics(self):
        """The current extrinsics matrix of the camera."""
        extrinsics = np.array(self.transform)
        extrinsics[:3, 1] *= -1
        extrinsics[:3, 2] *= -1
        return np.linalg.inv(extrinsics)

    @property
    def intrinsics(self):
        """The current intrinsics matrix of the camera."""
        # compute intrinsics using fov and resolution
        f = 0.5 * self._res[1] / np.tan(np.deg2rad(0.5 * self._fov))
        cx = 0.5 * self._res[0]
        cy = 0.5 * self._res[1]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
