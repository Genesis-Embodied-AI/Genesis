import inspect
import os
import time

import cv2
import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.repr_base import RBC


class Camera(RBC):
    """
    Genesis camera class. The camera can be used to render RGB, depth, and segmentation images. The camera can use either rasterizer or raytracer for rendering, specified by `scene.renderer`.
    The camera also comes with handy tools such as video recording.

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
        Samples per pixel. Defaults to 256.
    denoise : bool
        Whether to denoise the camera's rendered image.
    near : float
        The near plane of the camera.
    far : float
        The far plane of the camera.
    transform : np.ndarray, shape (4, 4), optional
        The transform matrix of the camera.
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
        self._pos = pos
        self._lookat = lookat
        self._up = up
        self._transform = transform
        self._aspect_ratio = self._res[0] / self._res[1]
        self._visualizer = visualizer
        self._is_built = False
        self._attached_link = None
        self._attached_offset_T = None

        self._in_recording = False
        self._recorded_imgs = []

        if self._model not in ["pinhole", "thinlens"]:
            gs.raise_exception(f"Invalid camera model: {self._model}")

        if self._focus_dist is None:
            self._focus_dist = np.linalg.norm(np.array(lookat) - np.array(pos))

    def _build(self):
        self._rasterizer = self._visualizer.rasterizer
        self._raytracer = self._visualizer.raytracer

        if self._rasterizer is not None:
            self._rasterizer.add_camera(self)
        if self._raytracer is not None:
            self._raytracer.add_camera(self)

        self._is_built = True
        self.set_pose(self._transform, self._pos, self._lookat, self._up)

    def attach(self, rigid_link, offset_T):
        self._attached_link = rigid_link
        self._attached_offset_T = offset_T

    def detach(self):
        self._attached_link = None
        self._attached_offset_T = None

    @gs.assert_built
    def move_to_attach(self):
        if self._attached_link is None:
            gs.raise_exception(f"The camera hasn't been mounted!")
        if self._visualizer._scene.n_envs > 0:
            gs.raise_exception(f"Mounted camera not supported in parallel simulation!")

        link_pos = self._attached_link.get_pos().cpu().numpy()
        link_quat = self._attached_link.get_quat().cpu().numpy()
        link_T = gu.trans_quat_to_T(link_pos, link_quat)
        transform = link_T @ self._attached_offset_T
        self.set_pose(transform=transform)

    @gs.assert_built
    def render(self, rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False):
        """
        Render the camera view. Note that the segmentation mask can be colorized, and if not colorized, it will store an object index in each pixel based on the segmentation level specified in `vis_options.segmentation_level`. For example, if `segmentation_level='link'`, the segmentation mask will store `link_idx`, which can then be used to retrieve the actual link objects using `scene.rigid_solver.links[link_idx]`.

        Parameters
        ----------
        rgb : bool, optional
            Whether to render an RGB image.
        depth : bool, optional
            Whether to render a depth image.
        segmentation : bool, optional
            Whether to render the segmentation mask.
        colorize_seg : bool, optional
            If True, the segmentation mask will be colorized.
        normal : bool, optional
            Whether to render the surface normal.

        Returns
        -------
        rgb_arr : np.ndarray
            The rendered RGB image.
        depth_arr : np.ndarray
            The rendered depth image.
        seg_arr : np.ndarray
            The rendered segmentation mask.
        normal_arr : np.ndarray
            The rendered surface normal.
        """

        if (rgb or depth or segmentation or normal) is False:
            gs.raise_exception("Nothing to render.")

        rgb_arr, depth_arr, seg_idxc_arr, seg_arr, normal_arr = None, None, None, None, None

        if self._raytracer is not None:
            if rgb:
                self._raytracer.update_scene()
                rgb_arr = self._raytracer.render_camera(self)

            if depth or segmentation or normal:
                if self._rasterizer is not None:
                    self._rasterizer.update_scene()
                    _, depth_arr, seg_idxc_arr, normal_arr = self._rasterizer.render_camera(
                        self, False, depth, segmentation, normal=normal
                    )
                else:
                    gs.raise_exception("Cannot render depth or segmentation image.")

        elif self._rasterizer is not None:
            self._rasterizer.update_scene()
            rgb_arr, depth_arr, seg_idxc_arr, normal_arr = self._rasterizer.render_camera(
                self, rgb, depth, segmentation, normal=normal
            )

        else:
            gs.raise_exception("No renderer was found.")

        if seg_idxc_arr is not None:
            seg_idx_arr = self._rasterizer._context.seg_idxc_to_idx(seg_idxc_arr)
            if colorize_seg or (self._GUI and self._visualizer.connected_to_display):
                seg_color_arr = self._rasterizer._context.colorize_seg_idxc_arr(seg_idxc_arr)
            if colorize_seg:
                seg_arr = seg_color_arr
            else:
                seg_arr = seg_idx_arr

        # succeed rendering, and display image
        if self._GUI and self._visualizer.connected_to_display:
            if rgb:
                cv2.imshow(f"Genesis - Camera {self._idx} [RGB]", rgb_arr[..., [2, 1, 0]])

            if depth:
                depth_min = depth_arr.min()
                depth_max = depth_arr.max()
                depth_normalized = (depth_arr - depth_min) / (depth_max - depth_min)
                # closer objects appear brighter
                depth_normalized = 1 - depth_normalized
                depth_img = (depth_normalized * 255).astype(np.uint8)
                cv2.imshow(f"Genesis - Camera {self._idx} [Depth]", depth_img)

            if segmentation:
                cv2.imshow(f"Genesis - Camera {self._idx} [Segmentation]", seg_color_arr[..., [2, 1, 0]])

            if normal:
                cv2.imshow(f"Genesis - Camera {self._idx} [Normal]", normal_arr[..., [2, 1, 0]])

            cv2.waitKey(1)

        if self._in_recording and rgb_arr is not None:
            self._recorded_imgs.append(rgb_arr)

        return rgb_arr, depth_arr, seg_arr, normal_arr

    @gs.assert_built
    def set_pose(self, transform=None, pos=None, lookat=None, up=None):
        """
        Set the pose of the camera.
        Note that `transform` has a higher priority than `pos`, `lookat`, and `up`. If `transform` is provided, the camera pose will be set based on the transform matrix. Otherwise, the camera pose will be set based on `pos`, `lookat`, and `up`.

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

        """
        if transform is not None:
            assert transform.shape == (4, 4)
            self._transform = transform
            self._pos, self._lookat, self._up = gu.T_to_pos_lookat_up(transform)

        else:
            if pos is not None:
                self._pos = pos

            if lookat is not None:
                self._lookat = lookat

            if up is not None:
                self._up = up

            self._transform = gu.pos_lookat_up_to_T(self._pos, self._lookat, self._up)

        if self._rasterizer is not None:
            self._rasterizer.update_camera(self)
        if self._raytracer is not None:
            self._raytracer.update_camera(self)

    @gs.assert_built
    def set_params(self, fov=None, aperture=None, focus_dist=None):
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

        if self._rasterizer is not None:
            self._rasterizer.update_camera(self)
        if self._raytracer is not None:
            self._raytracer.update_camera(self)

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

        gs.tools.animate(self._recorded_imgs, save_to_filename, fps)
        self._recorded_imgs.clear()
        self._in_recording = False

    def _repr_brief(self):
        return f"{self._repr_type()}: idx: {self._idx}, pos: {self._pos}, lookat: {self._lookat}"

    @property
    def is_built(self):
        """Whether the camera is built."""
        return self._is_built

    @property
    def idx(self):
        """The integer index of the camera."""
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
    def pos(self):
        """The current position of the camera."""
        return np.array(self._pos)

    @property
    def lookat(self):
        """The current lookat point of the camera."""
        return np.array(self._lookat)

    @property
    def up(self):
        """The current up vector of the camera."""
        return np.array(self._up)

    @property
    def transform(self):
        """The current transform matrix of the camera."""
        return self._transform

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
