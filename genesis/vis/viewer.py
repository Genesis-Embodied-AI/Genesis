import numpy as np

import genesis as gs
import genesis.utils.geom as gu

try:
    from genesis.ext import pyrender
except:
    print("Failed to import pyrender. Rendering will not work.")
from genesis.repr_base import RBC
from genesis.utils.tools import Rate


class ViewerLock:
    def __init__(self, pyrender_viewer):
        self._pyrender_viewer = pyrender_viewer

    def __enter__(self):
        self._pyrender_viewer.render_lock.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self._pyrender_viewer.render_lock.release()


class DummyViewerLock:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class Viewer(RBC):
    def __init__(self, options, context):
        self._res = options.res
        self._refresh_rate = options.refresh_rate
        self._max_FPS = options.max_FPS
        self._camera_init_pos = options.camera_pos
        self._camera_init_lookat = options.camera_lookat
        self._camera_up = options.camera_up
        self._camera_fov = options.camera_fov

        self.context = context

        self._followed_entity = None
        self._follow_fixed_axis = None
        self._follow_smoothing = None
        self._follow_fix_orientation = None
        self._follow_lookat = None

        if self._max_FPS is not None:
            self.rate = Rate(self._max_FPS)

    def build(self, scene):
        self.scene = scene

        # set viewer camera
        self.setup_camera()

        # viewer
        if gs.platform == "Linux":
            run_in_thread = True
            auto_start = True
        elif gs.platform == "macOS":
            run_in_thread = False
            auto_start = False
            gs.logger.warning(
                "Non-linux system detected. In order to use the interactive viewer, you need to manually run simulation in a separate thread and then start viewer. See `examples/render_on_macos.py`."
            )
        elif gs.platform == "Windows":
            run_in_thread = True
            auto_start = True
            gs.logger.warning("Windows system detected. Viewer may have some issues.")

        self._pyrender_viewer = pyrender.Viewer(
            context=self.context,
            viewport_size=self._res,
            run_in_thread=run_in_thread,
            auto_start=auto_start,
            view_center=self._camera_init_lookat,
            shadow=self.context.shadow,
            plane_reflection=self.context.plane_reflection,
            viewer_flags={
                "window_title": f"Genesis {gs.__version__}",
                "refresh_rate": self._refresh_rate,
            },
        )
        if auto_start:
            self._pyrender_viewer.wait_until_initialized()

        self.lock = ViewerLock(self._pyrender_viewer)

        gs.logger.info(f"Viewer created. Resolution: ~<{self._res[0]}×{self._res[1]}>~, max_FPS: ~<{self._max_FPS}>~.")

    def start(self):
        # used for starting viewer thread in non-linux OS
        self._pyrender_viewer.start()

    def stop(self):
        # used for closing viewer thread in non-linux OS
        self._pyrender_viewer.close_external()

    def is_alive(self):
        return self._pyrender_viewer.is_active

    def setup_camera(self):
        pos = np.array(self._camera_init_pos)
        up = np.array(self._camera_up)
        lookat = np.array(self._camera_init_lookat)

        yfov = self._camera_fov / 180.0 * np.pi
        z = pos - lookat
        R = gu.z_up_to_R(z, up=up)
        pose = gu.trans_R_to_T(pos, R)
        self._camera_node = self.context.add_node(pyrender.PerspectiveCamera(yfov=yfov), pose=pose)

    def update(self):
        if self._followed_entity is not None:
            self.update_following()

        with self.lock:
            buffer_updates = self.context.update()
            for buffer_id, buffer_data in buffer_updates.items():
                self._pyrender_viewer.pending_buffer_updates[buffer_id] = buffer_data

        # lock FPS
        if self._max_FPS is not None:
            self.rate.sleep()

    def set_camera_pose(self, pose=None, pos=None, lookat=None):
        """
        Set viewer camera pose.

        Parameters
        ----------
        pose : [4,4] float, optional
            Camera-to-world pose. If provided, `pos` and `lookat` will be ignored.
        pos : (3,) float, optional
            Camera position.
        lookat : (3,) float, optional
            Camera lookat point.
        """
        if pose is None:
            if pos is None:
                pos = np.array(self._camera_init_pos)
            else:
                pos = np.array(pos)

            if lookat is None:
                lookat = np.array(self._camera_init_lookat)
            else:
                lookat = np.array(lookat)

            up = np.array(self._camera_up)

            z = pos - lookat
            R = gu.z_up_to_R(z, up=up)
            pose = gu.trans_R_to_T(pos, R)
        else:
            if np.array(pose).shape != (4, 4):
                gs.raise_exception("pose should be a 4x4 matrix.")

        self._pyrender_viewer._trackball.set_camera_pose(pose)

    def follow_entity(self, entity, fixed_axis=(None, None, None), smoothing=None, fix_orientation=False):
        """
        Set the viewer to follow a specified entity.
        Parameters
        ----------
        entity : genesis.Entity
            The entity to follow.
        fixed_axis : (float, float, float), optional
            The fixed axis for the viewer's movement. For each axis, if None, the viewer will move freely. If a float, the viewer will be fixed on at that value.
            For example, [None, None, None] will allow the viewer to move freely while following, [None, None, 0.5] will fix the viewer's z-axis at 0.5.
        smoothing : float, optional
            The smoothing factor in ]0,1[ for the viewer's movement. If None, no smoothing will be applied.
        fix_orientation : bool, optional
            If True, the viewer will maintain its orientation relative to the world. If False, the viewer will look at the base link of the entity.
        """
        self._followed_entity = entity
        self._follow_fixed_axis = fixed_axis
        self._follow_smoothing = smoothing
        self._follow_fix_orientation = fix_orientation
        self._follow_lookat = self._camera_init_lookat

    def update_following(self):
        """
        Update the viewer position to follow the specified entity.
        """
        entity_pos = self._followed_entity.get_pos().cpu().numpy()
        if entity_pos.ndim > 1:  # check for multiple envs
            entity_pos = entity_pos[0]
        camera_pose = np.array(self._pyrender_viewer._trackball.pose)
        camera_pos = np.array(self._pyrender_viewer._trackball.pose[:3, 3])

        if self._follow_smoothing is not None:
            # Smooth viewer movement with a low-pass filter
            camera_pos = self._follow_smoothing * camera_pos + (1 - self._follow_smoothing) * (
                entity_pos + np.array(self._camera_init_pos)
            )
            self._follow_lookat = (
                self._follow_smoothing * self._follow_lookat + (1 - self._follow_smoothing) * entity_pos
            )
        else:
            camera_pos = entity_pos + np.array(self._camera_init_pos)
            self._follow_lookat = entity_pos

        for i, fixed_axis in enumerate(self._follow_fixed_axis):
            # Fix the camera's position along the specified axis
            if fixed_axis is not None:
                camera_pos[i] = fixed_axis

        if self._follow_fix_orientation:
            # Keep the camera orientation fixed by overriding the lookat point
            camera_pose[:3, 3] = camera_pos
            self.set_camera_pose(pose=camera_pose)
        else:
            self.set_camera_pose(pos=camera_pos, lookat=self._follow_lookat)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def res(self):
        return self._res

    @property
    def refresh_rate(self):
        return self._refresh_rate

    @property
    def max_FPS(self):
        return self._max_FPS

    @property
    def camera_pos(self):
        """
        Get the camera's current position.
        """
        return np.array(self._pyrender_viewer._trackball._n_pose[:3, 3])

    @property
    def camera_lookat(self):
        """
        Get the camera's current lookat point.
        """
        pos = np.array(self._pyrender_viewer._trackball._n_pose[:3, 3])
        z = self._pyrender_viewer._trackball._n_pose[:3, 2]
        return pos - z

    @property
    def camera_pose(self):
        """
        Get the camera's current pose represented by a 4x4 matrix.
        """
        return np.array(self._pyrender_viewer._trackball._n_pose)

    @property
    def camera_up(self):
        return self._camera_up

    @property
    def camera_fov(self):
        return self._camera_fov
