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
