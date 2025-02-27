import pyglet
import genesis as gs
from genesis.repr_base import RBC

from .camera import Camera
from .rasterizer import Rasterizer


class DummyViewerLock:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class Visualizer(RBC):
    """
    This abstraction layer manages viewer and renderers.
    """

    def __init__(self, scene, show_viewer, vis_options, viewer_options, renderer):
        self._t = -1
        self._scene = scene

        # Rasterizer context is shared by viewer and rasterizer
        try:
            from .viewer import Viewer
            from .rasterizer_context import RasterizerContext

        except Exception as e:
            raise ImportError("Rendering not working on this machine.") from e
        self._context = RasterizerContext(vis_options)

        # try to connect to display
        try:
            display = pyglet.display.get_display()
            screen = display.get_default_screen()
            self._connected_to_display = True
        except Exception:
            self._connected_to_display = False

        if show_viewer:
            if not self._connected_to_display:
                gs.raise_exception("No display detected. Use `show_viewer=False` for headless mode.")

            if viewer_options.res is None:
                viewer_size_ratio = screen.get_scale() * 0.5
                viewer_options.res = (
                    int(screen.height * viewer_size_ratio / 0.75),
                    int(screen.height * viewer_size_ratio),
                )

            self._viewer = Viewer(viewer_options, self._context)

        else:
            self._viewer = None

        # Rasterizer is always needed for depth and segmentation mask rendering.
        self._rasterizer = Rasterizer(self._viewer, self._context)

        if isinstance(renderer, gs.renderers.RayTracer):
            from .raytracer import Raytracer

            self._renderer = self._raytracer = Raytracer(renderer, vis_options)

        else:
            self._renderer = self._rasterizer
            self._raytracer = None

        self._cameras = gs.List()

    def add_camera(self, res, pos, lookat, up, model, fov, aperture, focus_dist, GUI, spp, denoise):
        camera = Camera(
            self, len(self._cameras), model, res, pos, lookat, up, fov, aperture, focus_dist, GUI, spp, denoise
        )
        self._cameras.append(camera)
        return camera

    def reset(self):
        self._t = -1

        self._context.reset()

        # temp fix for cam.render() segfault
        if self._viewer is not None:
            # need to update viewer once here, because otherwise camera will update scene if render is called right after build, which will lead to segfault. TODO: this slows down visualizer.update(). Needs to remove this once the bug is fixed.
            try:
                self._viewer.update()
            except:
                pass

        if self._raytracer is not None:
            self._raytracer.reset()

    def build(self):
        self._context.build(self._scene)

        if self._viewer is not None:
            self._viewer.build(self._scene)
            self.viewer_lock = self._viewer.lock
        else:
            self.viewer_lock = DummyViewerLock()

        self._rasterizer.build()
        if self._raytracer is not None:
            self._raytracer.build(self._scene)

        for camera in self._cameras:
            camera._build()

        if self._cameras:
            # need to update viewer once here, because otherwise camera will update scene if render is called right
            # after build, which will lead to segfault.
            if self._viewer is not None:
                self._viewer.update()
            else:
                # viewer creation will compile rendering kernels if viewer is not created, render here once to compile
                self._rasterizer.render_camera(self._cameras[0])

    def update(self, force=True):
        if force:  # force update
            self.reset()

        if self._viewer is not None:
            if self._viewer.is_alive():
                self._viewer.update()
            else:
                gs.raise_exception("Viewer closed.")

    def update_visual_states(self):
        """
        Update all visualization-only variables here.
        """
        if self._t < self._scene._t:
            self._t = self._scene._t

            for camera in self._cameras:
                if camera._attached_link is not None:
                    camera.move_to_attach()

            if self._scene.rigid_solver.is_active():
                self._scene.rigid_solver.update_geoms_render_T()
                self._scene.rigid_solver._kernel_update_vgeoms()

                # drone propellers
                for entity in self._scene.rigid_solver.entities:
                    if isinstance(entity, gs.engine.entities.DroneEntity):
                        entity.update_propeller_vgeoms()

                self._scene.rigid_solver.update_vgeoms_render_T()

            if self._scene.avatar_solver.is_active():
                self._scene.avatar_solver.update_geoms_render_T()
                self._scene.avatar_solver._kernel_update_vgeoms()
                self._scene.avatar_solver.update_vgeoms_render_T()

            if self._scene.mpm_solver.is_active():
                self._scene.mpm_solver.update_render_fields()

            if self._scene.sph_solver.is_active():
                self._scene.sph_solver.update_render_fields()

            if self._scene.pbd_solver.is_active():
                self._scene.pbd_solver.update_render_fields()

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def viewer(self):
        return self._viewer

    @property
    def rasterizer(self):
        return self._rasterizer

    @property
    def context(self):
        return self._context

    @property
    def raytracer(self):
        return self._raytracer

    @property
    def renderer(self):
        return self._renderer

    @property
    def scene(self):
        return self._scene

    @property
    def connected_to_display(self):
        return self._connected_to_display

    @property
    def cameras(self):
        return self._cameras
