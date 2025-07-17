import pyglet

import genesis as gs
from genesis.repr_base import RBC

from .camera import Camera
from .rasterizer import Rasterizer


VIEWER_DEFAULT_HEIGHT_RATIO = 0.5
VIEWER_DEFAULT_ASPECT_RATIO = 0.75


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

        self._context = None
        self._viewer = None
        self._rasterizer = None
        self._raytracer = None
        self.viewer_lock = None  # check if null to know if the Visualizer has been built

        # Rasterizer context is shared by viewer and rasterizer
        try:
            from .viewer import Viewer
            from .rasterizer_context import RasterizerContext

        except Exception as e:
            gs.raise_exception_from("Rendering not working on this machine.", e)
        self._context = RasterizerContext(vis_options)

        # try to connect to display
        try:
            if pyglet.version < "2.0":
                display = pyglet.canvas.Display()
                screen = display.get_default_screen()
                scale = 1.0
            else:
                display = pyglet.display.get_display()
                screen = display.get_default_screen()
                scale = screen.get_scale()
            self._connected_to_display = True
        except Exception as e:
            if show_viewer:
                gs.raise_exception_from("No display detected. Use `show_viewer=False` for headless mode.", e)
            self._connected_to_display = False

        if show_viewer:
            if gs.global_scene_list:
                raise gs.raise_exception(
                    "Interactive viewer not supported when managing multiple scenes. Please set `show_viewer=False` "
                    "or call `scene.destroy`."
                )

            if viewer_options.res is None:
                viewer_height = (screen.height * scale) * VIEWER_DEFAULT_HEIGHT_RATIO
                viewer_width = viewer_height / VIEWER_DEFAULT_ASPECT_RATIO
                viewer_options.res = (int(viewer_width), int(viewer_height))
            if viewer_options.run_in_thread is None:
                if gs.platform == "Linux":
                    viewer_options.run_in_thread = True
                elif gs.platform == "macOS":
                    viewer_options.run_in_thread = False
                    gs.logger.warning(
                        "Mac OS detected. The interactive viewer will only be responsive if a simulation is running."
                    )
                elif gs.platform == "Windows":
                    viewer_options.run_in_thread = True
            if gs.platform == "macOS" and viewer_options.run_in_thread:
                gs.raise_exception("Running viewer in background thread is not supported on MacOS.")

            self._viewer = Viewer(viewer_options, self._context)

        # Rasterizer is always needed for depth and segmentation mask rendering.
        self._rasterizer = Rasterizer(self._viewer, self._context)

        if isinstance(renderer, gs.renderers.RayTracer):
            from .raytracer import Raytracer

            self._renderer = self._raytracer = Raytracer(renderer, vis_options)

        else:
            self._renderer = self._rasterizer
            self._raytracer = None

        self._cameras = gs.List()

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self._viewer is not None:
            self._viewer.stop()
            self._viewer = None
        if self._rasterizer is not None:
            self._rasterizer.destroy()
            self._rasterizer = None
        if self._raytracer is not None:
            self._raytracer.destroy()
            self._raytracer = None
        if self._context is not None:
            self._context.destroy()
            del self._context
            self._context = None
        self._renderer = None

    def add_camera(self, res, pos, lookat, up, model, fov, aperture, focus_dist, GUI, spp, denoise):
        camera = Camera(
            self, len(self._cameras), model, res, pos, lookat, up, fov, aperture, focus_dist, GUI, spp, denoise
        )
        self._cameras.append(camera)
        return camera

    def reset(self):
        self._t = -1

        self._context.reset()

        if self._raytracer is not None:
            self._raytracer.reset()

        if self.viewer_lock is not None:
            for camera in self._cameras:
                self._rasterizer.render_camera(camera)

            if self._viewer is not None:
                self._viewer.update(auto_refresh=True)

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

        # Make sure that the viewer is fully compiled and in a clean state
        self.reset()

    def update(self, force=True, auto=None):
        if force:  # force update
            self.reset()
        elif self._viewer is not None:
            if self._viewer.is_alive():
                self._viewer.update(auto_refresh=auto)
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
                self._scene.rigid_solver._kernel_update_vgeoms(
                    vgeoms_info=self._scene.rigid_solver.vgeoms_info,
                    vgeoms_state=self._scene.rigid_solver.vgeoms_state,
                    links_state=self._scene.rigid_solver.links_state,
                    static_rigid_sim_config=self._scene.rigid_solver._static_rigid_sim_config,
                )

                # drone propellers
                for entity in self._scene.rigid_solver.entities:
                    if isinstance(entity, gs.engine.entities.DroneEntity):
                        entity.update_propeller_vgeoms()

                self._scene.rigid_solver.update_vgeoms_render_T()

            if self._scene.avatar_solver.is_active():
                self._scene.avatar_solver.update_geoms_render_T()
                self._scene.avatar_solver._kernel_update_vgeoms(
                    vgeoms_info=self._scene.avatar_solver.vgeoms_info,
                    vgeoms_state=self._scene.avatar_solver.vgeoms_state,
                    links_state=self._scene.avatar_solver.links_state,
                    static_rigid_sim_config=self._scene.avatar_solver._static_rigid_sim_config,
                )
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
