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

    def __init__(self, scene, show_viewer, vis_options, viewer_options, renderer_options):
        self._t = -1
        self._scene = scene

        self._context = None
        self._viewer = None
        self._rasterizer = None
        self._raytracer = None
        self._batch_renderer = None
        self.viewer_lock = None  # check if null to know if the Visualizer has been built

        # Rasterizer context is shared by viewer and rasterizer
        try:
            from .rasterizer_context import RasterizerContext
            from .viewer import Viewer

        except Exception as e:
            gs.raise_exception_from("Rendering not working on this machine.", e)
        self._context = RasterizerContext(vis_options)

        try:
            screen_height, _screen_width, screen_scale = gs.utils.try_get_display_size()
            self._has_display = True
        except Exception as e:
            if show_viewer:
                gs.raise_exception_from("No display detected. Use `show_viewer=False` for headless mode.", e)
            self._has_display = False

        if show_viewer:
            if gs.global_scene_list:
                raise gs.raise_exception(
                    "Interactive viewer not supported when managing multiple scenes. Please set `show_viewer=False` "
                    "or call `del scene`."
                )

            if viewer_options.res is None:
                viewer_height = (screen_height * screen_scale) * VIEWER_DEFAULT_HEIGHT_RATIO
                viewer_width = viewer_height / VIEWER_DEFAULT_ASPECT_RATIO
                viewer_options.res = (int(viewer_width), int(viewer_height))
            if viewer_options.run_in_thread is None:
                if gs.platform == "Linux":
                    viewer_options.run_in_thread = True
                elif gs.platform == "macOS":
                    viewer_options.run_in_thread = False
                elif gs.platform == "Windows":
                    viewer_options.run_in_thread = True
            if gs.platform == "macOS" and viewer_options.run_in_thread:
                gs.raise_exception("Running viewer in background thread is not supported on MacOS.")

            self._viewer = Viewer(viewer_options, self._context)
            if not viewer_options.run_in_thread:
                gs.logger.warning(
                    "Interactive viewer running in main thread. It will only be responsive if a simulation is running."
                )

        # Rasterizer is always needed for depth and segmentation mask rendering.
        self._rasterizer = Rasterizer(self._viewer, self._context)

        if isinstance(renderer_options, gs.renderers.BatchRenderer):
            from .batch_renderer import BatchRenderer

            self._renderer = self._batch_renderer = BatchRenderer(self, renderer_options, vis_options)
        elif isinstance(renderer_options, gs.renderers.RayTracer):
            from .raytracer import Raytracer

            self._renderer = self._raytracer = Raytracer(renderer_options, vis_options)
        elif isinstance(renderer_options, gs.renderers.Rasterizer):
            self._renderer = self._rasterizer

        self._cameras = gs.List()

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self._rasterizer is not None:
            self._rasterizer.destroy()
            self._rasterizer = None
        if self._viewer is not None:
            self._viewer.stop()
            self._viewer = None
        if self._batch_renderer is not None:
            self._batch_renderer.destroy()
            self._batch_renderer = None
        if self._raytracer is not None:
            self._raytracer.destroy()
            self._raytracer = None
        if self._context is not None:
            self._context.destroy()
            del self._context
            self._context = None
        self.viewer_lock = None
        self._renderer = None

    def add_camera(
        self, res, pos, lookat, up, model, fov, aperture, focus_dist, GUI, spp, denoise, near, far, env_idx, debug
    ):
        cam_idx = len([camera for camera in self._cameras if camera.debug == debug])
        camera = Camera(
            self,
            cam_idx,
            model,
            res,
            pos,
            lookat,
            up,
            fov,
            aperture,
            focus_dist,
            GUI,
            spp,
            denoise,
            near,
            far,
            env_idx=env_idx,
            debug=debug,
        )
        self._cameras.append(camera)
        return camera

    def add_mesh_light(self, mesh, color, intensity, pos, quat, revert_dir, double_sided, cutoff):
        if self._raytracer is not None:
            self._raytracer.add_mesh_light(mesh, color, intensity, pos, quat, revert_dir, double_sided, cutoff)
        else:
            gs.raise_exception("`add_mesh_light` is specific to raytracer renderer.")

    def add_light(self, pos, dir, color, intensity, directional, castshadow, cutoff, attenuation):
        if self._batch_renderer is not None:
            self._batch_renderer.add_light(pos, dir, color, intensity, directional, castshadow, cutoff, attenuation)
        else:
            gs.raise_exception("`add_light` is specific to batch renderer.")

    def reset(self):
        self._t = -1

        self._context.reset()

        if self._raytracer is not None:
            self._raytracer.reset()

        if self._batch_renderer is not None:
            self._batch_renderer.reset()

        if self.viewer_lock is not None:
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
            camera.build()

        # Batch renderer needs to be built after cameras are built
        if self._batch_renderer is not None:
            self._batch_renderer.build()

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

    def update_visual_states(self, force_render: bool = False):
        """
        Update all visualization-only variables here.
        """
        # Early return if already updated previously
        if not force_render and self._t >= self.scene._t:
            return

        for camera in self._cameras:
            if camera.is_built:
                if camera._attached_link is not None:
                    camera.move_to_attach()
                elif camera._followed_entity is not None:
                    camera.update_following()

        if self._scene.rigid_solver.is_active():
            self._scene.rigid_solver.update_geoms_render_T()
            self._scene.rigid_solver.update_vgeoms()

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

        self._t = self._scene._t

    def colorize_seg_idxc_arr(self, seg_idxc_arr):
        if self._batch_renderer is not None:
            return self._batch_renderer.colorize_seg_idxc_arr(seg_idxc_arr)
        else:
            return self._context.colorize_seg_idxc_arr(seg_idxc_arr)

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
    def batch_renderer(self):
        return self._batch_renderer

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
    def has_display(self):
        return self._has_display

    @property
    def cameras(self):
        return self._cameras

    @property
    def segmentation_idx_dict(self):
        if self._batch_renderer is not None:
            return self._batch_renderer.seg_idxc_map
        else:
            return self._context.seg_idxc_map
