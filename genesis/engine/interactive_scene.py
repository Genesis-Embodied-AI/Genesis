from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Iterable

import genesis as gs
from genesis.utils.misc import with_lock
from genesis.vis.viewer_plugins.plugins import DefaultControlsPlugin

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity
    from genesis.engine.scene import Scene
    from genesis.options.sensors.options import SensorOptions


class InteractiveFeature(Enum):
    """Editing capabilities an InteractiveScene may expose to interactive frontends (e.g. the ImGui
    overlay). A frontend queries InteractiveScene.supported_features and enables the matching controls,
    without needing to know which simulator modes each feature requires."""

    SCALE_ENTITY = auto()
    ADD_ENTITY = auto()
    REMOVE_ENTITY = auto()
    REBUILD = auto()


class InteractiveScene:
    """
    Composition wrapper that gives a Scene the curated mutation surface needed by interactive workflows
    (GUI overlays, future web GUIs, teleop drivers, headless scripted control). ``rebuild()`` reconstructs
    the wrapped Scene in place - destroying and re-creating its solvers, viewer and entities on the same
    object - so external references to the Scene and its viewer stay valid across edits.

    The ImGui overlay creates and owns one of these internally, so the typical user never instantiates it:
    a plain ``gs.Scene(viewer_options=ViewerOptions(enable_gui=True))`` is the whole setup. It is also
    usable standalone, for power users building their own overlay: construct it (optionally wrapping an
    existing Scene) and drive ``rebuild()`` directly.

    Scene editing (rebuild and the add/remove/scale entity operations that go through it) relies on dynamic
    Quadrants arrays. In performance mode (gs.init(performance_mode=True)) those arrays are static and a
    rebuild would trigger systematic kernel recompilation, so the editing features report as unsupported
    via supported_features.
    """

    def __init__(self, scene: "Scene | None" = None):
        self._scene: "Scene | None" = scene
        self._scene_kwargs: dict[str, Any] = {}
        self._build_kwargs: dict[str, Any] = {}
        self._entities_kwargs: dict[str, dict[str, Any]] = {}
        self._sensors_kwargs: list["SensorOptions"] = []
        if scene is not None:
            # Capture the wrapped scene's construction so rebuild() can reconstruct it identically. The
            # stored option objects are already merged with sim_options; re-passing them is idempotent.
            self._scene_kwargs = dict(
                sim_options=scene.sim_options,
                coupler_options=scene.coupler_options,
                tool_options=scene.tool_options,
                rigid_options=scene.rigid_options,
                kinematic_options=scene.kinematic_options,
                mpm_options=scene.mpm_options,
                sph_options=scene.sph_options,
                fem_options=scene.fem_options,
                sf_options=scene.sf_options,
                pbd_options=scene.pbd_options,
                vis_options=scene.vis_options,
                viewer_options=scene.viewer_options,
                profiling_options=scene.profiling_options,
                renderer=scene.renderer_options,
                show_viewer=scene.viewer is not None,
            )
            self._build_kwargs = dict(
                n_envs=scene.n_envs,
                env_spacing=scene.env_spacing,
                n_envs_per_row=scene.n_envs_per_row,
            )

    @property
    def supported_features(self) -> frozenset[InteractiveFeature]:
        """Set of editing features available for the current simulator mode. Empty in performance mode
        since every editing operation reconstructs the scene through rebuild()."""
        if not gs.use_ndarray:
            return frozenset()
        return frozenset(InteractiveFeature)

    @property
    def scene(self) -> "Scene":
        if self._scene is None:
            gs.raise_exception("InteractiveScene has no scene yet; call `rebuild()` first.")
        return self._scene

    @property
    def viewer(self):
        return self.scene.viewer

    @property
    def entities(self):
        return self.scene.entities

    @property
    def rigid_solver(self):
        return self.scene.rigid_solver

    @property
    def n_envs(self) -> int:
        return self.scene.n_envs

    @property
    def t(self) -> int:
        return self.scene.t

    @property
    def dt(self) -> float:
        return self.scene.sim.dt

    @property
    def is_built(self) -> bool:
        return self._scene is not None and self._scene.is_built

    @property
    def _lock(self):
        return self.scene.viewer.render_lock

    @property
    def _ctx(self):
        return self.scene.viewer.context

    @with_lock
    def refresh_visual_transforms(self):
        """Refresh render transforms so visuals reflect the latest qpos. Idempotent."""
        self._refresh_visual_transforms_unlocked()

    def _refresh_visual_transforms_unlocked(self):
        rigid_solver = self.scene.rigid_solver
        if not rigid_solver.is_active:
            return
        rigid_solver.update_geoms_render_T()
        rigid_solver.update_vgeoms()
        rigid_solver.update_vgeoms_render_T()
        ctx = self._ctx
        ctx.update_link_frame()
        ctx.update_rigid()

    @with_lock
    def reset(self):
        """Reset the scene and refresh visuals. Clears contact arrows and other transient render nodes."""
        self.scene.reset()
        self._ctx.clear_dynamic_nodes(only_outdated=False)
        self._refresh_visual_transforms_unlocked()

    @with_lock
    def set_entity_qpos(self, entity: "RigidEntity", qpos, env_idx: int | None = None):
        """Set the entity's qpos and refresh visuals."""
        entity.set_qpos(qpos, envs_idx=env_idx)
        self._refresh_visual_transforms_unlocked()

    @with_lock
    def set_entity_dofs_position(self, entity: "RigidEntity", dofs_position, env_idx: int | None = None):
        """Set the entity's DOF positions and refresh visuals."""
        entity.set_dofs_position(dofs_position, envs_idx=env_idx)
        self._refresh_visual_transforms_unlocked()

    @with_lock
    def set_entity_vis_mode(self, entity: "RigidEntity", mode: str):
        """Switch entity rendering between ``"visual"`` and ``"collision"``."""
        from genesis.ext import pyrender

        if not isinstance(entity.surface, gs.surfaces.Surface):
            return
        old_mode = entity.surface.vis_mode
        if old_mode == mode:
            return

        ctx = self._ctx
        rigid_solver = self.scene.rigid_solver

        old_geoms = entity.vgeoms if old_mode == "visual" else entity.geoms
        for geom in old_geoms:
            if geom.uid in ctx.rigid_nodes:
                ctx.remove_node(ctx.rigid_nodes[geom.uid])
                del ctx.rigid_nodes[geom.uid]

        entity.surface.vis_mode = mode
        rigid_solver.update_geoms_render_T()
        rigid_solver.update_vgeoms()
        rigid_solver.update_vgeoms_render_T()

        if mode == "visual":
            geoms = entity.vgeoms
            geoms_T = rigid_solver._vgeoms_render_T
        else:
            geoms = entity.geoms
            geoms_T = rigid_solver._geoms_render_T

        is_collision = mode == "collision"
        for geom in geoms:
            geom_envs_idx = ctx._get_geom_active_envs_idx(geom, ctx.rendered_envs_idx)
            if len(geom_envs_idx) == 0:
                continue
            mesh = geom.get_trimesh()
            geom_T = geoms_T[geom.idx][geom_envs_idx]
            ctx.add_rigid_node(
                geom,
                pyrender.Mesh.from_trimesh(
                    mesh=mesh,
                    poses=geom_T,
                    smooth=geom.surface.smooth if not is_collision else False,
                    double_sided=geom.surface.double_sided if not is_collision else False,
                    is_floor=isinstance(entity._morph, gs.morphs.Plane),
                    env_shared=not ctx.env_separate_rigid,
                ),
            )

    def rebuild(
        self,
        *,
        scene_kwargs: dict[str, Any] | None = None,
        entities_kwargs: dict[str, dict[str, Any]] | None = None,
        sensors_kwargs: Iterable["SensorOptions"] | None = None,
    ):
        """
        Reconstruct the scene from the stored setup. When wrapping an existing scene (or after the first
        build), the scene is rebuilt in place - the same Scene object is torn down and re-initialized - so
        external references (and the user's ``scene`` / ``scene.viewer`` handles) stay valid. Non-default
        viewer plugins are re-attached and the camera pose restored.

        Any argument left as ``None`` reuses what was supplied previously. Pass an empty ``dict`` /
        iterable to explicitly clear stored state.

        Args:
            scene_kwargs: Keyword arguments forwarded to ``gs.Scene(...)`` (sim_options, viewer_options,
                show_viewer, etc.).
            entities_kwargs: Mapping from entity name to a kwargs dict forwarded to ``scene.add_entity``
                (morph, material, surface, visualize_contact, vis_mode). The dict key becomes the
                entity's ``name``.
            sensors_kwargs: Iterable of ``SensorOptions`` instances forwarded to ``scene.add_sensor``.
        """
        if InteractiveFeature.REBUILD not in self.supported_features:
            gs.raise_exception(
                "InteractiveScene.rebuild() is not supported in performance mode "
                "(gs.init(performance_mode=True)) since it would trigger systematic kernel recompilation."
            )
        if scene_kwargs is not None:
            self._scene_kwargs = dict(scene_kwargs)
        if entities_kwargs is not None:
            self._entities_kwargs = dict(entities_kwargs)
        if sensors_kwargs is not None:
            self._sensors_kwargs = list(sensors_kwargs)

        scene = self._scene
        cam_pos = None
        cam_lookat = None
        plugins_to_reattach: list = []
        pyrender_window = None

        if scene is not None:
            viewer = scene.viewer
            if viewer is not None:
                cam_pos = viewer.camera_pos.copy()
                cam_lookat = viewer.camera_lookat.copy()
                # Skip default plugins; the rebuilt viewer recreates them based on its ViewerOptions.
                plugins_to_reattach = [p for p in viewer._viewer_plugins if not isinstance(p, DefaultControlsPlugin)]
                # Preserve the live window/GL context so the rebuild does not close and reopen it. Detaching it
                # keeps scene.destroy() from closing the window; Viewer.build() re-points it at the rebuilt scene.
                pyrender_window = viewer._pyrender_viewer
                viewer._pyrender_viewer = None
            scene.destroy()
            # Re-initialize the SAME object in place so external references survive the rebuild.
            scene.__init__(**self._scene_kwargs)
        else:
            scene = gs.Scene(**self._scene_kwargs)

        for name, kwargs in self._entities_kwargs.items():
            scene.add_entity(name=name, **kwargs)
        for sensor_opts in self._sensors_kwargs:
            scene.add_sensor(sensor_opts)
        # Hand the preserved window to the new viewer so build() reuses it instead of opening a new one.
        if pyrender_window is not None and scene.viewer is not None:
            scene.viewer._pyrender_viewer = pyrender_window
        scene.build(**self._build_kwargs)

        new_viewer = scene.viewer
        if new_viewer is not None:
            # A scene built with enable_gui=True auto-attaches its own ImGui overlay. When re-attaching the previous
            # overlay (which carries user state - panel width, custom panels, pending edits), drop the fresh
            # auto-attached one of the same type so the viewer does not end up with two overlays. The plugin must be
            # cleared from both the wrapper staging list (_viewer_plugins) and pyrender's live list (plugins), since
            # build() already copied it into the live render loop.
            reattach_types = {type(p) for p in plugins_to_reattach}
            pyrender_viewer = new_viewer._pyrender_viewer
            for plugin in [p for p in new_viewer._viewer_plugins if type(p) in reattach_types]:
                new_viewer._viewer_plugins.remove(plugin)
                if pyrender_viewer is not None and plugin in pyrender_viewer.plugins:
                    pyrender_viewer.plugins.remove(plugin)
                    pyrender_viewer.remove_handlers(plugin)

            for plugin in plugins_to_reattach:
                new_viewer.add_plugin(plugin)
            if cam_pos is not None:
                new_viewer.set_camera_pose(pos=cam_pos, lookat=cam_lookat)

        self._scene = scene
