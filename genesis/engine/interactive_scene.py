from typing import TYPE_CHECKING, Any, Iterable

import genesis as gs
from genesis.utils.misc import with_lock
from genesis.vis.viewer_plugins.plugins import DefaultControlsPlugin

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity
    from genesis.engine.scene import Scene
    from genesis.options.sensors.options import SensorOptions


class InteractiveScene:
    """
    Composition wrapper that bundles Scene construction with the curated mutation surface needed by
    interactive workflows (GUI overlays, future web GUIs, teleop drivers, headless scripted control).
    A single ``rebuild()`` call performs construct + entity/sensor setup + ``build()`` in one pass;
    repeated calls safely destroy and recreate the scene while restoring viewer state. Re-attaching a
    plugin via ``viewer.add_plugin`` calls its ``build()`` lifecycle hook, which is where the plugin
    refreshes any cached state tied to the previous scene.

    Only available when Genesis was initialized with ``performance_mode=False`` (the default), since
    rebuild semantics rely on dynamic Quadrants arrays and would otherwise trigger systematic kernel
    recompilation.
    """

    def __init__(self):
        if not gs.use_ndarray:
            gs.raise_exception(
                "InteractiveScene is not supported in performance mode "
                "(gs.init(performance_mode=True)) since scene rebuilds would trigger systematic "
                "kernel recompilation."
            )
        self._scene: "Scene | None" = None
        self._scene_kwargs: dict[str, Any] = {}
        self._entities_kwargs: dict[str, dict[str, Any]] = {}
        self._sensors_kwargs: list["SensorOptions"] = []

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
        Construct (or reconstruct) the scene from the supplied kwargs. First call builds the initial
        scene; subsequent calls destroy the previous scene, construct a fresh one with the same setup,
        re-attach non-default viewer plugins, and restore the camera pose.

        Any argument left as ``None`` reuses what was supplied on the previous call. Pass an empty
        ``dict`` / iterable to explicitly clear stored state.

        Args:
            scene_kwargs: Keyword arguments forwarded to ``gs.Scene(...)`` (sim_options, viewer_options,
                show_viewer, etc.).
            entities_kwargs: Mapping from entity name to a kwargs dict forwarded to ``scene.add_entity``
                (morph, material, surface, visualize_contact, vis_mode). The dict key becomes the
                entity's ``name``.
            sensors_kwargs: Iterable of ``SensorOptions`` instances forwarded to ``scene.add_sensor``.
        """
        if scene_kwargs is not None:
            self._scene_kwargs = dict(scene_kwargs)
        if entities_kwargs is not None:
            self._entities_kwargs = dict(entities_kwargs)
        if sensors_kwargs is not None:
            self._sensors_kwargs = list(sensors_kwargs)

        had_previous = self._scene is not None
        cam_pos = None
        cam_lookat = None
        plugins_to_reattach: list = []

        if had_previous:
            viewer = self._scene.viewer
            cam_pos = viewer.camera_pos.copy()
            cam_lookat = viewer.camera_lookat.copy()
            # Skip default plugins; the new viewer recreates them based on its ViewerOptions.
            plugins_to_reattach = [p for p in viewer._viewer_plugins if not isinstance(p, DefaultControlsPlugin)]
            self._scene.destroy()

        new_scene = gs.Scene(**self._scene_kwargs)
        for name, kwargs in self._entities_kwargs.items():
            new_scene.add_entity(name=name, **kwargs)
        for sensor_opts in self._sensors_kwargs:
            new_scene.add_sensor(sensor_opts)
        new_scene.build()

        for plugin in plugins_to_reattach:
            new_scene.viewer.add_plugin(plugin)
        if had_previous:
            new_scene.viewer.set_camera_pose(pos=cam_pos, lookat=cam_lookat)

        self._scene = new_scene
