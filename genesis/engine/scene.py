import numpy as np
import torch

import genesis as gs
from genesis.engine.entities import Emitter
from genesis.engine.simulator import Simulator
from genesis.options import (
    AvatarOptions,
    CouplerOptions,
    FEMOptions,
    MPMOptions,
    PBDOptions,
    RigidOptions,
    SFOptions,
    SimOptions,
    SPHOptions,
    ToolOptions,
    ViewerOptions,
    VisOptions,
)
from genesis.options.renderers import Rasterizer, Renderer
from genesis.repr_base import RBC
from genesis.utils.tools import FPSTracker
from genesis.vis import Visualizer


@gs.assert_initialized
class Scene(RBC):
    """
    A ``genesis.Scene`` object wraps all components in a simulation environment, including a simulator (containing multiple physics solvers), entities, and a visualizer (controlling both the viewer and all the cameras).
    Basically, everything happens inside a scene.

    Parameters
    ----------
    sim_options : gs.options.SimOptions
        The options configuring the overarching `simulator`, which in turn manages all the solvers.
    coupler_options : gs.options.CouplerOptions
        The options configuring the `coupler` between different solvers.
    tool_options : gs.options.ToolOptions
        The options configuring the tool_solver (``scene.sim.ToolSolver``).
    rigid_options : gs.options.RigidOptions
        The options configuring the rigid_solver (``scene.sim.RigidSolver``).
    avatar_options : gs.options.AvatarOptions
        The options configuring the avatar_solver (``scene.sim.AvatarSolver``).
    mpm_options : gs.options.MPMOptions
        The options configuring the mpm_solver (``scene.sim.MPMSolver``).
    sph_options : gs.options.SPHOptions
        The options configuring the sph_solver (``scene.sim.SPHSolver``).
    fem_options : gs.options.FEMOptions
        The options configuring the fem_solver (``scene.sim.FEMSolver``).
    sf_options : gs.options.SFOptions
        The options configuring the sf_solver (``scene.sim.SFSolver``).
    pbd_options : gs.options.PBDOptions
        The options configuring the pbd_solver (``scene.sim.PBDSolver``).
    vis_options : gs.options.VisOptions
        The options configuring the visualization system (``scene.visualizer``). Visualizer controls both the interactive viewer and the cameras.
    viewer_options : gs.options.ViewerOptions
        The options configuring the viewer (``scene.visualizer.viewer``).
    renderer : gs.renderers.Renderer
        The renderer used by `camera` for rendering images. This doesn't affect the behavior of the interactive viewer.
    show_viewer : bool
        Whether to show the interactive viewer. Set it to False if you only need headless rendering.
    show_FPS : bool
        Whether to show the FPS in the terminal.
    """

    def __init__(
        self,
        sim_options=SimOptions(),
        coupler_options=CouplerOptions(),
        tool_options=ToolOptions(),
        rigid_options=RigidOptions(),
        avatar_options=AvatarOptions(),
        mpm_options=MPMOptions(),
        sph_options=SPHOptions(),
        fem_options=FEMOptions(),
        sf_options=SFOptions(),
        pbd_options=PBDOptions(),
        vis_options=VisOptions(),
        viewer_options=ViewerOptions(),
        renderer=Rasterizer(),
        show_viewer=True,
        show_FPS=True,
    ):
        self._uid = gs.UID()
        self._t = 0
        self._is_built = False
        self._show_FPS = show_FPS

        # validate options
        self._validate_options(
            sim_options,
            coupler_options,
            tool_options,
            rigid_options,
            avatar_options,
            mpm_options,
            sph_options,
            fem_options,
            sf_options,
            pbd_options,
            vis_options,
            viewer_options,
            renderer,
        )

        self.sim_options = sim_options
        self.coupler_options = coupler_options
        self.tool_options = tool_options
        self.rigid_options = rigid_options
        self.avatar_options = avatar_options
        self.mpm_options = mpm_options
        self.sph_options = sph_options
        self.fem_options = fem_options
        self.sf_options = sf_options
        self.pbd_options = pbd_options

        self.vis_options = vis_options
        self.viewer_options = viewer_options

        # merge options
        self.tool_options.copy_attributes_from(self.sim_options)
        self.rigid_options.copy_attributes_from(self.sim_options)
        self.avatar_options.copy_attributes_from(self.sim_options)
        self.mpm_options.copy_attributes_from(self.sim_options)
        self.sph_options.copy_attributes_from(self.sim_options)
        self.fem_options.copy_attributes_from(self.sim_options)
        self.sf_options.copy_attributes_from(self.sim_options)
        self.pbd_options.copy_attributes_from(self.sim_options)

        # simulator
        self._sim = Simulator(
            scene=self,
            options=self.sim_options,
            coupler_options=self.coupler_options,
            tool_options=self.tool_options,
            rigid_options=self.rigid_options,
            avatar_options=self.avatar_options,
            mpm_options=self.mpm_options,
            sph_options=self.sph_options,
            fem_options=self.fem_options,
            sf_options=self.sf_options,
            pbd_options=self.pbd_options,
        )

        # visualizer
        self._visualizer = Visualizer(
            scene=self,
            show_viewer=show_viewer,
            vis_options=vis_options,
            viewer_options=viewer_options,
            renderer=renderer,
        )

        # emitters
        self._emitters = gs.List()

        self._backward_ready = False
        self._forward_ready = False

        gs.logger.info(f"Scene ~~~<{self._uid}>~~~ created.")

    def _validate_options(
        self,
        sim_options,
        coupler_options,
        tool_options,
        rigid_options,
        avatar_options,
        mpm_options,
        sph_options,
        fem_options,
        sf_options,
        pbd_options,
        vis_options,
        viewer_options,
        renderer,
    ):
        if not isinstance(sim_options, SimOptions):
            gs.raise_exception("`sim_options` should be an instance of `SimOptions`.")

        if not isinstance(coupler_options, CouplerOptions):
            gs.raise_exception("`coupler_options` should be an instance of `CouplerOptions`.")

        if not isinstance(tool_options, ToolOptions):
            gs.raise_exception("`tool_options` should be an instance of `ToolOptions`.")

        if not isinstance(rigid_options, RigidOptions):
            gs.raise_exception("`rigid_options` should be an instance of `RigidOptions`.")

        if not isinstance(avatar_options, AvatarOptions):
            gs.raise_exception("`avatar_options` should be an instance of `AvatarOptions`.")

        if not isinstance(mpm_options, MPMOptions):
            gs.raise_exception("`mpm_options` should be an instance of `MPMOptions`.")

        if not isinstance(sph_options, SPHOptions):
            gs.raise_exception("`sph_options` should be an instance of `SPHOptions`.")

        if not isinstance(fem_options, FEMOptions):
            gs.raise_exception("`fem_options` should be an instance of `FEMOptions`.")

        if not isinstance(sf_options, SFOptions):
            gs.raise_exception("`sf_options` should be an instance of `SFOptions`.")

        if not isinstance(pbd_options, PBDOptions):
            gs.raise_exception("`pbd_options` should be an instance of `PBDOptions`.")

        if not isinstance(vis_options, VisOptions):
            gs.raise_exception("`vis_options` should be an instance of `VisOptions`.")

        if not isinstance(viewer_options, ViewerOptions):
            gs.raise_exception("`viewer_options` should be an instance of `ViewerOptions`.")

        if not isinstance(renderer, Renderer):
            gs.raise_exception("`renderer` should be an instance of `gs.renderers.Renderer`.")

    @gs.assert_unbuilt
    def add_entity(
        self,
        morph,
        material=None,
        surface=None,
        visualize_contact=False,
        vis_mode=None,
    ):
        """
        Add an entity to the scene.

        Parameters
        ----------
        morph : gs.morphs.Morph
            The morph of the entity.
        material : gs.materials.Material | None, optional
            The material of the entity. If None, use ``gs.materials.Rigid()``.
        surface : gs.surfaces.Surface | None, optional
            The surface of the entity. If None, use ``gs.surfaces.Default()``.
        visualize_contact : bool
            Whether to visualize contact forces applied to this entity as arrows in the viewer and rendered images. Note that this will not be displayed in images rendered by camera using the `RayTracer` renderer.
        vis_mode : str | None, optional
            The visualization mode of the entity. This is a handy shortcut for setting `surface.vis_mode` without explicitly creating a surface object.

        Returns
        -------
        entity : genesis.Entity
            The created entity.
        """
        if material is None:
            material = gs.materials.Rigid()

        if surface is None:
            surface = (
                gs.surfaces.Default()
            )  # assign a local surface, otherwise modification will apply on global default surface

        if isinstance(material, gs.materials.Rigid):
            # small sdf res is sufficient for primitives regardless of size
            if isinstance(morph, gs.morphs.Primitive):
                material._sdf_max_res = 32

        # some morph should not smooth surface normal
        if isinstance(morph, (gs.morphs.Box, gs.morphs.Cylinder, gs.morphs.Terrain)):
            surface.smooth = False

        if isinstance(morph, (gs.morphs.URDF, gs.morphs.MJCF, gs.morphs.Terrain)):
            if not isinstance(material, (gs.materials.Rigid, gs.materials.Avatar, gs.materials.Hybrid)):
                gs.raise_exception(f"Unsupported material for morph: {material} and {morph}.")

        if surface.double_sided is None:
            if isinstance(material, gs.materials.PBD.Cloth):
                surface.double_sided = True
            else:
                surface.double_sided = False

        if vis_mode is not None:
            surface.vis_mode = vis_mode
        # validate and populate default surface.vis_mode considering morph type
        if isinstance(material, (gs.materials.Rigid, gs.materials.Avatar, gs.materials.Tool)):
            if surface.vis_mode is None:
                surface.vis_mode = "visual"

            if surface.vis_mode not in ["visual", "collision", "sdf"]:
                gs.raise_exception(
                    f"Unsupported `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Expected one of: ['visual', 'collision', 'sdf']."
                )

        elif isinstance(
            material,
            (
                gs.materials.PBD.Liquid,
                gs.materials.PBD.Particle,
                gs.materials.MPM.Liquid,
                gs.materials.MPM.Sand,
                gs.materials.MPM.Snow,
                gs.materials.SPH.Liquid,
            ),
        ):
            if surface.vis_mode is None:
                surface.vis_mode = "particle"

            if surface.vis_mode not in ["particle", "recon"]:
                gs.raise_exception(
                    f"Unsupported `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Expected one of: ['particle', 'recon']."
                )

        elif isinstance(material, (gs.materials.SF.Smoke)):
            if surface.vis_mode is None:
                surface.vis_mode = "particle"

            if surface.vis_mode not in ["particle"]:
                gs.raise_exception(
                    f"Unsupported `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Expected one of: ['particle', 'recon']."
                )

        elif isinstance(material, (gs.materials.PBD.Base, gs.materials.MPM.Base, gs.materials.SPH.Base)):
            if surface.vis_mode is None:
                surface.vis_mode = "visual"

            if surface.vis_mode not in ["visual", "particle", "recon"]:
                gs.raise_exception(
                    f"Unsupported `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Expected one of: ['visual', 'particle', 'recon']."
                )

        elif isinstance(material, (gs.materials.FEM.Base)):
            if surface.vis_mode is None:
                surface.vis_mode = "visual"

            if surface.vis_mode not in ["visual"]:
                gs.raise_exception(
                    f"Unsupported `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Expected one of: ['visual']."
                )

        elif isinstance(material, (gs.materials.Hybrid)):  # determine the visual of the outer soft part
            if surface.vis_mode is None:
                surface.vis_mode = "particle"

            if surface.vis_mode not in ["particle", "visual"]:
                gs.raise_exception(
                    f"Unsupported `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Expected one of: ['particle', 'visual']."
                )

        else:
            gs.raise_exception()

        # Rigid entities will convexify geom by default
        if hasattr(morph, "convexify") and morph.convexify is None:
            if isinstance(material, (gs.materials.Rigid, gs.materials.Avatar)):
                morph.convexify = True
            else:
                morph.convexify = False

        # Rigid entities will decompose nonconvex geom by default
        if hasattr(morph, "decompose_nonconvex") and morph.decompose_nonconvex is None:
            if isinstance(material, (gs.materials.Rigid, gs.materials.Avatar)):
                morph.decompose_nonconvex = True
            else:
                morph.decompose_nonconvex = False

        entity = self._sim._add_entity(morph, material, surface, visualize_contact)

        return entity

    @gs.assert_unbuilt
    def link_entities(
        self,
        parent_entity,
        child_entity,
        parent_link_name="",
        child_link_name="",
    ):
        """
        links two entities to act as single entity.

        Parameters
        ----------
        parent_entity : genesis.Entity
            The entity in the scene that will be a parent of kinematic tree.
        child_entity : genesis.Entity
            The entity in the scene that will be a child of kinematic tree.
        parent_link_name : str
            The name of the link in the parent entity to be linked.
        child_link_name : str
            The name of the link in the child entity to be linked.
        """
        if not isinstance(parent_entity, gs.engine.entities.RigidEntity):
            gs.raise_exception("Currently only rigid entities are supported for merging.")
        if not isinstance(child_entity, gs.engine.entities.RigidEntity):
            gs.raise_exception("Currently only rigid entities are supported for merging.")

        if not child_link_name:
            for link in child_entity._links:
                if link.parent_idx == -1:
                    child_link = link
                    break
        else:
            child_link = child_entity.get_link(child_link_name)
        parent_link = parent_entity.get_link(parent_link_name)

        if child_link._parent_idx != -1:
            gs.logger.warning(
                "Child entity already has a parent link. This may cause the entity to break into parts. Make sure this operation is intended."
            )
        child_link._parent_idx = parent_link.idx
        parent_link._child_idxs.append(child_link.idx)

    @gs.assert_unbuilt
    def add_light(
        self,
        morph,
        color=(1.0, 1.0, 1.0, 1.0),
        intensity=20.0,
        revert_dir=False,
        double_sided=False,
        beam_angle=180.0,
    ):
        """
        Add a light to the scene. Note that lights added this way can be instantiated from morphs (supporting `gs.morphs.Primitive` or `gs.morphs.Mesh`), and will only be used by the RayTracer renderer.

        Parameters
        ----------
        morph : gs.morphs.Morph
            The morph of the light. Must be an instance of `gs.morphs.Primitive` or `gs.morphs.Mesh`.
        color : tuple of float, shape (3,)
            The color of the light, specified as (r, g, b).
        intensity : float
            The intensity of the light.
        revert_dir : bool
            Whether to revert the direction of the light. If True, the light will be emitted towards the mesh's inside.
        double_sided : bool
            Whether to emit light from both sides of surface.
        beam_angle : float
            The beam angle of the light.
        """
        if self.visualizer.raytracer is None:
            gs.logger.warning("Light is only supported by RayTracer renderer.")
            return

        if not isinstance(morph, (gs.morphs.Primitive, gs.morphs.Mesh)):
            gs.raise_exception("Light morph only supports `gs.morphs.Primitive` or `gs.morphs.Mesh`.")

        mesh = gs.Mesh.from_morph_surface(morph, gs.surfaces.Plastic(smooth=False))
        self.visualizer.raytracer.add_mesh_light(
            mesh, color, intensity, morph.pos, morph.quat, revert_dir, double_sided, beam_angle
        )

    @gs.assert_unbuilt
    def add_camera(
        self,
        model="pinhole",
        res=(320, 320),
        pos=(0.5, 2.5, 3.5),
        lookat=(0.5, 0.5, 0.5),
        up=(0.0, 0.0, 1.0),
        fov=30,
        aperture=2.0,
        focus_dist=None,
        GUI=False,
        spp=256,
        denoise=True,
    ):
        """
        Add a camera to the scene. The camera model can be either 'pinhole' or 'thinlens'. The 'pinhole' model is a simple camera model that captures light rays from a single point in space. The 'thinlens' model is a more complex camera model that simulates a lens with a finite aperture size, allowing for depth of field effects. When 'pinhole' is used, the `aperture` and `focal_len` parameters are ignored.

        Parameters
        ----------
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

        Returns
        -------
        camera : genesis.Camera
            The created camera object.
        """

        return self._visualizer.add_camera(res, pos, lookat, up, model, fov, aperture, focus_dist, GUI, spp, denoise)

    @gs.assert_unbuilt
    def add_emitter(
        self,
        material,
        max_particles=20000,
        surface=None,
    ):
        """
        Add a fluid emitter to the scene.

        Parameters
        ----------
        material : gs.materials.Material
            The material of the fluid to be emitted. Must be an instance of `gs.materials.MPM.Base` or `gs.materials.SPH.Base`.
        max_particles : int
            The maximum number of particles that can be emitted by the emitter. Particles will be recycled once this limit is reached.
        surface : gs.surfaces.Surface | None, optional
            The surface of the emitter. If None, use ``gs.surfaces.Default(color=(0.6, 0.8, 1.0, 1.0))``.

        Returns
        -------
        emitter : genesis.Emitter
            The created emitter object.

        """
        if self.requires_grad:
            gs.raise_exception("Emitter is not supported in differentiable mode.")

        if not isinstance(
            material, (gs.materials.MPM.Base, gs.materials.SPH.Base, gs.materials.PBD.Particle, gs.materials.PBD.Liquid)
        ):
            gs.raise_exception(
                "Non-supported material for emitter. Supported materials are: `gs.materials.MPM.Base`, `gs.materials.SPH.Base`, `gs.materials.PBD.Particle`, `gs.materials.PBD.Liquid`."
            )

        if surface is None:
            surface = gs.surfaces.Default(color=(0.6, 0.8, 1.0, 1.0))

        emitter = Emitter(max_particles)
        entity = self.add_entity(
            morph=gs.morphs.Nowhere(n_particles=max_particles),
            material=material,
            surface=surface,
        )
        emitter.set_entity(entity)
        self._emitters.append(emitter)
        return emitter

    @gs.assert_unbuilt
    def add_force_field(self, force_field: gs.force_fields.ForceField):
        """
        Add a force field to the scene.

        Parameters
        ----------
        force_field : gs.force_fields.ForceField
            The force field to add to the scene.

        Returns
        -------
        force_field : gs.force_fields.ForceField
            The added force field.
        """
        force_field.scene = self
        self._sim._add_force_field(force_field)
        return force_field

    @gs.assert_unbuilt
    def build(
        self,
        n_envs=0,
        env_spacing=(0.0, 0.0),
        n_envs_per_row=None,
        center_envs_at_origin=True,
        compile_kernels=True,
    ):
        """
        Builds the scene once all entities have been added. This operation is required before running the simulation.

        Parameters
        ----------
        n_envs : int
            Number of parallel environments to create. If `n_envs` is 0, the scene will not have a batching dimension. If `n_envs` is greater than 0, the first dimension of all the input and returned states will be the batch dimension.
        env_spacing : tuple of float, shape (2,)
            The spacing between adjacent environments in the scene. This is for visualization purposes only and does not change simulation-related poses.
        n_envs_per_row : int
            The number of environments per row for visualization. If None, it will be set to `sqrt(n_envs)`.
        center_envs_at_origin : bool
            Whether to put the center of all the environments at the origin (for visualization only).
        compile_kernels : bool
            Whether to compile the simulation kernels inside `build()`. If False, the kernels will not be compiled (or loaded if found in the cache) until the first call of `scene.step()`. This is useful for cases you don't want to run the actual simulation, but rather just want to visualize the created scene.
        """
        with gs.logger.timer(f"Building scene ~~~<{self._uid}>~~~..."):
            self._parallelize(n_envs, env_spacing, n_envs_per_row, center_envs_at_origin)

            # simulator
            self._sim.build()

            # reset state
            self._reset()

            self._is_built = True

        if compile_kernels:
            with gs.logger.timer("Compiling simulation kernels..."):
                self._sim.step()
                self._reset()

        # visualizer
        with gs.logger.timer("Building visualizer..."):
            self._visualizer.build()

        if self._show_FPS:
            self.FPS_tracker = FPSTracker(self.n_envs)

        gs.global_scene_list.add(self)

    def _parallelize(
        self,
        n_envs,
        env_spacing,
        n_envs_per_row,
        center_envs_at_origin,
    ):
        self.n_envs = n_envs
        self.env_spacing = env_spacing
        self.n_envs_per_row = n_envs_per_row

        # true batch size
        self._B = max(1, self.n_envs)
        self._envs_idx = torch.arange(self._B, dtype=gs.tc_int, device=gs.device)

        if self.n_envs_per_row is None:
            self.n_envs_per_row = np.ceil(np.sqrt(self._B)).astype(int)

        # compute offset values for visualizing each env
        if not isinstance(env_spacing, (list, tuple)) or len(env_spacing) != 2:
            gs.raise_exception("`env_spacing` should be a tuple of length 2.")
        idx_x = np.floor(np.arange(self._B) / self.n_envs_per_row)
        idx_y = np.arange(self._B) % self.n_envs_per_row
        idx_z = np.arange(self._B)
        offset_x = idx_x * self.env_spacing[0]
        offset_y = idx_y * self.env_spacing[1]
        offset_z = idx_z * 0.0
        self.envs_offset = np.vstack([offset_x, offset_y, offset_z]).T

        # move to center
        if center_envs_at_origin:
            center = (np.max(self.envs_offset, axis=0) + np.min(self.envs_offset, axis=0)) / 2.0
            self.envs_offset -= center

        """
        Notes:
        - When using gpu
            - for non-batched env, we only parallelize certain loops that have big loop size
            - for batched env, we parallelize all loops
        - When using cpu, we serialize everything.
            - This is emprically as fast as parallel loops even with big batchsize (tested up to B=10000), because invoking multiple cpu processes cannot utilize all cpu usage.
            - In order to exploit full cpu power, users are encouraged to launch multiple processes manually, and each will use a single cpu thred.
        """
        if gs.backend == gs.cpu:
            self._para_level = gs.PARA_LEVEL.NEVER
        elif self.n_envs == 0:
            self._para_level = gs.PARA_LEVEL.PARTIAL
        else:
            self._para_level = gs.PARA_LEVEL.ALL

    @gs.assert_built
    def reset(self, state=None):
        """
        Resets the scene to its initial state.

        Parameters
        ----------
        state : dict | None
            The state to reset the scene to. If None, the scene will be reset to its initial state. If this is given, the scene's registerered initial state will be updated to this state.
        """
        gs.logger.info(f"Resetting Scene ~~~<{self._uid}>~~~.")
        self._reset(state)

    def _reset(self, state=None):
        if self._is_built:
            if state is None:
                state = self._init_state
            else:
                self._init_state = state
            self._sim.reset(state)
        else:
            self._init_state = self._get_state()

        self._t = 0
        self._forward_ready = True
        self._reset_grad()

        self._visualizer.reset()

        for emitter in self._emitters:
            emitter.reset()

    def _reset_grad(self):
        self._backward_ready = True

    def _get_state(self):
        return self._sim.get_state()

    @gs.assert_built
    def get_state(self):
        """
        Returns the current state of the scene.

        Returns
        -------
        state : genesis.SimState
            The state of the scene at the current time step.
        """
        return self._get_state()

    @gs.assert_built
    def step(self, update_visualizer=True):
        """
        Runs a simulation step forward in time.
        """
        if not self._forward_ready:
            gs.raise_exception("Forward simulation not allowed after backward pass. Please reset scene state.")

        self._sim.step()

        self._t += 1

        if update_visualizer:
            self._visualizer.update(force=False)

        if self._show_FPS:
            self.FPS_tracker.step()

    def _step_grad(self):
        self._sim.collect_output_grads()
        self._sim._step_grad()
        self._t -= 1

    @gs.assert_built
    def draw_debug_line(self, start, end, radius=0.002, color=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws a line in the scene for visualization.

        Parameters
        ----------
        start : array_like, shape (3,)
            The starting point of the line.
        end : array_like, shape (3,)
            The ending point of the line.
        radius : float, optional
            The radius of the line (represented as a cylinder)
        color : array_like, shape (4,), optional
            The color of the line in RGBA format.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_line(start, end, radius, color)

    @gs.assert_built
    def draw_debug_arrow(self, pos, vec=(0, 0, 1), radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws an arrow in the scene for visualization.

        Parameters
        ----------
        pos : array_like, shape (3,)
            The starting position of the arrow.
        vec : array_like, shape (3,), optional
            The vector of the arrow.
        radius : float, optional
            The radius of the arrow body (represented as a cylinder).
        color : array_like, shape (4,), optional
            The color of the arrow in RGBA format.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_arrow(pos, vec, radius, color)

    @gs.assert_built
    def draw_debug_frame(self, T, axis_length=1.0, origin_size=0.015, axis_radius=0.01):
        """
        Draws a 3-axis coordinate frame in the scene for visualization.

        Parameters
        ----------
        T : array_like, shape (4, 4)
            The transformation matrix of the frame.
        axis_length : float, optional
            The length of the axes.
        origin_size : float, optional
            The size of the origin point (represented as a sphere).
        axis_radius : float, optional
            The radius of the axes (represented as cylinders).

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_frame(T, axis_length, origin_size, axis_radius)

    @gs.assert_built
    def draw_debug_mesh(self, mesh, pos=np.zeros(3), T=None):
        """
        Draws a mesh in the scene for visualization.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh to be drawn.
        pos : array_like, shape (3,), optional
            The position of the mesh in the scene.
        T : array_like, shape (4, 4) | None, optional
            The transformation matrix of the mesh. If None, the mesh will be drawn at the position specified by `pos`. Otherwise, `T` has a higher priority than `pos`.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_mesh(mesh, pos, T)

    @gs.assert_built
    def draw_debug_sphere(self, pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws a sphere in the scene for visualization.

        Parameters
        ----------
        pos : array_like, shape (3,)
            The center position of the sphere.
        radius : float, optional
            radius of the sphere.
        color : array_like, shape (4,), optional
            The color of the sphere in RGBA format.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_sphere(pos, radius, color)

    @gs.assert_built
    def draw_debug_spheres(self, poss, radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws multiple spheres in the scene for visualization.

        Parameters
        ----------
        poss : array_like, shape (N, 3)
            The positions of the spheres.
        radius : float, optional
            The radius of the spheres.
        color : array_like, shape (4,), optional
            The color of the spheres in RGBA format.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_spheres(poss, radius, color)

    @gs.assert_built
    def draw_debug_box(
        self,
        bounds,
        color=(1.0, 0.0, 0.0, 1.0),
        wireframe=True,
        wireframe_radius=0.0015,
    ):
        """
        Draws a box in the scene for visualization.

        Parameters
        ----------
        bounds : array_like, shape (2, 3)
            The bounds of the box, specified as [[min_x, min_y, min_z], [max_x, max_y, max_z]].
        color : array_like, shape (4,), optional
            The color of the box in RGBA format.
        wireframe : bool, optional
            Whether to draw the box as a wireframe.
        wireframe_radius : float, optional
            The radius of the wireframe lines.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_box(
                bounds, color, wireframe=wireframe, wireframe_radius=wireframe_radius
            )

    @gs.assert_built
    def draw_debug_points(self, poss, colors=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws points in the scene for visualization.

        Parameters
        ----------
        poss : array_like, shape (N, 3)
            The positions of the points.
        colors : array_like, shape (4,), optional
            The color of the points in RGBA format.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_points(poss, colors)

    @gs.assert_built
    def clear_debug_object(self, object):
        """
        Clears all the debug objects in the scene.
        """
        with self._visualizer.viewer_lock:
            self._visualizer.context.clear_debug_object(object)

    @gs.assert_built
    def clear_debug_objects(self):
        """
        Clears all the debug objects in the scene.
        """
        with self._visualizer.viewer_lock:
            self._visualizer.context.clear_debug_objects()

    def _backward(self):
        """
        At this point, all the scene states the simulation run should have been filled with gradients.
        Next, we run backward from scene state back to scene's internal taichi variables, then back through time.
        """

        if not self._backward_ready:
            gs.raise_exception("Multiple backward calls not allowed.")

        # backward pass through time
        while self._t > 0:
            self._step_grad()

        self._backward_ready = False
        self._forward_ready = False

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """The unique ID of the scene."""
        return self._uid

    @property
    def dt(self):
        """The time duration for each simulation step."""
        return self._sim.dt

    @property
    def t(self):
        """The current simulation time step."""
        return self._t

    @property
    def substeps(self):
        """The number of substeps per simulation step."""
        return self._sim.substeps

    @property
    def requires_grad(self):
        """Whether the scene is in differentiable mode."""
        return self._sim.requires_grad

    @property
    def is_built(self):
        """Whether the scene has been built."""
        return self._is_built

    @property
    def show_FPS(self):
        """Whether to print the frames per second (FPS) in the terminal."""
        return self._show_FPS

    @property
    def gravity(self):
        """The gravity in the scene."""
        return self._sim.gravity

    @property
    def viewer(self):
        """The viewer object for the scene."""
        return self._visualizer.viewer

    @property
    def visualizer(self):
        """The visualizer object for the scene."""
        return self._visualizer

    @property
    def sim(self):
        """The scene's top-level simulator."""
        return self._sim

    @property
    def cur_t(self):
        """The current simulation time."""
        return self._sim.cur_t

    @property
    def solvers(self):
        """All the solvers managed by the scene's simulator."""
        return self._sim.solvers

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self._sim.active_solvers

    @property
    def entities(self):
        """All the entities in the scene."""
        return self._sim.entities

    @property
    def emitters(self):
        """All the emitters in the scene."""
        return self._emitters

    @property
    def tool_solver(self):
        """The scene's `tool_solver`, managing all the `ToolEntity` in the scene."""
        return self._sim.tool_solver

    @property
    def rigid_solver(self):
        """The scene's `rigid_solver`, managing all the `RigidEntity` in the scene."""
        return self._sim.rigid_solver

    @property
    def avatar_solver(self):
        """The scene's `avatar_solver`, managing all the `AvatarEntity` in the scene."""
        return self._sim.avatar_solver

    @property
    def mpm_solver(self):
        """The scene's `mpm_solver`, managing all the `MPMEntity` in the scene."""
        return self._sim.mpm_solver

    @property
    def sph_solver(self):
        """The scene's `sph_solver`, managing all the `SPHEntity` in the scene."""
        return self._sim.sph_solver

    @property
    def fem_solver(self):
        """The scene's `fem_solver`, managing all the `FEMEntity` in the scene."""
        return self._sim.fem_solver

    @property
    def pbd_solver(self):
        """The scene's `pbd_solver`, managing all the `PBDEntity` in the scene."""
        return self._sim.pbd_solver
