from typing import TYPE_CHECKING

import numpy as np
import gstaichi as ti

import genesis as gs
from genesis.options.solvers import IPCCouplerOptions
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

# Check if libuipc is available
try:
    import uipc

    UIPC_AVAILABLE = True
except ImportError:
    UIPC_AVAILABLE = False
    uipc = None


@ti.data_oriented
class IPCCoupler(RBC):
    """
    Coupler class for handling Incremental Potential Contact (IPC) simulation coupling.

    This coupler manages the communication between Genesis solvers and the IPC system,
    including rigid bodies (as ABD objects) and FEM bodies in a unified contact framework.
    """

    def __init__(self, simulator: "Simulator", options: "IPCCouplerOptions") -> None:
        """
        Initialize IPC Coupler.

        Parameters
        ----------
        simulator : Simulator
            The simulator containing all solvers
        options : IPCCouplerOptions
            IPC configuration options
        """
        # Check if uipc is available
        if not UIPC_AVAILABLE:
            raise ImportError(
                "libuipc is required for IPC coupling but not found.\n"
                "Please build and install libuipc from source:\n"
                "https://github.com/spiriMirror/libuipc"
            )

        self.sim = simulator
        self.options = options

        # Validate coupling strategy
        valid_strategies = ["two_way_soft_constraint", "contact_proxy"]
        if self.options.coupling_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid coupling_strategy '{self.options.coupling_strategy}'. " f"Must be one of {valid_strategies}"
            )

        # Store solver references
        self.rigid_solver = self.sim.rigid_solver
        self.fem_solver = self.sim.fem_solver

        # IPC system components (will be initialized in build)
        self._ipc_engine = None
        self._ipc_world = None
        self._ipc_scene = None
        self._ipc_abd = None
        self._ipc_stk = None
        self._ipc_abd_contact = None
        self._ipc_fem_contact = None
        self._ipc_scene_subscenes = {}
        self._use_subscenes = False  # Will be set in _init_ipc based on number of environments

        # IPC link filter: maps entity_idx -> set of link_idx to include in IPC
        # If entity_idx not in dict or value is None, all links of that entity participate
        self._ipc_link_filters = {}

        # IPC-only links: maps entity_idx -> set of link_idx that should ONLY exist in IPC
        # These links will not have soft constraints, use full density, and directly set Genesis transforms
        self._ipc_only_links = {}

        # Storage for Genesis rigid body states before IPC advance
        # Maps link_idx -> {env_idx: transform_matrix}
        self._genesis_stored_states = {}

        # Storage for IPC contact forces on rigid links (both coupling mode)
        # Maps link_idx -> {env_idx: {'force': np.array, 'torque': np.array}}
        self._ipc_contact_forces = {}

    def build(self) -> None:
        """Build IPC system"""
        # Initialize IPC system
        self._init_ipc()
        self._add_objects_to_ipc()
        self._finalize_ipc()
        if self.options.enable_ipc_gui:
            self._init_ipc_gui()

    def _init_ipc(self):
        """Initialize IPC system components"""
        from uipc.core import Engine, World, Scene
        from uipc.constitution import AffineBodyConstitution, StableNeoHookean, NeoHookeanShell, DiscreteShellBending

        # Disable IPC logging if requested
        if self.options.disable_ipc_logging:
            from uipc import Logger, Timer

            Logger.set_level(Logger.Level.Error)
            Timer.disable_all()

        # Create IPC engine and world
        import os
        import tempfile

        # Create workspace directory for IPC output
        workspace = os.path.join(tempfile.gettempdir(), "genesis_ipc_workspace")
        os.makedirs(workspace, exist_ok=True)
        self._ipc_engine = Engine("cuda", workspace)
        self._ipc_world = World(self._ipc_engine)

        # Create IPC scene with configuration
        config = Scene.default_config()
        config["dt"] = self.options.dt
        config["gravity"] = [[self.options.gravity[0]], [self.options.gravity[1]], [self.options.gravity[2]]]
        config["contact"]["d_hat"] = self.options.contact_d_hat
        config["contact"]["friction"]["enable"] = self.options.contact_friction_enable
        config["newton"]["velocity_tol"] = self.options.newton_velocity_tol
        config["line_search"]["max_iter"] = self.options.line_search_max_iter
        config["linear_system"]["tol_rate"] = self.options.linear_system_tol_rate
        config["sanity_check"]["enable"] = self.options.sanity_check_enable

        self._ipc_scene = Scene(config)

        # Create constitutions
        self._ipc_abd = AffineBodyConstitution()
        self._ipc_stk = StableNeoHookean()
        self._ipc_nks = NeoHookeanShell()  # For cloth
        self._ipc_dsb = DiscreteShellBending()  # For cloth bending

        # Add constitutions to scene
        self._ipc_scene.constitution_tabular().insert(self._ipc_abd)
        self._ipc_scene.constitution_tabular().insert(self._ipc_stk)
        # Note: Shell constitutions are added on-demand when cloth entities exist

        # Set up contact model (physical parameters)
        self._ipc_scene.contact_tabular().default_model(
            self.options.contact_friction_mu, self.options.contact_resistance
        )

        # Create separate contact elements for ABD, FEM, Cloth, and Ground to control their interactions
        self._ipc_abd_contact = self._ipc_scene.contact_tabular().create("abd_contact")
        self._ipc_fem_contact = self._ipc_scene.contact_tabular().create("fem_contact")
        self._ipc_cloth_contact = self._ipc_scene.contact_tabular().create("cloth_contact")
        self._ipc_ground_contact = self._ipc_scene.contact_tabular().create("ground_contact")

        # Configure contact interactions based on IPC coupler options
        # FEM-FEM: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_fem_contact,
            self._ipc_fem_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )
        # FEM-ABD: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_fem_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )
        # ABD-ABD: controlled by IPC_self_contact option
        self._ipc_scene.contact_tabular().insert(
            self._ipc_abd_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            self.options.IPC_self_contact,
        )
        # Cloth-Cloth: always enabled for cloth self-collision (necessary to prevent self-penetration)
        self._ipc_scene.contact_tabular().insert(
            self._ipc_cloth_contact,
            self._ipc_cloth_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )  # Always enable cloth self-collision
        # Cloth-FEM: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_cloth_contact,
            self._ipc_fem_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )
        # Cloth-ABD: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_cloth_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )

        # Ground contact interactions
        # Ground-ABD (rigid bodies): controlled by disable_ipc_ground_contact option
        self._ipc_scene.contact_tabular().insert(
            self._ipc_ground_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            not self.options.disable_ipc_ground_contact,
        )
        # Ground-FEM: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_ground_contact,
            self._ipc_fem_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )
        # Ground-Cloth: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_ground_contact,
            self._ipc_cloth_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )

        # Set up subscenes for multi-environment (scene grouping)
        # Only use subscenes when B > 1 to avoid issues with ground collision
        # (ground's subscene support is incomplete in libuipc)
        B = self.sim._B
        self._ipc_scene_subscenes = {}
        self._use_subscenes = B > 1

        if self._use_subscenes:
            for i in range(B):
                self._ipc_scene_subscenes[i] = self._ipc_scene.subscene_tabular().create(f"subscene{i}")

            # Disable contact between different environments
            for i in range(B):
                for j in range(B):
                    if i != j:
                        self._ipc_scene.subscene_tabular().insert(
                            self._ipc_scene_subscenes[i], self._ipc_scene_subscenes[j], False
                        )

    def _add_objects_to_ipc(self):
        """Add objects from solvers to IPC system"""
        # Add FEM entities to IPC
        if self.fem_solver.is_active:
            self._add_fem_entities_to_ipc()

        # Add rigid geoms to IPC
        if self.rigid_solver.is_active:
            self._add_rigid_geoms_to_ipc()

    def _add_fem_entities_to_ipc(self):
        """Add FEM entities to the existing IPC scene (includes both volumetric FEM and cloth)"""
        from uipc.constitution import ElasticModuli
        from uipc.geometry import label_surface, tetmesh, trimesh
        from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial

        fem_solver = self.fem_solver
        scene = self._ipc_scene
        stk = self._ipc_stk  # StableNeoHookean for volumetric FEM
        nks = self._ipc_nks  # NeoHookeanShell for cloth
        dsb = self._ipc_dsb  # DiscreteShellBending for cloth
        scene_subscenes = self._ipc_scene_subscenes

        fem_solver._mesh_handles = {}
        fem_solver.list_env_obj = []
        fem_solver.list_env_mesh = []

        for i_b in range(self.sim._B):
            fem_solver.list_env_obj.append([])
            fem_solver.list_env_mesh.append([])
            for i_e, entity in enumerate(fem_solver._entities):
                is_cloth = isinstance(entity.material, ClothMaterial)

                # Create object in IPC
                obj_name = f"cloth_{i_b}_{i_e}" if is_cloth else f"fem_{i_b}_{i_e}"
                fem_solver.list_env_obj[i_b].append(scene.objects().create(obj_name))

                # Create mesh: trimesh for cloth (2D shell), tetmesh for volumetric FEM (3D)
                if is_cloth:
                    # Cloth: use surface triangles only
                    verts = entity.init_positions.cpu().numpy().astype(np.float64)
                    faces = entity.surface_triangles.astype(np.int32)
                    mesh = trimesh(verts, faces)
                else:
                    # Volumetric FEM: use tetrahedral mesh
                    mesh = tetmesh(entity.init_positions.cpu().numpy(), entity.elems)

                fem_solver.list_env_mesh[i_b].append(mesh)

                # Add to contact subscene (only for multi-environment)
                if self._use_subscenes:
                    scene_subscenes[i_b].apply_to(mesh)

                # Apply contact element based on type
                if is_cloth:
                    self._ipc_cloth_contact.apply_to(mesh)
                else:
                    self._ipc_fem_contact.apply_to(mesh)

                label_surface(mesh)

                # Apply material constitution based on type
                moduli = ElasticModuli.youngs_poisson(entity.material.E, entity.material.nu)
                if is_cloth:
                    # Apply shell material for cloth
                    nks.apply_to(
                        mesh, moduli=moduli, mass_density=entity.material.rho, thickness=entity.material.thickness
                    )
                    # Apply bending stiffness if specified
                    if entity.material.bending_stiffness is not None:
                        dsb.apply_to(mesh, bending_stiffness=entity.material.bending_stiffness)
                else:
                    # Apply volumetric material for FEM
                    stk.apply_to(mesh, moduli, mass_density=entity.material.rho)

                # Add metadata to identify geometry type
                meta_attrs = mesh.meta()
                meta_attrs.create("solver_type", "cloth" if is_cloth else "fem")
                meta_attrs.create("env_idx", str(i_b))
                meta_attrs.create("entity_idx", str(i_e))

                # Create geometry in IPC scene
                fem_solver.list_env_obj[i_b][i_e].geometries().create(mesh)
                fem_solver._mesh_handles[f"gs_ipc_{i_b}_{i_e}"] = mesh

    def _add_rigid_geoms_to_ipc(self):
        """Add rigid geoms to the existing IPC scene as ABD objects, merging geoms by link_idx"""
        from uipc.geometry import tetmesh, label_surface, label_triangle_orient, flip_inward_triangles, merge, ground
        from genesis.utils import mesh as mu
        import numpy as np
        import trimesh

        rigid_solver = self.rigid_solver
        scene = self._ipc_scene
        abd = self._ipc_abd
        scene_subscenes = self._ipc_scene_subscenes

        # Initialize lists following FEM solver pattern
        rigid_solver.list_env_obj = []
        rigid_solver.list_env_mesh = []
        rigid_solver._mesh_handles = {}
        rigid_solver._abd_transforms = {}

        for i_b in range(self.sim._B):
            rigid_solver.list_env_obj.append([])
            rigid_solver.list_env_mesh.append([])

            # Group geoms by link_idx for merging
            link_geoms = {}  # link_idx -> dict with 'meshes', 'link_world_pos', 'link_world_quat', 'entity_idx'
            link_planes = {}  # link_idx -> list of plane geoms (handle separately)

            # First pass: collect and group geoms by link_idx
            for i_g in range(rigid_solver.n_geoms_):
                geom_type = rigid_solver.geoms_info.type[i_g]
                link_idx = rigid_solver.geoms_info.link_idx[i_g]
                entity_idx = rigid_solver.links_info.entity_idx[link_idx]
                entity = rigid_solver._entities[entity_idx]

                # Check if this link should be included in IPC based on coupler's filter
                if entity_idx in self._ipc_link_filters:
                    link_filter = self._ipc_link_filters[entity_idx]
                    if link_filter is not None and link_idx not in link_filter:
                        continue  # Skip this geom/link

                # Initialize link group if not exists
                if link_idx not in link_geoms:
                    link_geoms[link_idx] = {
                        "meshes": [],
                        "link_world_pos": None,
                        "link_world_quat": None,
                        "entity_idx": entity_idx,
                    }
                    link_planes[link_idx] = []

                try:
                    if geom_type == gs.GEOM_TYPE.PLANE:
                        # Handle planes separately (they can't be merged with SimplicialComplex)
                        # Ground/plane will be assigned to ground_contact element for selective collision control
                        pos = rigid_solver.geoms_info.pos[i_g].to_numpy()
                        normal = np.array([0.0, 0.0, 1.0])  # Z-up
                        height = np.dot(pos, normal)
                        plane_geom = ground(height, normal)
                        link_planes[link_idx].append((i_g, plane_geom))

                    else:
                        # For all non-plane geoms, create tetmesh
                        vert_num = rigid_solver.geoms_info.vert_num[i_g]
                        if vert_num == 0:
                            continue  # Skip geoms without vertices

                        # Extract vertex and face data
                        vert_start = rigid_solver.geoms_info.vert_start[i_g]
                        vert_end = rigid_solver.geoms_info.vert_end[i_g]
                        face_start = rigid_solver.geoms_info.face_start[i_g]
                        face_end = rigid_solver.geoms_info.face_end[i_g]

                        # Get vertices and faces
                        geom_verts = rigid_solver.verts_info.init_pos.to_numpy()[vert_start:vert_end]
                        geom_faces = rigid_solver.faces_info.verts_idx.to_numpy()[face_start:face_end]
                        geom_faces = geom_faces - vert_start  # Adjust indices

                        # Apply geom-relative transform to vertices (needed for merging)
                        geom_rel_pos = rigid_solver.geoms_info.pos[i_g].to_numpy()
                        geom_rel_quat = rigid_solver.geoms_info.quat[i_g].to_numpy()

                        # Transform vertices by geom relative transform
                        import genesis.utils.geom as gu

                        geom_rot_mat = gu.quat_to_R(geom_rel_quat)
                        transformed_verts = geom_verts @ geom_rot_mat.T + geom_rel_pos

                        # Create uipc trimesh for rigid body (ABD doesn't need tetmesh)
                        try:
                            from uipc.geometry import trimesh as uipc_trimesh

                            # Create uipc trimesh directly (dim=2, surface mesh for ABD)
                            rigid_mesh = uipc_trimesh(transformed_verts.astype(np.float64), geom_faces.astype(np.int32))

                            # Store uipc mesh (SimplicialComplex) for merging
                            link_geoms[link_idx]["meshes"].append((i_g, rigid_mesh))

                        except Exception as e:
                            gs.logger.warning(f"Failed to convert trimesh to tetmesh for geom {i_g}: {e}")
                            continue

                    # Store link transform info (same for all geoms in link)
                    if link_geoms[link_idx]["link_world_pos"] is None:
                        link_geoms[link_idx]["link_world_pos"] = rigid_solver.links_state.pos[link_idx, i_b]
                        link_geoms[link_idx]["link_world_quat"] = rigid_solver.links_state.quat[link_idx, i_b]

                except Exception as e:
                    gs.logger.warning(f"Failed to process geom {i_g}: {e}")
                    continue

            # Second pass: merge geoms per link and create IPC objects
            link_obj_counter = 0
            for link_idx, link_data in link_geoms.items():
                try:
                    # Handle regular meshes (merge if multiple)
                    if link_data["meshes"]:
                        if len(link_data["meshes"]) == 1:
                            # Single mesh in link
                            geom_idx, merged_mesh = link_data["meshes"][0]
                        else:
                            # Multiple meshes in link - merge them using uipc.geometry.merge
                            meshes_to_merge = [mesh for geom_idx, mesh in link_data["meshes"]]
                            merged_mesh = merge(meshes_to_merge)
                            geom_idx = link_data["meshes"][0][0]  # Use first geom's index for metadata

                        # Apply link world transform
                        from uipc import view, Transform, Vector3, Quaternion

                        trans_view = view(merged_mesh.transforms())
                        t = Transform.Identity()

                        link_world_pos = link_data["link_world_pos"]
                        link_world_quat = link_data["link_world_quat"]

                        # Ensure numpy format
                        link_world_pos = link_world_pos.to_numpy()
                        link_world_quat = link_world_quat.to_numpy()

                        t.translate(Vector3.Values((link_world_pos[0], link_world_pos[1], link_world_pos[2])))
                        uipc_link_quat = Quaternion(link_world_quat)
                        t.rotate(uipc_link_quat)
                        trans_view[0] = t.matrix()

                        # Process surface for contact
                        label_surface(merged_mesh)

                        # Create rigid object
                        rigid_obj = scene.objects().create(f"rigid_link_{i_b}_{link_idx}")
                        rigid_solver.list_env_obj[i_b].append(rigid_obj)
                        rigid_solver.list_env_mesh[i_b].append(merged_mesh)

                        # Add to contact subscene and apply ABD constitution (only for multi-environment)
                        if self._use_subscenes:
                            scene_subscenes[i_b].apply_to(merged_mesh)
                        self._ipc_abd_contact.apply_to(merged_mesh)
                        from uipc.unit import MPa

                        # Check if this link is IPC-only
                        is_ipc_only = (
                            link_data["entity_idx"] in self._ipc_only_links
                            and link_idx in self._ipc_only_links[link_data["entity_idx"]]
                        )

                        entity_rho = rigid_solver._entities[link_data["entity_idx"]].material.rho

                        if is_ipc_only:
                            # IPC-only links use full density (no mass splitting with Genesis)
                            abd.apply_to(
                                merged_mesh,
                                kappa=10.0 * MPa,
                                mass_density=entity_rho,
                            )
                        else:
                            # Regular coupled links use half density for IPC ABD to avoid double-counting mass
                            # (the other half is in Genesis rigid solver, scaled in _scale_genesis_rigid_link_masses)
                            abd.apply_to(
                                merged_mesh,
                                kappa=10.0 * MPa,
                                mass_density=entity_rho / 2.0,
                            )

                            # Apply soft transform constraints only for coupled links (not IPC-only)
                            from uipc.constitution import SoftTransformConstraint

                            if not hasattr(self, "_ipc_stc"):
                                self._ipc_stc = SoftTransformConstraint()
                                scene.constitution_tabular().insert(self._ipc_stc)

                            strength_tuple = self.options.ipc_constraint_strength
                            constraint_strength = np.array(
                                [
                                    strength_tuple[0],  # translation strength
                                    strength_tuple[1],  # rotation strength
                                ]
                            )
                            self._ipc_stc.apply_to(merged_mesh, constraint_strength)

                        # Add metadata
                        meta_attrs = merged_mesh.meta()
                        meta_attrs.create("solver_type", "rigid")
                        meta_attrs.create("env_idx", str(i_b))
                        meta_attrs.create("link_idx", str(link_idx))  # Use link_idx instead of geom_idx

                        rigid_obj.geometries().create(merged_mesh)

                        # Set up animator for this link
                        if not hasattr(self, "_ipc_animator"):
                            self._ipc_animator = scene.animator()

                        def create_animate_function(env_idx, link_idx, coupler_ref):
                            def animate_rigid_link(info):
                                from uipc import view, builtin

                                geo_slots = info.geo_slots()
                                if len(geo_slots) == 0:
                                    return
                                geo = geo_slots[0].geometry()

                                try:
                                    # Read stored Genesis transform (q_genesis^n)
                                    # This was stored in _store_genesis_rigid_states() before advance()
                                    if hasattr(coupler_ref, "_genesis_stored_states"):
                                        stored_states = coupler_ref._genesis_stored_states
                                        if link_idx in stored_states and env_idx in stored_states[link_idx]:
                                            transform_matrix = stored_states[link_idx][env_idx]

                                            # Enable constraint and set target transform
                                            is_constrained = geo.instances().find(builtin.is_constrained)
                                            aim_transform_attr = geo.instances().find(builtin.aim_transform)

                                            if is_constrained and aim_transform_attr:
                                                view(is_constrained)[0] = 1
                                                view(aim_transform_attr)[:] = transform_matrix

                                except Exception as e:
                                    gs.logger.warning(f"Error setting IPC animation target: {e}")

                            return animate_rigid_link

                        animate_func = create_animate_function(i_b, link_idx, self)
                        self._ipc_animator.insert(rigid_obj, animate_func)

                        rigid_solver._mesh_handles[f"rigid_link_{i_b}_{link_idx}"] = merged_mesh
                        link_obj_counter += 1

                    # Handle planes for this link separately
                    for geom_idx, plane_geom in link_planes[link_idx]:
                        plane_obj = scene.objects().create(f"rigid_plane_{i_b}_{geom_idx}")
                        rigid_solver.list_env_obj[i_b].append(plane_obj)
                        rigid_solver.list_env_mesh[i_b].append(None)  # Planes are ImplicitGeometry

                        # Apply ground contact element to plane
                        self._ipc_ground_contact.apply_to(plane_geom)

                        plane_obj.geometries().create(plane_geom)
                        rigid_solver._mesh_handles[f"rigid_plane_{i_b}_{geom_idx}"] = plane_geom
                        link_obj_counter += 1

                except Exception as e:
                    gs.logger.warning(f"Failed to create IPC object for link {link_idx}: {e}")
                    continue

        # Scale down Genesis rigid solver masses for links added to IPC
        # Since both Genesis and IPC simulate these rigid bodies, divide mass by 2
        self._scale_genesis_rigid_link_masses(link_geoms)

    def _scale_genesis_rigid_link_masses(self, link_geoms_dict):
        """
        Scale down Genesis rigid solver mass properties for links that were added to IPC.
        Both Genesis and IPC will simulate these rigid bodies, so we divide by 2 to avoid
        double-counting mass.

        This scales:
        - inertial_mass: scalar mass
        - inertial_i: 3x3 inertia tensor (scales linearly with mass)

        Parameters
        ----------
        link_geoms_dict : dict
            Dictionary mapping link_idx to their geometry data (from _add_rigid_geoms_to_ipc)
        """
        rigid_solver = self.rigid_solver

        # Get all link indices that were added to IPC
        ipc_link_indices = set(link_geoms_dict.keys())

        if not ipc_link_indices:
            return

        gs.logger.info(f"Scaling Genesis rigid mass for {len(ipc_link_indices)} links added to IPC (dividing by 2)")

        # Scale mass properties for each link
        for link_idx in ipc_link_indices:
            # Scale inertial mass
            original_mass = float(rigid_solver.links_info.inertial_mass[link_idx])
            rigid_solver.links_info.inertial_mass[link_idx] = original_mass / 2.0

            # Scale inertia tensor (inertia scales linearly with mass for same geometry)
            original_inertia = rigid_solver.links_info.inertial_i[link_idx]
            rigid_solver.links_info.inertial_i[link_idx] = original_inertia / 2.0

            gs.logger.debug(
                f"  Link {link_idx}: mass {original_mass:.6f} -> {original_mass/2.0:.6f} kg, " f"inertia scaled by 0.5"
            )

        # After scaling inertial_mass and inertial_i, we need to recompute derived quantities:
        # - mass_mat: mass matrix (computed from inertial_mass and inertial_i)
        # - invweight: inverse weight (computed from mass_mat)
        # - meaninertia: mean inertia (computed from mass_mat)
        gs.logger.info("Recomputing mass matrix and derived quantities after scaling")
        rigid_solver._init_invweight_and_meaninertia(force_update=True)

    def _finalize_ipc(self):
        """Finalize IPC setup"""
        self._ipc_world.init(self._ipc_scene)
        gs.logger.info("IPC world initialized successfully")

    @property
    def is_active(self) -> bool:
        """Check if IPC coupling is active"""
        return self._ipc_world is not None

    def set_link_ipc_coupling_type(self, entity, coupling_type: str, link_names=None, link_indices=None):
        """
        Set IPC coupling type for links of an entity.

        Parameters
        ----------
        entity : RigidEntity
            The rigid entity to configure
        coupling_type : str
            Type of coupling: 'both', 'ipc_only', or 'genesis_only'
            - 'both': Two-way coupling between IPC and Genesis (default behavior)
            - 'ipc_only': Links only simulated in IPC, transforms copied to Genesis (one-way)
            - 'genesis_only': Links only simulated in Genesis, excluded from IPC
        link_names : list of str, optional
            Names of links to configure. If None and link_indices is None, applies to all links.
        link_indices : list of int, optional
            Local indices of links to configure. If None and link_names is None, applies to all links.

        Notes
        -----
        - 'both': Links use half density in IPC, have SoftTransformConstraint, bidirectional forces
        - 'ipc_only': Links use full density in IPC, no SoftTransformConstraint, transforms copied to Genesis
        - 'genesis_only': Links excluded from IPC simulation entirely
        """
        entity_idx = entity._idx

        # Determine which links to configure
        if link_names is None and link_indices is None:
            # Apply to all links
            target_links = set()
            for local_idx in range(entity.n_links):
                solver_link_idx = local_idx + entity._link_start
                target_links.add(solver_link_idx)
        else:
            # Apply to specified links
            target_links = set()

            if link_names is not None:
                for name in link_names:
                    link = entity.get_link(name=name)
                    if link is not None:
                        target_links.add(link.idx)
                    else:
                        gs.logger.warning(f"Link name '{name}' not found in entity")

            if link_indices is not None:
                for local_idx in link_indices:
                    solver_link_idx = local_idx + entity._link_start
                    target_links.add(solver_link_idx)

        # Apply coupling type
        if coupling_type == "both":
            # Two-way coupling: include in IPC, not in IPC-only
            self._ipc_link_filters[entity_idx] = target_links

            # Remove from IPC-only if present
            if entity_idx in self._ipc_only_links:
                self._ipc_only_links[entity_idx] -= target_links
                if not self._ipc_only_links[entity_idx]:
                    del self._ipc_only_links[entity_idx]

            gs.logger.info(f"Entity {entity_idx}: {len(target_links)} link(s) set to 'both' coupling")

        elif coupling_type == "ipc_only":
            # One-way coupling: IPC -> Genesis
            if entity_idx not in self._ipc_only_links:
                self._ipc_only_links[entity_idx] = set()
            self._ipc_only_links[entity_idx].update(target_links)

            # Also add to IPC link filter
            if entity_idx not in self._ipc_link_filters:
                self._ipc_link_filters[entity_idx] = set()
            self._ipc_link_filters[entity_idx].update(target_links)

            gs.logger.info(f"Entity {entity_idx}: {len(target_links)} link(s) set to 'ipc_only' coupling")

        elif coupling_type == "genesis_only":
            # Genesis-only: remove from both filters
            if entity_idx in self._ipc_link_filters:
                self._ipc_link_filters[entity_idx] -= target_links
                if not self._ipc_link_filters[entity_idx]:
                    del self._ipc_link_filters[entity_idx]

            if entity_idx in self._ipc_only_links:
                self._ipc_only_links[entity_idx] -= target_links
                if not self._ipc_only_links[entity_idx]:
                    del self._ipc_only_links[entity_idx]

            gs.logger.info(
                f"Entity {entity_idx}: {len(target_links)} link(s) set to 'genesis_only' (excluded from IPC)"
            )

        else:
            raise ValueError(
                f"Invalid coupling_type '{coupling_type}'. " f"Must be 'both', 'ipc_only', or 'genesis_only'."
            )

    def preprocess(self, f):
        """Preprocessing step before coupling"""
        pass

    def _store_genesis_rigid_states(self):
        """
        Store current Genesis rigid body states before IPC advance.
        These stored states will be used by:
        1. Animator: to set aim_transform for IPC soft constraints
        2. Force computation: to ensure action-reaction force consistency
        """
        if not self.rigid_solver.is_active:
            return

        rigid_solver = self.rigid_solver

        # Clear previous stored states
        self._genesis_stored_states.clear()

        # Store transforms for all rigid links
        # Iterate through mesh handles to get all links
        if hasattr(rigid_solver, "_mesh_handles"):
            for handle_key in rigid_solver._mesh_handles.keys():
                if handle_key.startswith("rigid_link_"):
                    # Parse: "rigid_link_{env_idx}_{link_idx}"
                    parts = handle_key.split("_")
                    if len(parts) >= 4:
                        env_idx = int(parts[2])
                        link_idx = int(parts[3])

                        # Get and store current Genesis transform
                        genesis_transform = self._get_genesis_link_transform(link_idx, env_idx)

                        if link_idx not in self._genesis_stored_states:
                            self._genesis_stored_states[link_idx] = {}
                        self._genesis_stored_states[link_idx][env_idx] = genesis_transform

    def couple(self, f):
        """Execute IPC coupling step"""
        if not self.is_active:
            return

        # Dispatch to strategy-specific coupling logic
        if self.options.coupling_strategy == "two_way_soft_constraint":
            self._couple_two_way_soft_constraint(f)
        elif self.options.coupling_strategy == "contact_proxy":
            self._couple_contact_proxy(f)

    def _couple_two_way_soft_constraint(self, f):
        """Two-way coupling using SoftTransformConstraint"""
        # Step 1: Store current Genesis rigid body states (q_genesis^n)
        # This will be used by both animator (to set aim_transform) and
        # force computation (to ensure action-reaction force consistency)
        self._store_genesis_rigid_states()

        # Step 2: Advance IPC simulation
        # Animator reads stored Genesis states and sets them as IPC targets
        self._ipc_world.advance()
        self._ipc_world.retrieve()

        # Step 3: Retrieve IPC results and apply coupling forces
        # Now use IPC's new positions (q_ipc^{n+1}) and stored Genesis states (q_genesis^n)
        # to compute forces: F = M * (q_ipc^{n+1} - q_genesis^n)
        self._retrieve_fem_states(f)  # This handles both volumetric FEM and cloth
        self._retrieve_rigid_states(f)
        # Handle IPC-only links: directly set Genesis transform to IPC result (one-way coupling)
        self._set_genesis_transforms_from_ipc()

        # Apply coupling forces from IPC ABD to Genesis rigid bodies (two-way coupling)
        # Based on soft_transform_constraint.cu gradient computation
        if self.options.two_way_coupling:
            self._apply_abd_coupling_forces()

    def _couple_contact_proxy(self, f):
        """Contact proxy coupling strategy (placeholder)"""
        # TODO: Implement contact proxy coupling logic

        # Step 2: Advance IPC simulation
        # Animator reads stored Genesis states and sets them as IPC targets
        self._ipc_world.advance()
        self._ipc_world.retrieve()

        # Step 3: Retrieve IPC results and apply coupling forces
        # Now use IPC's new positions (q_ipc^{n+1}) and stored Genesis states (q_genesis^n)
        # to compute forces: F = M * (q_ipc^{n+1} - q_genesis^n)
        self._retrieve_fem_states(f)  # This handles both volumetric FEM and cloth
        self._retrieve_rigid_states(f)
        self._record_ipc_contact_forces()

        pass

    def _retrieve_fem_states(self, f):
        # IPC world advance/retrieve is handled at Scene level
        # This method handles both volumetric FEM (3D) and cloth (2D) post-processing

        if not self.fem_solver.is_active:
            return

        # Gather FEM states (both volumetric and cloth) using metadata filtering
        from uipc import builtin
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot, apply_transform, merge
        import numpy as np

        visitor = SceneVisitor(self._ipc_scene)

        # Collect FEM and cloth geometries using metadata
        fem_geo_by_entity = {}
        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                # Accept both 3D (volumetric FEM) and 2D (cloth) geometries
                if geo.dim() in [2, 3]:
                    try:
                        # Check solver type using metadata
                        meta_attrs = geo.meta()
                        solver_type_attr = meta_attrs.find("solver_type")

                        if solver_type_attr and solver_type_attr.name() == "solver_type":
                            # Read solver type from metadata
                            try:
                                solver_type_view = solver_type_attr.view()
                                if len(solver_type_view) > 0:
                                    solver_type = str(solver_type_view[0])
                                else:
                                    continue
                            except:
                                continue

                            # Accept both "fem" and "cloth" (both are FEM entities)
                            if solver_type in ["fem", "cloth"]:
                                env_idx_attr = meta_attrs.find("env_idx")
                                entity_idx_attr = meta_attrs.find("entity_idx")

                                if env_idx_attr and entity_idx_attr:
                                    # Read string values and convert to int
                                    env_idx_str = str(env_idx_attr.view()[0])
                                    entity_idx_str = str(entity_idx_attr.view()[0])
                                    env_idx = int(env_idx_str)
                                    entity_idx = int(entity_idx_str)

                                    if entity_idx not in fem_geo_by_entity:
                                        fem_geo_by_entity[entity_idx] = {}

                                    proc_geo = geo
                                    if geo.instances().size() >= 1:
                                        proc_geo = merge(apply_transform(geo))
                                    pos = proc_geo.positions().view().reshape(-1, 3)
                                    fem_geo_by_entity[entity_idx][env_idx] = pos

                    except Exception as e:
                        # Skip this geometry if metadata reading fails
                        continue

        # Update FEM entities using filtered geometries
        for entity_idx, env_positions in fem_geo_by_entity.items():
            if entity_idx < len(self.fem_solver._entities):
                entity = self.fem_solver._entities[entity_idx]
                env_pos_list = []

                for env_idx in range(self.sim._B):
                    if env_idx in env_positions:
                        env_pos_list.append(env_positions[env_idx])
                    else:
                        # Fallback for missing environment
                        env_pos_list.append(np.zeros((0, 3)))

                if env_pos_list:
                    all_env_pos = np.stack(env_pos_list, axis=0, dtype=gs.np_float)
                    entity.set_pos(0, all_env_pos)

    def _retrieve_rigid_states(self, f):
        """
        Handle rigid body IPC: Retrieve ABD transforms/affine matrices after IPC step
        and apply coupling forces back to Genesis rigid bodies
        """
        # IPC world advance/retrieve is handled at Scene level
        # Retrieve ABD transform matrices after IPC simulation

        if not hasattr(self, "_ipc_scene") or not hasattr(self.rigid_solver, "list_env_mesh"):
            return

        from uipc import builtin, view
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot
        import numpy as np
        import genesis.utils.geom as gu

        rigid_solver = self.rigid_solver
        visitor = SceneVisitor(self._ipc_scene)

        # Collect ABD geometries and their constraint data using metadata
        abd_data_by_link = {}  # link_idx -> {env_idx: {transform, gradient, mass}}

        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                if geo.dim() in [2, 3]:
                    try:
                        # Check if this is an ABD geometry using metadata
                        meta_attrs = geo.meta()
                        solver_type_attr = meta_attrs.find("solver_type")

                        if solver_type_attr and solver_type_attr.name() == "solver_type":
                            # Actually read solver type from metadata
                            try:
                                solver_type_view = solver_type_attr.view()
                                if len(solver_type_view) > 0:
                                    solver_type = str(solver_type_view[0])
                                else:
                                    continue
                            except:
                                continue

                            if solver_type == "rigid":
                                env_idx_attr = meta_attrs.find("env_idx")
                                link_idx_attr = meta_attrs.find("link_idx")

                                if env_idx_attr and link_idx_attr:
                                    # Read metadata values
                                    env_idx_str = str(env_idx_attr.view()[0])
                                    link_idx_str = str(link_idx_attr.view()[0])
                                    env_idx = int(env_idx_str)
                                    link_idx = int(link_idx_str)

                                    # Initialize link data structure
                                    if link_idx not in abd_data_by_link:
                                        abd_data_by_link[link_idx] = {}

                                    # Get current transform matrix from ABD object (after IPC solve)
                                    # This is q_ipc^{n+1}
                                    transforms = geo.transforms()
                                    transform_matrix = None
                                    if transforms.size() > 0:
                                        transform_matrix = view(transforms)[0].copy()  # 4x4 affine matrix

                                    # Get aim transform that was used by IPC during solve
                                    # This is q_genesis^n (stored before advance)
                                    aim_transform = None
                                    if (
                                        link_idx in self._genesis_stored_states
                                        and env_idx in self._genesis_stored_states[link_idx]
                                    ):
                                        aim_transform = self._genesis_stored_states[link_idx][env_idx]

                                    abd_data_by_link[link_idx][env_idx] = {
                                        "transform": transform_matrix,  # q_ipc^{n+1}
                                        "aim_transform": aim_transform,  # q_genesis^n
                                    }

                    except Exception as e:
                        gs.logger.warning(f"Failed to retrieve ABD geometry data: {e}")
                        continue

        # Store transforms for later access
        self.abd_data_by_link = abd_data_by_link

    def _set_genesis_transforms_from_ipc(self, ipc_only=True):
        """
        Set Genesis transforms from IPC results.

        Parameters
        ----------
        ipc_only : bool
            If True, only process links that are both IPC-only AND in IPC filters.
            If False, process all links in IPC filters (regardless of IPC-only status).
        """
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        import torch

        rigid_solver = self.rigid_solver
        is_parallelized = self.sim._scene.n_envs > 0

        # Step 1: Filter links based on ipc_only flag
        filtered_links = {}  # {(entity_idx, link_idx, env_idx): transform_data}

        for link_idx, env_data in self.abd_data_by_link.items():
            # Find which entity this link belongs to
            entity_idx = None
            for ent_idx in rigid_solver._entities.keys():
                entity = rigid_solver._entities[ent_idx]
                if entity._link_start <= link_idx < entity._link_start + entity.n_links:
                    entity_idx = ent_idx
                    break

            if entity_idx is None:
                continue

            # Check filtering criteria
            if ipc_only:
                # Must be both IPC-only AND in IPC filters
                is_ipc_only = entity_idx in self._ipc_only_links and link_idx in self._ipc_only_links[entity_idx]
                is_in_filter = entity_idx in self._ipc_link_filters and link_idx in self._ipc_link_filters[entity_idx]
                if not (is_ipc_only and is_in_filter):
                    continue
            else:
                # Must be in IPC filters
                is_in_filter = entity_idx in self._ipc_link_filters and link_idx in self._ipc_link_filters[entity_idx]
                if not is_in_filter:
                    continue

            # Store filtered link data
            for env_idx, data in env_data.items():
                filtered_links[(entity_idx, link_idx, env_idx)] = data

        # Step 2: Group filtered links by entity and env
        entity_env_links = {}  # {(entity_idx, env_idx): [(link_idx, transform_data), ...]}

        for (entity_idx, link_idx, env_idx), data in filtered_links.items():
            key = (entity_idx, env_idx)
            if key not in entity_env_links:
                entity_env_links[key] = []
            entity_env_links[key].append((link_idx, data))

        # Step 3: Process each entity-env group
        for (entity_idx, env_idx), link_data_list in entity_env_links.items():
            entity = rigid_solver._entities[entity_idx]

            try:
                # Check if entity has only one link and it's the base link
                if len(link_data_list) == 1:
                    link_idx, data = link_data_list[0]
                    if link_idx == entity.base_link_idx:
                        # Simple case: single base link
                        ipc_transform = data.get("transform")
                        if ipc_transform is None:
                            continue

                        # Extract position and rotation
                        pos = ipc_transform[:3, 3]
                        rot_mat = ipc_transform[:3, :3]
                        quat_xyzw = R.from_matrix(rot_mat).as_quat()
                        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                        # Convert to tensors
                        pos_tensor = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                        quat_tensor = torch.as_tensor(quat_wxyz, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                        base_links_idx = torch.tensor([link_idx], dtype=gs.tc_int, device=gs.device)

                        # Set base link transform
                        if is_parallelized:
                            rigid_solver.set_base_links_pos(
                                pos_tensor,
                                base_links_idx,
                                envs_idx=env_idx,
                                relative=False,
                                unsafe=True,
                                skip_forward=False,
                            )
                            rigid_solver.set_base_links_quat(
                                quat_tensor,
                                base_links_idx,
                                envs_idx=env_idx,
                                relative=False,
                                unsafe=True,
                                skip_forward=False,
                            )
                        else:
                            rigid_solver.set_base_links_pos(
                                pos_tensor,
                                base_links_idx,
                                envs_idx=None,
                                relative=False,
                                unsafe=True,
                                skip_forward=False,
                            )
                            rigid_solver.set_base_links_quat(
                                quat_tensor,
                                base_links_idx,
                                envs_idx=None,
                                relative=False,
                                unsafe=True,
                                skip_forward=False,
                            )

                        # Zero velocities
                        if is_parallelized:
                            entity.zero_all_dofs_velocity(envs_idx=env_idx, unsafe=True)
                        else:
                            entity.zero_all_dofs_velocity(envs_idx=None, unsafe=True)

                        continue

                # Complex case: multiple links or non-base link
                # Use inverse kinematics to compute qpos

                # Prepare target positions and quaternions for IK
                links = []
                poss = []
                quats = []

                for link_idx, data in link_data_list:
                    ipc_transform = data.get("transform")
                    if ipc_transform is None:
                        continue

                    # Get link object
                    link = entity.get_link(idx=link_idx)
                    if link is None:
                        continue

                    # Extract position and quaternion
                    pos = ipc_transform[:3, 3]
                    rot_mat = ipc_transform[:3, :3]
                    quat_xyzw = R.from_matrix(rot_mat).as_quat()
                    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                    links.append(link)
                    poss.append(pos)
                    quats.append(quat_wxyz)

                if not links:
                    continue

                # Call inverse kinematics
                qpos = entity.inverse_kinematics_multilink(
                    links=links,
                    poss=poss,
                    quats=quats,
                    envs_idx=env_idx if is_parallelized else None,
                    return_error=False,
                )

                if qpos is not None:
                    # Set qpos for this entity
                    entity.set_qpos(qpos, envs_idx=env_idx if is_parallelized else None, zero_velocity=True)

            except Exception as e:
                gs.logger.warning(f"Failed to set Genesis transforms for entity {entity_idx}, env {env_idx}: {e}")
                continue

    def _get_genesis_link_transform(self, link_idx, env_idx):
        """
        Get the current transform (4x4 matrix) of a Genesis rigid body link.

        Parameters
        ----------
        link_idx : int
            The link index
        env_idx : int
            The environment index

        Returns
        -------
        np.ndarray
            4x4 transformation matrix
        """
        from uipc import Transform, Vector3, Quaternion
        import numpy as np

        rigid_solver = self.rigid_solver

        # Get current link state from Genesis
        link_pos = rigid_solver.get_links_pos(links_idx=link_idx, envs_idx=env_idx)
        link_quat = rigid_solver.get_links_quat(links_idx=link_idx, envs_idx=env_idx)

        link_pos = link_pos.detach().cpu().numpy()
        link_quat = link_quat.detach().cpu().numpy()

        # Handle array shapes - squeeze down to 1D
        while len(link_pos.shape) > 1 and link_pos.shape[0] == 1:
            link_pos = link_pos[0]
        while len(link_quat.shape) > 1 and link_quat.shape[0] == 1:
            link_quat = link_quat[0]

        pos_1d = link_pos.flatten()[:3]
        quat_1d = link_quat.flatten()[:4]

        # Create transform matrix
        t = Transform.Identity()
        t.translate(Vector3.Values((pos_1d[0], pos_1d[1], pos_1d[2])))
        uipc_quat = Quaternion(quat_1d)
        t.rotate(uipc_quat)

        return t.matrix().copy()

    def _apply_abd_coupling_forces(self):
        """
        Apply coupling forces from IPC ABD constraint to Genesis rigid bodies.

        This ensures action-reaction force consistency:
        - IPC constraint force: G_ipc = M * (q_ipc^{n+1} - q_genesis^n)
        - Genesis reaction force: F_genesis = M * (q_ipc^{n+1} - q_genesis^n) = G_ipc

        Where:
        - q_ipc^{n+1}: IPC ABD position after solve (from geo.transforms())
        - q_genesis^n: Genesis position before IPC advance (stored in _genesis_stored_states)
        - M: Mass matrix scaled by constraint strengths

        Based on soft_transform_constraint.cu implementation:
        - q is 12D: [translation(3), rotation_matrix_col_major(9)]
        - G is 12D: [linear_force(3), rotational_force(9)]
        - We extract linear force and convert rotational force to 3D torque
        """
        import numpy as np
        import genesis.utils.geom as gu

        rigid_solver = self.rigid_solver
        strength_tuple = self.options.ipc_constraint_strength
        translation_strength = strength_tuple[0]
        rotation_strength = strength_tuple[1]

        dt = self.sim._dt
        dt2 = dt * dt

        for link_idx, env_data in self.self.abd_data_by_link.items():
            # Skip IPC-only links (they don't need coupling forces)
            is_ipc_only = False
            for entity_idx, link_set in self._ipc_only_links.items():
                if link_idx in link_set:
                    is_ipc_only = True
                    break

            if is_ipc_only:
                continue  # Skip IPC-only links

            for env_idx, data in env_data.items():
                ipc_transform = data.get("transform")  # Current transform after IPC solve
                aim_transform = data.get("aim_transform")  # Target from Genesis

                if ipc_transform is None or aim_transform is None:
                    continue

                try:
                    # Extract current and target transforms (4x4 matrices)
                    T_current = ipc_transform  # Current ABD transform from IPC
                    T_aim = aim_transform  # Target transform from Genesis animator

                    # Extract translation and rotation components
                    # Current state (from IPC)
                    pos_current = T_current[:3, 3]
                    R_current = T_current[:3, :3]

                    # Target state (from Genesis)
                    pos_aim = T_aim[:3, 3]
                    R_aim = T_aim[:3, :3]

                    # Compute translation error: delta_pos = pos_current - pos_aim
                    delta_pos = pos_current - pos_aim

                    # Get link mass for scaling (similar to body_masses in CUDA code)
                    link_mass = rigid_solver.links_info.inertial_mass[link_idx]

                    # Compute generalized force (gradient) following soft_transform_constraint.cu:
                    # G = M * (q - q_aim)
                    # where M is the mass matrix, scaled by strength ratios

                    # Linear force component: F = translation_strength * mass * delta_pos
                    linear_force = translation_strength * link_mass * delta_pos / dt2

                    R_rel = R_current @ R_aim.T

                    from scipy.spatial.transform import Rotation as R

                    rotvec = R.from_matrix(R_rel).as_rotvec()

                    inertia_tensor_local = rigid_solver.links_info.inertial_i[link_idx].to_numpy()

                    # I_world = R_current * I_local * R_current^T
                    inertia_tensor_world = R_current @ inertia_tensor_local @ R_current.T

                    angular_torque = rotation_strength * inertia_tensor_world @ rotvec / dt2

                    # Format forces for Genesis API
                    # _sanitize_2D_io_variables expects:
                    # - Non-parallelized (n_envs=0): shape (n_links, 3)
                    # - Parallelized (n_envs>0): shape (n_envs, n_links, 3)
                    # It will use torch.as_tensor to convert numpy arrays to tensors

                    if self.sim._scene.n_envs > 0:
                        # Parallelized scene: shape (1, 1, 3) for (n_envs, n_links, 3)
                        force_input = linear_force.reshape(1, 1, 3)
                        torque_input = angular_torque.reshape(1, 1, 3)
                        apply_kwargs = {"links_idx": link_idx, "envs_idx": env_idx}
                    else:
                        # Non-parallelized scene: shape (1, 3) for (n_links, 3)
                        force_input = linear_force.reshape(1, 3)
                        torque_input = angular_torque.reshape(1, 3)
                        apply_kwargs = {
                            "links_idx": link_idx,
                        }

                    # Apply forces to Genesis rigid body
                    rigid_solver.apply_links_external_force(force=force_input, **apply_kwargs)

                    rigid_solver.apply_links_external_torque(torque=torque_input, **apply_kwargs)

                except Exception as e:
                    gs.logger.warning(f"Failed to apply ABD coupling force for link {link_idx}, env {env_idx}: {e}")
                    continue

    def couple_grad(self, f):
        """Gradient computation for coupling"""
        # IPC doesn't support gradients yet
        pass

    def reset(self, envs_idx=None):
        """Reset coupling state"""
        # IPC doesn't need special reset logic currently
        pass

    def _init_ipc_gui(self):
        """Initialize IPC GUI for debugging"""
        try:
            import polyscope as ps
            from uipc.gui import SceneGUI

            self.ps = ps

            # Initialize SceneGUI for IPC scene
            self._ipc_scene_gui = SceneGUI(self._ipc_scene)

            # Initialize polyscope if not already done
            if not ps.is_initialized():
                ps.init()

            # Register IPC GUI with polyscope
            self._ipc_scene_gui.register()
            self._ipc_scene_gui.set_edge_width(1)

            # Set up ground plane display in polyscope to match Genesis z=0
            ps.set_up_dir("z_up")
            ps.set_ground_plane_height(0.0)  # Set at z=0 to match Genesis

            # Show polyscope window for first frame to initialize properly
            ps.show(forFrames=1)
            # Flag to control GUI updates
            self.sim._scene._ipc_gui_enabled = True

            gs.logger.info("IPC GUI initialized successfully")

        except Exception as e:
            gs.logger.warning(f"Failed to initialize IPC GUI: {e}")
            self.sim._scene._ipc_gui_enabled = False

    def update_ipc_gui(self):
        """Update IPC GUI"""
        self.ps.frame_tick()  # Non-blocking frame update
        self._ipc_scene_gui.update()

    def _record_ipc_contact_forces(self):
        """
        Record contact forces from IPC for 'both' coupling links.

        This method extracts contact forces and torques from IPC's contact system
        and stores them for later application to Genesis rigid bodies.
        Only processes links that are in _ipc_link_filters but NOT in _ipc_only_links.
        """
        import numpy as np
        from uipc import view
        from uipc.geometry import Geometry

        # Clear previous contact forces
        self._ipc_contact_forces.clear()

        # Get contact feature from IPC world
        features = self._ipc_world.features()
        contact_feature = features.find("contact_system")

        if contact_feature is None:
            return  # No contact system available

        # Get available contact primitive types
        prim_types = contact_feature.contact_primitive_types()

        # Accumulate contact gradients (forces) for all vertices
        total_force_dict = {}  # {vertex_index: force_vector}

        for prim_type in prim_types:
            # Get contact gradient for this primitive type
            vert_grad = Geometry()
            contact_feature.contact_gradient(prim_type, vert_grad)

            # Extract gradient data from instances
            instances = vert_grad.instances()
            i_attr = instances.find("i")  # Vertex indices
            grad_attr = instances.find("grad")  # Gradient vectors

            if i_attr is not None and grad_attr is not None:
                indices = view(i_attr)
                gradients = view(grad_attr)

                # Accumulate gradients for each vertex
                for idx, grad in zip(indices, gradients):
                    grad_vec = np.array(grad).flatten()
                    if idx not in total_force_dict:
                        total_force_dict[idx] = np.zeros(3)
                    total_force_dict[idx] += grad_vec[:3]  # Take first 3 components

        if not total_force_dict:
            return  # No contact forces to process

        # Get current vertex positions from IPC scene
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot

        scene_visitor = SceneVisitor()
        self._ipc_scene.accept(scene_visitor)
        geometries = scene_visitor.geometries()

        # Build mapping from vertex index to link
        # We need to track which vertices belong to which rigid link
        vertex_to_link = {}  # {global_vertex_idx: (link_idx, env_idx, local_vertex_idx)}
        link_vertex_positions = {}  # {(link_idx, env_idx): [vertex_positions]}

        global_vertex_offset = 0

        for geo_idx, geo in enumerate(geometries):
            # Check if this geometry is a rigid body (has metadata)
            meta = geo.meta()
            solver_type_attr = meta.find("solver_type")

            if solver_type_attr is None:
                # Skip non-rigid geometries
                sc_slot = SimplicialComplexSlot(geo)
                if sc_slot.is_valid():
                    sc = sc_slot.topo()
                    n_verts = sc.vertices().size()
                    global_vertex_offset += n_verts
                continue

            solver_type = str(solver_type_attr.view()[0])

            if solver_type == "rigid":
                # Get link and env indices from metadata
                env_idx_attr = meta.find("env_idx")
                link_idx_attr = meta.find("link_idx")

                if env_idx_attr and link_idx_attr:
                    env_idx = int(str(env_idx_attr.view()[0]))
                    link_idx = int(str(link_idx_attr.view()[0]))

                    # Check if this is a 'both' coupling link
                    # (in _ipc_link_filters but NOT in _ipc_only_links)
                    is_both_coupling = False
                    for entity_idx, link_set in self._ipc_link_filters.items():
                        if link_idx in link_set:
                            # Check if it's NOT IPC-only
                            if (
                                entity_idx not in self._ipc_only_links
                                or link_idx not in self._ipc_only_links[entity_idx]
                            ):
                                is_both_coupling = True
                            break

                    if is_both_coupling:
                        # Get vertex positions and build mapping
                        sc_slot = SimplicialComplexSlot(geo)
                        if sc_slot.is_valid():
                            sc = sc_slot.topo()
                            n_verts = sc.vertices().size()

                            # Get vertex positions
                            positions = view(geo.positions())

                            # Store mapping and positions
                            if (link_idx, env_idx) not in link_vertex_positions:
                                link_vertex_positions[(link_idx, env_idx)] = []

                            for local_idx in range(n_verts):
                                global_idx = global_vertex_offset + local_idx
                                vertex_to_link[global_idx] = (link_idx, env_idx, local_idx)

                                # Store position
                                pos = np.array(positions[local_idx]).flatten()[:3]
                                link_vertex_positions[(link_idx, env_idx)].append(pos)

                            global_vertex_offset += n_verts
                    else:
                        # Not a both-coupling link, skip but update offset
                        sc_slot = SimplicialComplexSlot(geo)
                        if sc_slot.is_valid():
                            sc = sc_slot.topo()
                            n_verts = sc.vertices().size()
                            global_vertex_offset += n_verts
            else:
                # Not a rigid body, skip but update offset
                sc_slot = SimplicialComplexSlot(geo)
                if sc_slot.is_valid():
                    sc = sc_slot.topo()
                    n_verts = sc.vertices().size()
                    global_vertex_offset += n_verts

        # Compute contact forces and torques for each link
        link_forces = {}  # {(link_idx, env_idx): {'force': np.array, 'torque': np.array}}

        for vert_idx, force_grad in total_force_dict.items():
            if vert_idx not in vertex_to_link:
                continue  # Vertex doesn't belong to a both-coupling link

            link_idx, env_idx, local_idx = vertex_to_link[vert_idx]

            # Initialize force/torque storage for this link
            if (link_idx, env_idx) not in link_forces:
                link_forces[(link_idx, env_idx)] = {"force": np.zeros(3), "torque": np.zeros(3), "center": None}

                # Compute link center of mass (average of vertex positions)
                if (link_idx, env_idx) in link_vertex_positions:
                    verts = link_vertex_positions[(link_idx, env_idx)]
                    link_forces[(link_idx, env_idx)]["center"] = np.mean(verts, axis=0)

            # Force is negative gradient
            force = -force_grad
            link_forces[(link_idx, env_idx)]["force"] += force

            # Compute torque: τ = r × F
            if (link_idx, env_idx) in link_vertex_positions:
                contact_pos = link_vertex_positions[(link_idx, env_idx)][local_idx]
                center_pos = link_forces[(link_idx, env_idx)]["center"]
                r = contact_pos - center_pos
                torque = np.cross(r, force)
                link_forces[(link_idx, env_idx)]["torque"] += torque

        # Store forces in the proper format
        for (link_idx, env_idx), data in link_forces.items():
            if link_idx not in self._ipc_contact_forces:
                self._ipc_contact_forces[link_idx] = {}

            self._ipc_contact_forces[link_idx][env_idx] = {"force": data["force"], "torque": data["torque"]}

    def _apply_ipc_contact_forces(self):
        """
        Apply recorded IPC contact forces to Genesis rigid bodies.

        This method takes the contact forces and torques recorded by _record_ipc_contact_forces
        and applies them to the corresponding Genesis rigid links.
        """
        import torch

        if not self._ipc_contact_forces:
            return  # No contact forces to apply

        rigid_solver = self.rigid_solver

        for link_idx, env_data in self._ipc_contact_forces.items():
            for env_idx, force_data in env_data.items():
                force = force_data["force"]
                torque = force_data["torque"]

                # Convert numpy arrays to torch tensors
                force_tensor = torch.as_tensor(force, dtype=gs.tc_float, device=gs.device).unsqueeze(0)  # (1, 3)
                torque_tensor = torch.as_tensor(torque, dtype=gs.tc_float, device=gs.device).unsqueeze(0)  # (1, 3)

                # Prepare kwargs for apply methods
                apply_kwargs = {
                    "links_idx": link_idx,
                    "local": False,  # World frame
                    "zero_v": False,
                }

                # Add env_idx only if scene is parallelized
                if self.sim._scene.n_envs > 0:
                    apply_kwargs["envs_idx"] = env_idx

                # Apply force and torque to the link
                rigid_solver.apply_links_external_force(force=force_tensor, **apply_kwargs)
                rigid_solver.apply_links_external_torque(torque=torque_tensor, **apply_kwargs)
