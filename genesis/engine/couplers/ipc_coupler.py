import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import trimesh
import quadrants as ti

import genesis as gs
from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial
from genesis.options.solvers import IPCCouplerOptions
from genesis.repr_base import RBC
from genesis.utils import mesh as mu
from genesis.utils import geom as gu


if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

# Import IPC Solver if available
try:
    import uipc

    UIPC_AVAILABLE = True
except ImportError:
    UIPC_AVAILABLE = False

try:
    import polyscope as ps

    POLYSCOPE_AVAILABLE = True
except ImportError:
    POLYSCOPE_AVAILABLE = False


if UIPC_AVAILABLE:
    from uipc import view, builtin, Transform, Vector3, Quaternion, Logger, Timer
    from uipc.backend import SceneVisitor
    from uipc.constitution import (
        AffineBodyConstitution,
        SoftTransformConstraint,
        StableNeoHookean,
        NeoHookeanShell,
        DiscreteShellBending,
        ElasticModuli,
        ElasticModuli2D,
    )
    from uipc.core import Engine, World, Scene
    from uipc.geometry import (
        SimplicialComplexSlot,
        apply_transform,
        trimesh as uipc_trimesh,
        tetmesh,
        label_surface,
        label_triangle_orient,
        flip_inward_triangles,
        merge,
        ground,
    )
    from uipc.unit import MPa

if POLYSCOPE_AVAILABLE:
    from uipc.gui import SceneGUI


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
            gs.raise_exception(
                "Python module 'uipc' but not found. Please build and install libuipc from source following the "
                "official instructions: https://spirimirror.github.io/libuipc-doc/build_install/"
            )

        self.sim = simulator
        self.options = options

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
        # Disable IPC logging if requested
        if self.options.disable_ipc_logging:
            Logger.set_level(Logger.Level.Error)
            Timer.disable_all()

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
                    mesh = uipc_trimesh(verts, faces)
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
                if is_cloth:
                    # Apply shell material for cloth
                    moduli = ElasticModuli2D.youngs_poisson(entity.material.E, entity.material.nu)
                    nks.apply_to(
                        mesh, moduli=moduli, mass_density=entity.material.rho, thickness=entity.material.thickness
                    )
                    # Apply bending stiffness if specified
                    if entity.material.bending_stiffness is not None:
                        dsb.apply_to(mesh, bending_stiffness=entity.material.bending_stiffness)
                else:
                    # Apply volumetric material for FEM
                    moduli = ElasticModuli.youngs_poisson(entity.material.E, entity.material.nu)
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

                        # Convert trimesh to tetmesh
                        try:
                            tri_mesh = trimesh.Trimesh(vertices=transformed_verts, faces=geom_faces)
                            verts, elems = mu.tetrahedralize_mesh(tri_mesh, tet_cfg=dict())
                            rigid_mesh = tetmesh(verts.astype(np.float64), elems.astype(np.int32))

                            # Store mesh and geom info
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
                            # Multiple meshes in link - merge them
                            meshes_to_merge = [mesh for geom_idx, mesh in link_data["meshes"]]
                            merged_mesh = merge(meshes_to_merge)
                            geom_idx = link_data["meshes"][0][0]  # Use first geom's index for metadata

                        # Apply link world transform
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
                        label_triangle_orient(merged_mesh)
                        merged_mesh = flip_inward_triangles(merged_mesh)

                        # Create rigid object
                        rigid_obj = scene.objects().create(f"rigid_link_{i_b}_{link_idx}")
                        rigid_solver.list_env_obj[i_b].append(rigid_obj)
                        rigid_solver.list_env_mesh[i_b].append(merged_mesh)

                        # Add to contact subscene and apply ABD constitution (only for multi-environment)
                        if self._use_subscenes:
                            scene_subscenes[i_b].apply_to(merged_mesh)
                        self._ipc_abd_contact.apply_to(merged_mesh)

                        # Use half density for IPC ABD to avoid double-counting mass
                        # (the other half is in Genesis rigid solver, scaled in _scale_genesis_rigid_link_masses)
                        entity_rho = rigid_solver._entities[link_data["entity_idx"]].material.rho
                        abd.apply_to(merged_mesh, kappa=10.0 * MPa, mass_density=entity_rho / 2.0)

                        # Apply soft transform constraints
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

                        def create_animate_function(env_idx, link_idx, rigid_solver_ref):
                            def animate_rigid_link(info):
                                geo_slots = info.geo_slots()
                                if len(geo_slots) == 0:
                                    return
                                geo = geo_slots[0].geometry()

                                try:
                                    # Read stored Genesis transform (q_genesis^n)
                                    # This was stored in _store_genesis_rigid_states() before advance()
                                    if hasattr(rigid_solver_ref, "_genesis_stored_states"):
                                        stored_states = rigid_solver_ref._genesis_stored_states
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

                        animate_func = create_animate_function(i_b, link_idx, rigid_solver)
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
                f"  Link {link_idx}: mass {original_mass:.6f} -> {original_mass / 2.0:.6f} kg, inertia scaled by 0.5"
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

    def set_ipc_link_filter(self, entity, link_names=None, link_indices=None):
        """
        Set which links of an entity should participate in IPC simulation.

        Parameters
        ----------
        entity : RigidEntity
            The rigid entity to set the filter for
        link_names : list of str, optional
            Names of links to include in IPC. If None and link_indices is None, all links participate.
        link_indices : list of int, optional
            Local indices of links to include in IPC. If None and link_names is None, all links participate.
        """
        if link_names is None and link_indices is None:
            # Remove filter for this entity (all links participate)
            if entity._idx in self._ipc_link_filters:
                del self._ipc_link_filters[entity._idx]
            return

        link_filter = set()

        if link_names is not None:
            # Convert link names to solver-level indices
            for name in link_names:
                link = entity.get_link(name=name)
                if link is not None:
                    # Use solver-level index
                    link_filter.add(link.idx)
                else:
                    gs.logger.warning(f"Link name '{name}' not found in entity")

        if link_indices is not None:
            # Convert local link indices to solver-level indices
            for local_idx in link_indices:
                solver_link_idx = local_idx + entity._link_start
                link_filter.add(solver_link_idx)

        # Store filter for this entity
        self._ipc_link_filters[entity._idx] = link_filter

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

        # Initialize storage if not exists
        if not hasattr(rigid_solver, "_genesis_stored_states"):
            rigid_solver._genesis_stored_states = {}

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

                        if link_idx not in rigid_solver._genesis_stored_states:
                            rigid_solver._genesis_stored_states[link_idx] = {}
                        rigid_solver._genesis_stored_states[link_idx][env_idx] = genesis_transform

    def couple(self, f):
        """Execute IPC coupling step"""
        if not self.is_active:
            return

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

    def _retrieve_fem_states(self, f):
        # IPC world advance/retrieve is handled at Scene level
        # This method handles both volumetric FEM (3D) and cloth (2D) post-processing
        if not self.fem_solver.is_active:
            return

        # Gather FEM states (both volumetric and cloth) using metadata filtering
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
                            except Exception:
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

        rigid_solver = self.rigid_solver
        visitor = SceneVisitor(self._ipc_scene)

        # Collect ABD geometries and their constraint data using metadata
        abd_data_by_link = {}  # link_idx -> {env_idx: {transform, gradient, mass}}

        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                if geo.dim() == 3:
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
                            except Exception:
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
                                        hasattr(rigid_solver, "_genesis_stored_states")
                                        and link_idx in rigid_solver._genesis_stored_states
                                        and env_idx in rigid_solver._genesis_stored_states[link_idx]
                                    ):
                                        aim_transform = rigid_solver._genesis_stored_states[link_idx][env_idx]

                                    abd_data_by_link[link_idx][env_idx] = {
                                        "transform": transform_matrix,  # q_ipc^{n+1}
                                        "aim_transform": aim_transform,  # q_genesis^n
                                    }

                    except Exception as e:
                        gs.logger.warning(f"Failed to retrieve ABD geometry data: {e}")
                        continue

        # Store transforms for later access
        rigid_solver._abd_affines = abd_data_by_link

        # Apply coupling forces from IPC ABD to Genesis rigid bodies (two-way coupling)
        # Based on soft_transform_constraint.cu gradient computation
        if self.options.two_way_coupling:
            self._apply_abd_coupling_forces(abd_data_by_link)

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

    def _apply_abd_coupling_forces(self, abd_data_by_link):
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
        rigid_solver = self.rigid_solver
        strength_tuple = self.options.ipc_constraint_strength
        translation_strength = strength_tuple[0]
        rotation_strength = strength_tuple[1]

        dt = self.sim._dt
        dt2 = dt * dt

        for link_idx, env_data in abd_data_by_link.items():
            for env_idx, data in env_data.items():
                ipc_transform = data.get("transform")  # Current transform after IPC solve
                aim_transform = data.get("aim_transform")  # Target from Genesis

                if rigid_solver.n_envs == 0:
                    assert env_idx == 0
                    env_idx = None

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
                    rotvec = gu.R_to_rotvec(R_rel)

                    inertia_tensor_local = rigid_solver.links_info.inertial_i[link_idx].to_numpy()

                    # I_world = R_current * I_local * R_current^T
                    inertia_tensor_world = R_current @ inertia_tensor_local @ R_current.T

                    angular_torque = rotation_strength * inertia_tensor_world @ rotvec / dt2

                    # Apply forces to Genesis rigid body
                    rigid_solver.apply_links_external_force(force=linear_force, links_idx=link_idx, envs_idx=env_idx)
                    rigid_solver.apply_links_external_torque(
                        torque=angular_torque, links_idx=link_idx, envs_idx=env_idx
                    )

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
        if not POLYSCOPE_AVAILABLE:
            gs.raise_exception("Polyscope module is not installed. Please install it with `pip install polyscope`.")

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

    def update_ipc_gui(self):
        """Update IPC GUI"""
        ps.frame_tick()  # Non-blocking frame update
        self._ipc_scene_gui.update()
