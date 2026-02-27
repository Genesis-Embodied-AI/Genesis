import logging
import os
import tempfile
import weakref
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.repr_base import RBC
from genesis.utils.misc import tensor_to_array, qd_to_numpy, geometric_mean, harmonic_mean

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial
from genesis.engine.materials.rigid import Rigid
from genesis.options.solvers import IPCCouplerOptions

# Check if libuipc is available
try:
    import uipc

    UIPC_AVAILABLE = True
except ImportError:
    UIPC_AVAILABLE = False
    uipc = None

if UIPC_AVAILABLE:
    from uipc import Logger as UIPCLogger, Timer as UIPCTimer, Transform, Vector3, Quaternion, Vector12
    from uipc import builtin, view
    from uipc.core import Engine, World, Scene
    from uipc.core import AffineBodyStateAccessorFeature
    from uipc.constitution import (
        AffineBodyConstitution,
        AffineBodyExternalBodyForce,
        AffineBodyPrismaticJoint,
        AffineBodyRevoluteJoint,
        DiscreteShellBending,
        ElasticModuli,
        ElasticModuli2D,
        ExternalArticulationConstraint,
        NeoHookeanShell,
        SoftTransformConstraint,
        StableNeoHookean,
        StrainLimitingBaraffWitkinShell,
    )
    from uipc.geometry import (
        Geometry,
        SimplicialComplexSlot,
        affine_body,
        apply_transform,
        ground,
        label_surface,
        linemesh,
        merge,
        tetmesh,
        trimesh as uipc_trimesh,
    )
    from uipc.backend import SceneVisitor
    from uipc.unit import MPa

    ABD_KAPPA = 100.0 * MPa

from .data import ABDLinkEntry, ArticulatedEntityData, ForceBatch, IPCCouplingData, ArticulationData
from .utils import (
    find_target_link_for_fixed_merge,
    compute_link_to_link_transform,
    compute_link_init_world_rotation,
    is_robot_entity,
    build_ipc_scene_config,
    extract_articulated_joints,
    read_ipc_geometry_metadata,
    compute_coupling_forces,
)


def _animate_rigid_link(coupler_ref, env_idx, link_idx, info):
    """Animator callback for a soft-constraint coupled rigid link.

    Uses a weakref to the coupler to avoid preventing garbage collection.
    """
    coupler = coupler_ref()
    if coupler is None:
        gs.raise_exception("IPCCoupler was garbage collected while animator callback is still active.")

    geom_slots = info.geo_slots()
    if not geom_slots:
        return
    geom = geom_slots[0].geometry()

    # Read stored Genesis transform (q_genesis^n)
    # This was stored in _store_gs_rigid_states() before advance()
    transform_matrix = coupler._stored_rigid_transforms[link_idx][env_idx]
    if transform_matrix is None:
        return

    # Enable constraint and set target transform
    is_constrained_attr = geom.instances().find(builtin.is_constrained)
    aim_transform_attr = geom.instances().find(builtin.aim_transform)
    assert is_constrained_attr and aim_transform_attr
    view(is_constrained_attr)[0] = 1
    view(aim_transform_attr)[:] = transform_matrix


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
        self.rigid_solver = self.sim.rigid_solver
        self.fem_solver = self.sim.fem_solver

        # ==== IPC System Infrastructure ====
        self._ipc_engine = None
        self._ipc_world = None
        self._ipc_scene = None
        self._ipc_scene_subscenes = {}
        self._use_subscenes = False  # Set in _init_ipc based on number of environments

        # ==== IPC Constitutions (lazy-initialized on demand) ====
        self._ipc_abd = None  # AffineBodyDynamics component
        self._ipc_stk = None  # Contacts/impulses component
        self._ipc_stc = None  # SoftTransformConstraint
        self._ipc_nks = None  # NeoHookean / StableNeoHookean
        self._ipc_dsb = None  # DiscreteShellBending
        self._ipc_ext_force = None  # AffineBodyExternalBodyForce
        self._ipc_eac = None  # ExternalArticulationConstraint

        # ==== IPC Contact Elements ====
        self._ipc_no_collision_contact = None
        self._ipc_fem_contacts = {}  # i_e -> ContactElement
        self._ipc_cloth_contacts = {}  # i_e -> ContactElement
        self._ipc_abd_contacts = {}  # i_e -> ContactElement
        self._ipc_ground_contacts = {}  # i_g -> ContactElement

        # ==== Entity Coupling Configuration (keyed by i_e, solver-local) ====
        self._coupling_types = {}  # i_e -> coupling_type str
        self._coupling_link_filters = {}  # i_e -> set of link indices (two_way_soft_constraint only)
        self._coupling_collision_settings = {}  # i_e -> {i_l: bool}
        self._coupling_type_to_entities = None  # coupling_type -> [i_e, ...] (built lazily)

        # ==== IPC Mesh Tracking (per solver) ====
        self._rigid_env_objs = None
        self._rigid_env_meshes = None
        self._rigid_meshes_handle = None
        self._fem_env_objs = None
        self._fem_env_meshes = None
        self._fem_meshes_handle = None

        # ==== ABD Geometry & State ====
        self._abd_transforms = None  # ABD transform data
        self._abd_link_to_geo = {}  # (env_idx, i_l) -> abd_geometry
        self._abd_link_to_slot = {}  # (env_idx, i_l) -> abd_geometry_slot
        self._abd_primary_links = []  # [(env_idx, i_l), ...] excludes merged aliases
        self._abd_body_idx_to_link = {}  # abd_body_idx -> (env_idx, i_l, i_e)
        self._abd_state_feature = None  # AffineBodyStateAccessorFeature
        self._abd_state_geo = None  # Geometry for batch data transfer
        self._abd_data_by_link = {}  # (i_l, env_idx) -> ABDData

        # ==== Two-Way Coupling State ====
        self._stored_rigid_transforms = {}  # i_l -> {env_idx: transform_matrix}
        self._coupling_contact_forces = {}  # i_l -> {env_idx: ContactForceEntry}
        self.coupling_data = None  # IPCCouplingData (pre-allocated numpy buffers)

        # ==== Vertex Mapping ====
        self._vertex_to_link_mapping = {}  # global_vertex_idx -> (i_l, env_idx, local_vertex_idx)
        self._global_vertex_offset = 0  # Counter for vertex indices across geometries

        # ==== External Articulation ====
        self._articulation_metadata_initialized = False  # Flag for one-time articulation setup
        self._articulation_non_fixed_base_entities = []  # [i_e, ...] entities with non-fixed base
        self._articulation_entities = {}  # i_e -> ArticulatedEntityData
        self._articulation_stored_qpos = {}  # i_e -> np.ndarray (n_envs, n_qs)
        self.articulation_data = None  # ArticulationData (batched arrays, allocated lazily)

    def build(self) -> None:
        """Build IPC system"""
        # IPC coupler builds a single IPC scene shared across all envs, so it requires
        # identical geometry topology (links, joints, geoms) across environments.
        # Batched info options allow per-env topology which is incompatible.
        if self.rigid_solver.is_active:
            opts = self.rigid_solver._options
            if opts.batch_links_info or opts.batch_dofs_info or opts.batch_joints_info:
                gs.raise_exception(
                    "IPC coupler does not support batched rigid info "
                    "(batch_links_info, batch_dofs_info, batch_joints_info). "
                    "Please disable these options when using IPC coupling."
                )

        self._collect_coupling_config_from_materials()
        self._init_ipc()
        self._add_objects_to_ipc()
        self._finalize_ipc()

        # Categorize entities by coupling type (stable after build)
        self._categorize_coupling_type_to_entities()

        # Pre-compute the entity set for rigid state retrieval
        self._rigid_retrieve_entity_set = set(
            self._coupling_type_to_entities["two_way_soft_constraint"]
            + self._coupling_type_to_entities["ipc_only"]
            + self._articulation_non_fixed_base_entities
        )

        # Pre-allocate coupling data for the exact number of ABD bodies matching the entity set
        n_coupling_items = sum(
            i_e in self._rigid_retrieve_entity_set for _, _, i_e in self._abd_body_idx_to_link.values()
        )
        self.coupling_data = IPCCouplingData(n_coupling_items)

    def _init_ipc(self):
        """Initialize IPC system components"""

        # Derive IPC logging level from Genesis logger
        if gs.logger.level > logging.DEBUG:
            UIPCLogger.set_level(UIPCLogger.Level.Error)
            UIPCTimer.disable_all()

        # Create workspace directory for IPC output, named after scene UID.
        workspace = os.path.join(tempfile.gettempdir(), f"genesis_ipc_{self.sim.scene.uid.full()}")
        os.makedirs(workspace, exist_ok=True)

        # Note: gpu_device option may need to be set via CUDA environment variables (CUDA_VISIBLE_DEVICES)
        # before Genesis initialization, as libuipc Engine does not expose device selection in constructor
        self._ipc_engine = Engine("cuda", workspace)
        self._ipc_world = World(self._ipc_engine)

        # Create IPC scene with configuration
        config = build_ipc_scene_config(self.options, self.sim.options)
        self._ipc_scene = Scene(config)

        # Cache tabulars for reuse across methods
        self._ipc_constitution_tabular = self._ipc_scene.constitution_tabular()
        self._ipc_contact_tabular = self._ipc_scene.contact_tabular()
        self._ipc_subscene_tabular = self._ipc_scene.subscene_tabular()
        self._ipc_objects = self._ipc_scene.objects()
        self._ipc_animator = self._ipc_scene.animator()

        # Constitutions are created lazily in _add_*_to_ipc methods
        # based on entity types actually present in the scene.

        # Set up subscenes for multi-environment (scene grouping)
        # Only use subscenes when n_envs > 0 to isolate per-environment contacts
        self._ipc_scene_subscenes = {}
        self._use_subscenes = self.sim.n_envs > 0

        if self._use_subscenes:
            for i_b in range(self.sim._B):
                self._ipc_scene_subscenes[i_b] = self._ipc_subscene_tabular.create(f"subscene_{i_b}")
                for j_b in range(i_b):
                    self._ipc_subscene_tabular.insert(
                        self._ipc_scene_subscenes[j_b], self._ipc_scene_subscenes[i_b], False
                    )

    def _add_objects_to_ipc(self):
        """Add objects from solvers to IPC system"""
        # Per-entity and per-ground contact elements are created in _add_*_to_ipc
        # and registered in _register_contact_pairs.
        self._ipc_no_collision_contact = self._ipc_contact_tabular.create("no_collision_contact")

        # Add FEM entities to IPC
        if self.fem_solver.is_active:
            self._add_fem_entities_to_ipc()

        # Add rigid geoms to IPC based on per-entity coupling types
        if self.rigid_solver.is_active:
            self._add_rigid_geoms_to_ipc()

            # Add articulated entities for entities with external_articulation coupling type
            has_articulation_entities = any(ct == "external_articulation" for ct in self._coupling_types.values())
            if has_articulation_entities:
                self._add_articulation_entities_to_ipc()

        # Register all per-entity contact pair models with per-material friction
        self._register_contact_pairs()

    def _add_fem_entities_to_ipc(self):
        """Add FEM entities to the existing IPC scene (includes both volumetric FEM and cloth)"""

        # Create constitutions based on entity types present
        entities = self.fem_solver._entities
        has_cloth = any(isinstance(e.material, ClothMaterial) for e in entities)
        has_volumetric = any(not isinstance(e.material, ClothMaterial) for e in entities)

        if has_volumetric:
            self._ipc_stk = StableNeoHookean()
            self._ipc_constitution_tabular.insert(self._ipc_stk)
        if has_cloth:
            self._ipc_nks = StrainLimitingBaraffWitkinShell()
            self._ipc_constitution_tabular.insert(self._ipc_nks)
            self._ipc_dsb = DiscreteShellBending()
            self._ipc_constitution_tabular.insert(self._ipc_dsb)

        self._fem_meshes_handle = {}
        self._fem_env_objs = []
        self._fem_env_meshes = []

        for i_b in range(self.sim._B):
            self._fem_env_objs.append([])
            self._fem_env_meshes.append([])
            for i_e, entity in enumerate(self.fem_solver._entities):
                is_cloth = isinstance(entity.material, ClothMaterial)

                # Create object in IPC
                obj_name = f"cloth_{i_b}_{i_e}" if is_cloth else f"fem_{i_b}_{i_e}"
                self._fem_env_objs[i_b].append(self._ipc_objects.create(obj_name))

                # Create mesh: trimesh for cloth (2D shell), tetmesh for volumetric FEM (3D)
                if is_cloth:
                    # Cloth: use surface triangles only
                    verts = tensor_to_array(entity.init_positions).astype(np.float64, copy=False)
                    faces = entity.surface_triangles.astype(np.int32, copy=False)
                    mesh = uipc_trimesh(verts, faces)
                else:
                    # Volumetric FEM: use tetrahedral mesh
                    mesh = tetmesh(tensor_to_array(entity.init_positions), entity.elems)
                label_surface(mesh)

                self._fem_env_meshes[i_b].append(mesh)

                # Add to contact subscene (only for multi-environment)
                if self._use_subscenes:
                    self._ipc_scene_subscenes[i_b].apply_to(mesh)

                # Apply per-entity contact element (created once per entity on first env iteration)
                if is_cloth:
                    if i_e not in self._ipc_cloth_contacts:
                        self._ipc_cloth_contacts[i_e] = self._ipc_contact_tabular.create(f"cloth_contact_{i_e}")
                    self._ipc_cloth_contacts[i_e].apply_to(mesh)
                else:
                    if i_e not in self._ipc_fem_contacts:
                        self._ipc_fem_contacts[i_e] = self._ipc_contact_tabular.create(f"fem_contact_{i_e}")
                    self._ipc_fem_contacts[i_e].apply_to(mesh)

                # Apply material constitution based on type
                if is_cloth:
                    moduli = ElasticModuli2D.youngs_poisson(entity.material.E, entity.material.nu)
                    # Apply shell material for cloth
                    self._ipc_nks.apply_to(
                        mesh, moduli=moduli, mass_density=entity.material.rho, thickness=entity.material.thickness
                    )
                    # Apply bending stiffness if specified
                    if entity.material.bending_stiffness is not None:
                        self._ipc_dsb.apply_to(mesh, bending_stiffness=entity.material.bending_stiffness)
                else:
                    # Apply volumetric material for FEM
                    moduli = ElasticModuli.youngs_poisson(entity.material.E, entity.material.nu)
                    self._ipc_stk.apply_to(mesh, moduli, mass_density=entity.material.rho)

                # Add metadata to identify geometry type
                meta_attrs = mesh.meta()
                meta_attrs.create("solver_type", "cloth" if is_cloth else "fem")
                meta_attrs.create("env_idx", str(i_b))
                meta_attrs.create("entity_idx", str(i_e))

                # Create geometry in IPC scene
                self._fem_env_objs[i_b][i_e].geometries().create(mesh)
                self._fem_meshes_handle[f"fem_{i_b}_{i_e}"] = mesh

                # Update global vertex offset (FEM vertices occupy index space but aren't in mapping)
                self._global_vertex_offset += mesh.vertices().size()

    def _add_rigid_geoms_to_ipc(self):
        """Add rigid geoms to the IPC scene as ABD objects, merging geoms by link."""

        # Create constitutions based on coupling types present
        coupling_types_present = set(self._coupling_types.values())
        if coupling_types_present:
            self._ipc_abd = AffineBodyConstitution()
            self._ipc_constitution_tabular.insert(self._ipc_abd)
            self._ipc_ext_force = AffineBodyExternalBodyForce()
            self._ipc_constitution_tabular.insert(self._ipc_ext_force)

        has_soft_constraint = coupling_types_present - {"ipc_only"}
        if has_soft_constraint:
            self._ipc_stc = SoftTransformConstraint()
            self._ipc_constitution_tabular.insert(self._ipc_stc)

        self._rigid_env_objs = []
        self._rigid_env_meshes = []
        self._rigid_meshes_handle = {}
        self._abd_transforms = {}

        gs.logger.debug(f"Registered entity coupling types: {self._coupling_types}")

        # Pre-fetch geom info arrays (not batched: batch_*_info is rejected in build())
        geoms_pos = qd_to_numpy(self.rigid_solver.geoms_info.pos)
        geoms_quat = qd_to_numpy(self.rigid_solver.geoms_info.quat)
        geoms_type = qd_to_numpy(self.rigid_solver.geoms_info.type)
        geoms_data = qd_to_numpy(self.rigid_solver.geoms_info.data)
        geoms_vert_num = qd_to_numpy(self.rigid_solver.geoms_info.vert_num)
        geoms_vert_start = qd_to_numpy(self.rigid_solver.geoms_info.vert_start)
        geoms_vert_end = qd_to_numpy(self.rigid_solver.geoms_info.vert_end)
        geoms_face_start = qd_to_numpy(self.rigid_solver.geoms_info.face_start)
        geoms_face_end = qd_to_numpy(self.rigid_solver.geoms_info.face_end)
        verts_init_pos = qd_to_numpy(self.rigid_solver.verts_info.init_pos)
        faces_verts_idx = qd_to_numpy(self.rigid_solver.faces_info.verts_idx)
        links_entity_idx = qd_to_numpy(self.rigid_solver.links_info.entity_idx)

        # Pre-fetch link state (varies per env, batched when n_envs > 0)
        links_pos = qd_to_numpy(self.rigid_solver.links_state.pos)
        links_quat = qd_to_numpy(self.rigid_solver.links_state.quat)

        # ========== Pre-compute link groups (env-independent) ==========
        # Group links by fixed-joint merge target, matching mjcf.py behavior
        # where geoms from fixed-joint children are merged into the parent body's mesh.
        target_groups = {}  # target_link_idx -> [source_link_idx, ...]
        merge_transforms = {}  # source_link_idx -> (R, t) relative to target frame

        for link in self.rigid_solver.links:
            i_l = link.idx
            i_e = int(links_entity_idx[i_l])
            coupling_type = self._coupling_types.get(i_e)
            if coupling_type is None:
                continue

            # Link filter for two_way_soft_constraint
            if coupling_type == "two_way_soft_constraint" and i_e in self._coupling_link_filters:
                link_filter = self._coupling_link_filters[i_e]
                if link_filter is not None and i_l not in link_filter:
                    continue

            i_target_link = find_target_link_for_fixed_merge(self.rigid_solver, i_l)
            if i_target_link not in target_groups:
                target_groups[i_target_link] = []
            target_groups[i_target_link].append(i_l)

            if i_l != i_target_link:
                R, t = compute_link_to_link_transform(self.rigid_solver, i_l, i_target_link)
                merge_transforms[i_l] = (R, t)
                gs.logger.debug(
                    f"Merging link {i_l} ({self.rigid_solver.links[i_l].name}) "
                    f"into target link {i_target_link} ({self.rigid_solver.links[i_target_link].name}) via fixed joint"
                )

        # ========== Process each environment ==========
        for i_b in range(self.sim._B):
            self._rigid_env_objs.append([])
            self._rigid_env_meshes.append([])

            for i_target_link, source_links in target_groups.items():
                i_e = int(links_entity_idx[i_target_link])
                entity = self.rigid_solver._entities[i_e]
                entity_coupling_type = self._coupling_types[i_e]

                # Inner loop: iterate geoms of all source links
                meshes = []
                for i_l in source_links:
                    link = self.rigid_solver.links[i_l]
                    for i_g in range(link.geom_start, link.geom_end):
                        if geoms_type[i_g] == gs.GEOM_TYPE.PLANE:
                            if entity_coupling_type != "ipc_only":
                                gs.raise_exception(
                                    f"Plane entity (solver idx={i_e}) has coupling_mode='{entity_coupling_type}', "
                                    f"but only 'ipc_only' is supported for plane geoms."
                                )

                            local_normal = geoms_data[i_g][:3].astype(np.float64)
                            normal = gu.transform_by_quat(local_normal, geoms_quat[i_g])
                            normal = normal / np.linalg.norm(normal)
                            height = np.dot(geoms_pos[i_g], normal)
                            plane_geom = ground(height, normal)

                            if i_e not in self._ipc_ground_contacts:
                                plane_contact = self._ipc_contact_tabular.create(f"ground_contact_{i_e}")
                                self._ipc_ground_contacts[i_e] = plane_contact
                            self._ipc_ground_contacts[i_e].apply_to(plane_geom)

                            plane_obj = self._ipc_objects.create(f"rigid_plane_{i_b}_{i_g}")
                            self._rigid_env_objs[i_b].append(plane_obj)
                            self._rigid_env_meshes[i_b].append(None)

                            if self._use_subscenes:
                                self._ipc_scene_subscenes[i_b].apply_to(plane_geom)

                            plane_obj.geometries().create(plane_geom)
                            self._rigid_meshes_handle[f"rigid_plane_{i_b}_{i_g}"] = plane_geom

                        elif geoms_vert_num[i_g] > 0:
                            try:
                                vert_start, vert_end = int(geoms_vert_start[i_g]), int(geoms_vert_end[i_g])
                                face_start, face_end = int(geoms_face_start[i_g]), int(geoms_face_end[i_g])

                                geom_verts = verts_init_pos[vert_start:vert_end]
                                geom_faces = faces_verts_idx[face_start:face_end] - vert_start

                                # Apply geom transform to vertices
                                geom_rot_mat = gu.quat_to_R(geoms_quat[i_g])
                                transformed_verts = geom_verts @ geom_rot_mat.T + geoms_pos[i_g]

                                # Apply additional transform for fixed joint merging
                                if i_l != i_target_link:
                                    R, t = merge_transforms[i_l]
                                    transformed_verts = transformed_verts @ R.T + t

                                meshes.append(
                                    uipc_trimesh(
                                        transformed_verts.astype(np.float64, copy=False),
                                        geom_faces.astype(np.int32, copy=False),
                                    )
                                )
                            except (ValueError, RuntimeError, KeyError) as e:
                                gs.raise_exception_from(f"Failed to process geom {i_g} for IPC.", e)

                if not meshes:
                    continue

                # Merge meshes belonging to this target link
                merged_mesh = meshes[0] if len(meshes) == 1 else merge(meshes)
                label_surface(merged_mesh)

                # Apply target link world transform
                trans_view = view(merged_mesh.transforms())
                if self.sim.n_envs > 0:
                    trans_view[0] = gu.trans_quat_to_T(links_pos[i_target_link, i_b], links_quat[i_target_link, i_b])
                else:
                    trans_view[0] = gu.trans_quat_to_T(links_pos[i_target_link], links_quat[i_target_link])

                # Create rigid object
                rigid_obj = self._ipc_objects.create(f"rigid_link_{i_b}_{i_target_link}")
                self._rigid_env_objs[i_b].append(rigid_obj)
                self._rigid_env_meshes[i_b].append(merged_mesh)

                if self._use_subscenes:
                    self._ipc_scene_subscenes[i_b].apply_to(merged_mesh)

                # Apply per-entity contact element or no-collision marker
                collision_enabled = True
                if i_e in self._coupling_collision_settings:
                    if i_target_link in self._coupling_collision_settings[i_e]:
                        collision_enabled = self._coupling_collision_settings[i_e][i_target_link]

                if collision_enabled:
                    if i_e not in self._ipc_abd_contacts:
                        abd_contact = self._ipc_contact_tabular.create(f"abd_contact_{i_e}")
                        self._ipc_abd_contacts[i_e] = abd_contact
                    self._ipc_abd_contacts[i_e].apply_to(merged_mesh)
                else:
                    self._ipc_no_collision_contact.apply_to(merged_mesh)

                # Determine coupling behavior
                is_ipc_only = entity_coupling_type == "ipc_only"
                is_free_base = (
                    entity_coupling_type == "external_articulation"
                    and i_target_link == entity.base_link_idx
                    and not entity.links[0].is_fixed
                )
                is_soft_constraint_target = entity_coupling_type == "two_way_soft_constraint" or (
                    is_free_base and not self.options.free_base_driven_by_ipc
                )

                # Apply ABD constitution
                self._ipc_abd.apply_to(merged_mesh, kappa=ABD_KAPPA, mass_density=entity.material.rho)

                # Set external_kinetic: 1 for bodies driven by rigid solver, 0 for IPC-only and IPC-driven free base
                external_kinetic_attr = merged_mesh.instances().find(builtin.external_kinetic)
                if external_kinetic_attr is not None:
                    is_free_base_ipc_driven = is_free_base and self.options.free_base_driven_by_ipc
                    view(external_kinetic_attr)[:] = int(not is_free_base_ipc_driven and not is_ipc_only)

                # Set is_fixed for links that are fixed in Genesis
                is_fixed_attr = merged_mesh.instances().find(builtin.is_fixed)
                if is_fixed_attr is not None:
                    view(is_fixed_attr)[0] = int(self.rigid_solver.links[i_target_link].is_fixed)

                # For external_articulation, store reference DOF for articulation constraint sync
                if entity_coupling_type == "external_articulation" and self.options.enable_rigid_dofs_sync:
                    ref_dof_prev_attr = merged_mesh.instances().create("ref_dof_prev", Vector12.Zero())
                    view(ref_dof_prev_attr)[0] = affine_body.transform_to_q(trans_view[0])

                # Apply SoftTransformConstraint and animator for coupled links
                if is_soft_constraint_target:
                    constraint_strength = np.array(
                        [self.options.constraint_strength_translation, self.options.constraint_strength_rotation],
                        dtype=np.float64,
                    )
                    self._ipc_stc.apply_to(merged_mesh, constraint_strength)

                    self._ipc_animator.insert(
                        rigid_obj, partial(_animate_rigid_link, weakref.ref(self), i_b, i_target_link)
                    )

                # Apply external force (initially zero)
                self._ipc_ext_force.apply_to(merged_mesh, np.zeros(12, dtype=np.float64))

                # Metadata
                meta_attrs = merged_mesh.meta()
                meta_attrs.create("solver_type", "rigid")
                meta_attrs.create("env_idx", str(i_b))
                meta_attrs.create("link_idx", str(i_target_link))

                # Vertex-to-link mapping for two_way_soft_constraint contact force feedback
                if entity_coupling_type == "two_way_soft_constraint":
                    n_verts = merged_mesh.vertices().size()
                    for local_idx in range(n_verts):
                        self._vertex_to_link_mapping[self._global_vertex_offset + local_idx] = (
                            i_target_link,
                            i_b,
                            local_idx,
                        )

                self._global_vertex_offset += merged_mesh.vertices().size()

                abd_slot, _ = rigid_obj.geometries().create(merged_mesh)
                self._rigid_meshes_handle[f"rigid_link_{i_b}_{i_target_link}"] = merged_mesh
                self._abd_link_to_geo[(i_b, i_target_link)] = merged_mesh
                self._abd_link_to_slot[(i_b, i_target_link)] = abd_slot
                self._abd_primary_links.append((i_b, i_target_link))

                # Alias mappings for links merged via fixed joints
                for i_l in source_links:
                    if i_l != i_target_link:
                        self._abd_link_to_geo[(i_b, i_l)] = merged_mesh
                        self._abd_link_to_slot[(i_b, i_l)] = abd_slot
                        gs.logger.debug(
                            f"ABD slot alias: link {i_l} -> target link {i_target_link} (fixed joint merge)"
                        )

        # NOTE: Mass scaling removed - now using external_kinetic=1 instead
        # All mass is handled by IPC, Genesis uses external_kinetic for kinematic coupling

    def _register_contact_pairs(self):
        """Register pairwise contact models for all entity contact elements.

        Friction is combined by geometric mean, resistance by harmonic mean (series spring).
        When an entity material does not define
        ``contact_resistance``, ``options.contact_resistance`` is used as the per-entity fallback.
        Ground pairs combine entity parameters with the plane entity's material friction.
        """
        global_resistance = self.options.contact_resistance

        def resolve_entity_resistance(material_resistance):
            return global_resistance if material_resistance is None else material_resistance

        # Collect (ContactElement, friction_mu, resistance, is_abd) for all entity contact elements.
        all_elems = []
        for i_e, elem in (*self._ipc_cloth_contacts.items(), *self._ipc_fem_contacts.items()):
            mat = self.fem_solver._entities[i_e].material
            all_elems.append((elem, mat.friction_mu, resolve_entity_resistance(mat.contact_resistance), False))
        for i_e, elem in self._ipc_abd_contacts.items():
            mat = self.rigid_solver._entities[i_e].material
            all_elems.append((elem, mat.coup_friction, resolve_entity_resistance(mat.contact_resistance), True))

        # Register entity-entity pairs (upper triangle including self-pairs).
        for i, (elem_i, mu_i, res_i, is_abd_i) in enumerate(all_elems):
            for elem_j, mu_j, res_j, is_abd_j in all_elems[i:]:
                friction_ij = geometric_mean(mu_i, mu_j)
                resistance_ij = harmonic_mean(res_i, res_j)
                enabled = not (is_abd_i and is_abd_j) or self.options.enable_rigid_rigid_contact
                self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, enabled)

        # Register per-plane ground contact pairs.
        for i_e, ground_elem in self._ipc_ground_contacts.items():
            plane_mat = self.rigid_solver._entities[i_e].material
            plane_mu = plane_mat.coup_friction
            plane_res = resolve_entity_resistance(plane_mat.contact_resistance)
            for elem, mu, resistance, is_abd in all_elems:
                friction_ground = geometric_mean(mu, plane_mu)
                resistance_ground = harmonic_mean(resistance, plane_res)
                enabled = not is_abd or self.options.enable_rigid_ground_contact
                self._ipc_contact_tabular.insert(ground_elem, elem, friction_ground, resistance_ground, enabled)
            self._ipc_contact_tabular.insert(self._ipc_no_collision_contact, ground_elem, 0.0, global_resistance, False)

        # Register no_collision pairs (always disabled).
        for elem, _mu, _res, _is_abd in all_elems:
            self._ipc_contact_tabular.insert(self._ipc_no_collision_contact, elem, 0.0, global_resistance, False)
        self._ipc_contact_tabular.insert(
            self._ipc_no_collision_contact, self._ipc_no_collision_contact, 0.0, global_resistance, False
        )

    def _finalize_ipc(self):
        """Finalize IPC setup and initialize AffineBodyStateAccessorFeature"""
        self._ipc_world.init(self._ipc_scene)
        self._ipc_world.dump()
        gs.logger.info("IPC world initialized successfully")

        # Initialize AffineBodyStateAccessorFeature for optimized ABD state retrieval
        self._init_abd_state_accessor()

    def _init_abd_state_accessor(self):
        """
        Initialize AffineBodyStateAccessorFeature for efficient batch ABD state retrieval.

        This feature allows O(num_rigid_bodies) retrieval instead of O(total_geometries).
        Creates index mapping from ABD body index to (env_idx, link_idx, i_e) for
        fast lookups during runtime.
        """
        self._abd_state_feature = self._ipc_world.features().find(AffineBodyStateAccessorFeature)
        body_count = self._abd_state_feature.body_count()
        gs.logger.debug(f"AffineBodyStateAccessorFeature initialized with {body_count} ABD bodies")

        if body_count == 0:
            # No ABD bodies, feature not needed
            self._abd_state_feature = None
            return

        # Create state geometry for batch data transfer
        self._abd_state_geo = self._abd_state_feature.create_geometry()
        identity_matrix = np.eye(4, dtype=np.float64)
        self._abd_state_geo.instances().create(builtin.transform, identity_matrix)
        self._abd_state_geo.instances().create(builtin.velocity, identity_matrix)

        # Build index mapping: ABD body index -> (env_idx, link_idx, i_e)
        # The order matches the order ABD bodies were added to IPC scene
        # IMPORTANT: Only iterate over primary ABD bodies (not merged link aliases)
        self._abd_body_idx_to_link = {}
        abd_body_idx = 0

        # Only iterate over primary ABD bodies (excludes merged link aliases)
        for env_idx, link_idx in self._abd_primary_links:
            i_e = self.rigid_solver.links_info.entity_idx[link_idx]
            self._abd_body_idx_to_link[abd_body_idx] = (env_idx, link_idx, i_e)
            abd_body_idx += 1

        # Verify the count matches IPC's ABD body count
        if abd_body_idx != body_count:
            gs.raise_exception(
                f"ABD body count mismatch: expected {body_count}, got {abd_body_idx}. "
                f"This will cause indexing errors in AffineBodyStateAccessorFeature."
            )

        gs.logger.debug(
            f"ABD body index mapping created: {len(self._abd_body_idx_to_link)} entries. "
            f"Optimized state retrieval enabled (O(N) instead of O(M), N={body_count})."
        )

    # ============================================================
    # Section 1: Configuration API
    # ============================================================

    @property
    def is_active(self) -> bool:
        """Check if IPC coupling is active"""
        return self._ipc_world is not None

    def _collect_coupling_config_from_materials(self):
        """Read coupling_mode, coupling_link_filter, and collision settings from entity materials."""

        for i_e, entity in enumerate(self.rigid_solver.entities):
            coupling_mode = entity.material.coupling_mode
            if coupling_mode is None:
                continue

            self._coupling_types[i_e] = coupling_mode
            gs.logger.debug(f"Rigid entity (solver idx={i_e}): coupling mode '{coupling_mode}'")

            # Resolve link filter from material
            link_filter_names = entity.material.coupling_link_filter
            if link_filter_names is not None:
                target_links = set()
                for name in link_filter_names:
                    link = entity.get_link(name=name)
                    target_links.add(link.idx)
                self._coupling_link_filters[i_e] = target_links
                gs.logger.debug(f"Entity {i_e}: IPC link filter set to {len(target_links)} link(s)")

            # Resolve collision settings from material
            if not entity.material.enable_coupling_collision:
                collision_link_names = entity.material.coupling_collision_links
                if collision_link_names is not None:
                    collision_links = [entity.get_link(name=name) for name in collision_link_names]
                else:
                    collision_links = entity.links
                for link in collision_links:
                    self._coupling_collision_settings.setdefault(i_e, {})[link.idx] = False
                gs.logger.debug(
                    f"Entity {i_e}: IPC collision disabled for "
                    f"{len(self._coupling_collision_settings.get(i_e, {}))} link(s)"
                )

    @property
    def has_any_rigid_coupling(self) -> bool:
        """
        Check if any rigid entity is coupled to IPC.

        Returns
        -------
        bool
            True if at least one rigid entity has a coupling type (two_way_soft_constraint,
            external_articulation, or ipc_only).
        """
        return len(self._coupling_types) > 0

    # ============================================================
    # Section 2: Main Coupling Loop & Shared Helpers
    # ============================================================

    def _apply_forces_to_rigid_links(self, env_batches):
        """Apply batched forces/torques from *env_batches* to Genesis rigid links.

        Parameters
        ----------
        env_batches : dict[int, ForceBatch]
            Mapping from env_idx to ForceBatch.
        """
        for env_idx, batch in env_batches.items():
            if not batch.link_indices:
                continue
            envs_idx = env_idx if self.sim.n_envs > 0 else None
            self.rigid_solver.apply_links_external_force(
                force=batch.forces,
                links_idx=batch.link_indices,
                envs_idx=envs_idx,
                local=False,
            )
            self.rigid_solver.apply_links_external_torque(
                torque=batch.torques,
                links_idx=batch.link_indices,
                envs_idx=envs_idx,
                local=False,
            )

    def preprocess(self, f):
        """Preprocessing step before coupling"""
        pass

    def _store_gs_rigid_states(self):
        """
        Store current Genesis rigid body states before IPC advance.

        These stored states will be used by:
        1. Animator: to set aim_transform for IPC soft constraints
        2. Force computation: to ensure action-reaction force consistency
        3. User modification detection: to detect if user called set_qpos
        """
        if not self.rigid_solver.is_active:
            return

        # Clear previous stored states
        self._stored_rigid_transforms.clear()

        # Store qpos for all entities (used by external_articulation coupling)
        entities_qpos = qd_to_numpy(self.rigid_solver.qpos, copy=True)  # (n_total_qs, n_envs) or (n_total_qs,)
        if self.sim.n_envs == 0:
            entities_qpos = entities_qpos[:, np.newaxis]
        for i_e, entity in enumerate(self.rigid_solver._entities):
            if entity.n_qs > 0:
                q_start = entity._q_start
                n_qs = entity.n_qs
                self._articulation_stored_qpos[i_e] = entities_qpos[q_start : q_start + n_qs, :]

        # Store transforms for all rigid links
        links_pos = tensor_to_array(self.rigid_solver.get_links_pos())  # (n_envs, n_links, 3) or (n_links, 3)
        links_quat = tensor_to_array(self.rigid_solver.get_links_quat())  # (n_envs, n_links, 4) or (n_links, 4)
        # Ensure 3D: (n_envs, n_links, dim)
        if self.sim.n_envs == 0:
            links_pos = links_pos[np.newaxis]
            links_quat = links_quat[np.newaxis]

        n_links = links_pos.shape[1]

        # Iterate directly over primary ABD links instead of parsing string keys
        for env_idx, link_idx in self._abd_primary_links:
            if link_idx < n_links:
                link_transform = gu.trans_quat_to_T(links_pos[env_idx, link_idx], links_quat[env_idx, link_idx])
                self._stored_rigid_transforms.setdefault(link_idx, {})[env_idx] = link_transform

    def _categorize_coupling_type_to_entities(self):
        """Categorize entities by their coupling type. Called once during build()."""
        result = {"two_way_soft_constraint": [], "external_articulation": [], "ipc_only": []}
        for i_e, coupling_type in self._coupling_types.items():
            if coupling_type in result:
                result[coupling_type].append(i_e)
        self._coupling_type_to_entities = result

    def couple(self, f):
        """
        Execute IPC coupling step with per-entity coupling types.

        This unified coupling flow handles all entity types:
        - 'two_way_soft_constraint': Uses Animator + SoftTransformConstraint
        - 'external_articulation': Uses ExternalArticulationConstraint at joint level
        - 'ipc_only': One-way coupling, IPC controls rigid body transforms

        Flow:
        1. Store Genesis rigid states (common)
        2. Pre-advance processing (per entity type)
        3. IPC advance + retrieve (common, only once)
        4. Retrieve FEM states (common)
        5. Post-advance processing (per entity type)
        """
        if not self.is_active:
            return

        # Step 1: Store Genesis rigid states (common)
        self._store_gs_rigid_states()

        # Step 2: Pre-advance processing (per entity type)
        self._pre_advance_external_articulation()

        # Step 3: IPC advance + retrieve (common)
        self._ipc_world.advance()
        self._ipc_world.retrieve()

        # Step 4: Retrieve states
        self._retrieve_fem_states()
        self._retrieve_rigid_states()

        # Step 5: Post-advance processing (per entity type)
        self._apply_abd_coupling_forces()
        self._post_advance_external_articulation()
        self._post_advance_ipc_only()

    # ============================================================
    # Section 3: External Articulation Coupling
    # ============================================================

    def _pre_advance_external_articulation(self):
        """
        Pre-advance processing for external_articulation entities.
        Prepares articulation data and updates IPC geometry before advance().
        """
        if not self._coupling_type_to_entities["external_articulation"]:
            return

        if len(self._articulation_entities) == 0:
            return

        ad = self.articulation_data

        # Initialize ArticulationData on first call (exact-size allocation)
        if not self._articulation_metadata_initialized:
            n_entities = len(self._articulation_entities)
            n_envs = self.sim._B

            # Determine max dimensions across all entities
            max_dofs = 0
            max_joints = 0
            for idx, (i_e, art_data) in enumerate(self._articulation_entities.items()):
                entity = art_data.entity
                actual_qpos = tensor_to_array(entity.get_qpos(envs_idx=0 if n_envs > 1 else None))
                if actual_qpos.ndim > 1:
                    actual_qpos = actual_qpos[0]
                n_dofs_actual = len(actual_qpos)
                art_data.n_dofs_actual = n_dofs_actual
                max_dofs = max(max_dofs, n_dofs_actual)
                max_joints = max(max_joints, art_data.n_joints)

            # Allocate exact-size ArticulationData
            ad = ArticulationData(n_entities, max_dofs, max_joints, n_envs)
            self.articulation_data = ad

            for idx, (i_e, art_data) in enumerate(self._articulation_entities.items()):
                entity = art_data.entity
                n_joints = art_data.n_joints
                joint_qpos_indices = art_data.joint_qpos_indices
                joint_dof_indices = art_data.joint_dof_indices
                n_dofs_actual = art_data.n_dofs_actual

                ad.entity_indices[idx] = i_e
                ad.entity_n_dofs[idx] = n_dofs_actual
                ad.entity_n_joints[idx] = n_joints
                ad.entity_dof_start[idx] = entity.dof_start

                for j_idx in range(n_joints):
                    ad.joint_qpos_indices[idx, j_idx] = joint_qpos_indices[j_idx]
                    ad.joint_dof_indices[idx, j_idx] = joint_dof_indices[j_idx]

            self._articulation_metadata_initialized = True

        n_envs = self.sim._B
        dofs_mass_mat = qd_to_numpy(self.rigid_solver.mass_mat)

        # Copy stored qpos to articulation_data.qpos_current
        for idx in range(ad.n_entities):
            i_e = int(ad.entity_indices[idx])
            n_dofs = int(ad.entity_n_dofs[idx])
            if i_e in self._articulation_stored_qpos:
                stored = self._articulation_stored_qpos[i_e]  # (n_qs, n_envs)
                ad.qpos_current[idx, :n_envs, :n_dofs] = stored[:n_dofs, :n_envs].T

        # Compute delta_theta_tilde = qpos_current - ref_dof_prev (per joint)
        for idx in range(ad.n_entities):
            n_joints = int(ad.entity_n_joints[idx])
            for j in range(n_joints):
                qi = int(ad.joint_qpos_indices[idx, j])
                ad.delta_theta_tilde[idx, :n_envs, j] = (
                    ad.qpos_current[idx, :n_envs, qi] - ad.ref_dof_prev[idx, :n_envs, qi]
                )

        # Update IPC geometry for each articulated entity
        for idx, (i_e, art_data) in enumerate(self._articulation_entities.items()):
            for env_idx in range(self.sim._B):
                articulation_slot = art_data.articulation_slots_by_env[env_idx]
                articulation_geo = articulation_slot.geometry()

                # Update ref_dof_prev on all ABD instances
                if self.options.enable_rigid_dofs_sync:
                    for joint_idx, joint in enumerate(art_data.revolute_joints + art_data.prismatic_joints):
                        child_link_idx = joint.link.idx

                        abd_geo_slot = self._abd_link_to_slot.get((env_idx, child_link_idx))
                        if abd_geo_slot is None:
                            continue

                        abd_geo = abd_geo_slot.geometry()
                        ref_dof_prev_attr = abd_geo.instances().find("ref_dof_prev")
                        ref_dof_prev_view = view(ref_dof_prev_attr)

                        key = (idx, joint_idx, env_idx)
                        link_transform = ad.prev_link_transforms.get(key)
                        if link_transform is None and child_link_idx in self._stored_rigid_transforms:
                            link_transform = self._stored_rigid_transforms[child_link_idx].get(env_idx)

                        if link_transform is not None:
                            ref_dof_prev_view[0] = affine_body.transform_to_q(link_transform)

                # Set delta_theta_tilde to IPC geometry
                delta_theta_tilde_attr = articulation_geo["joint"].find("delta_theta_tilde")
                delta_theta_tilde_view = view(delta_theta_tilde_attr)
                for joint_idx in range(art_data.n_joints):
                    delta_theta_tilde_view[joint_idx] = ad.delta_theta_tilde[idx, env_idx, joint_idx]

                # Extract and transfer mass matrix from Genesis to IPC
                dof_start = int(ad.entity_dof_start[idx])
                env_mass_mat = dofs_mass_mat[:, :, env_idx] if self.sim.n_envs > 0 else dofs_mass_mat
                s = slice(dof_start, dof_start + n_joints)
                mass_flat = env_mass_mat[s, s].ravel()
                ad.mass_matrix[idx, : n_joints * n_joints] = mass_flat

                mass_attr = articulation_geo["joint_joint"].find("mass")
                if mass_attr is not None:
                    mass_view = view(mass_attr)
                    for i in range(n_joints * n_joints):
                        mass_view[i] = mass_flat[i]

    def _post_advance_external_articulation(self):
        """
        Post-advance processing for external_articulation entities.
        Reads delta_theta from IPC and updates Genesis qpos.
        """
        if not self._coupling_type_to_entities["external_articulation"]:
            return

        if len(self._articulation_entities) == 0:
            return

        ad = self.articulation_data
        n_envs = self.sim._B

        # Read delta_theta_ipc from IPC
        for idx, art_data in enumerate(self._articulation_entities.values()):
            for env_idx in range(self.sim._B):
                scene_art_geo = art_data.articulation_slots_by_env[env_idx].geometry()
                delta_theta_attr = scene_art_geo["joint"].find("delta_theta")
                delta_theta_view = view(delta_theta_attr)
                for joint_idx in range(art_data.n_joints):
                    ad.delta_theta_ipc[idx, env_idx, joint_idx] = delta_theta_view[joint_idx]

        # Compute qpos_new: copy ref_dof_prev then scatter joint deltas
        for idx in range(ad.n_entities):
            n_dofs = ad.entity_n_dofs[idx]
            n_joints = ad.entity_n_joints[idx]
            ad.qpos_new[idx, :n_envs, :n_dofs] = ad.ref_dof_prev[idx, :n_envs, :n_dofs]
            for j in range(n_joints):
                qi = ad.joint_qpos_indices[idx, j]
                if qi < n_dofs:
                    ad.qpos_new[idx, :n_envs, qi] = (
                        ad.ref_dof_prev[idx, :n_envs, qi] + ad.delta_theta_ipc[idx, :n_envs, j]
                    )

        # Write qpos_new back to Genesis
        for idx, art_data in enumerate(self._articulation_entities.values()):
            entity = art_data.entity
            n_dofs = ad.entity_n_dofs[idx]

            # Set qpos for all DOFs
            envs_qpos = np.empty((self.sim._B, n_dofs), dtype=gs.np_float)
            for env_idx in range(self.sim._B):
                # Note: For non-fixed base robots, 'qpos_new' preserves base pose from 'ref_dof_prev' as only joint
                # DOFs were updated by the above computation.
                envs_qpos[env_idx] = ad.qpos_new[idx, env_idx, :n_dofs]

                # For non-fixed base robots, apply base link transform  from IPC
                if not art_data.has_free_base:
                    continue

                # Get IPC transform for base link from _abd_data_by_link
                abd_entry = self._abd_data_by_link[(art_data.base_link_idx, env_idx)]
                envs_qpos[env_idx, :3], envs_qpos[env_idx, 3:7] = gu.T_to_trans_quat(abd_entry.transform)

            self.rigid_solver.set_qpos(
                envs_qpos if self.sim.n_envs > 0 else envs_qpos[0],
                qs_idx=slice(entity.q_start, entity.q_end),
                skip_forward=False,
            )

            # Set base link velocities from IPC if available
            if art_data.has_free_base:
                self._apply_base_link_velocity_from_ipc(entity)

        # Update ref_dof_prev for next timestep
        for idx in range(ad.n_entities):
            n_dofs = int(ad.entity_n_dofs[idx])
            ad.ref_dof_prev[idx, :n_envs, :n_dofs] = ad.qpos_new[idx, :n_envs, :n_dofs]

        # Store current link transforms to prev_link_transforms
        for idx, art_data in enumerate(self._articulation_entities.values()):
            for env_idx in range(self.sim._B):
                for joint_idx, joint in enumerate(art_data.revolute_joints + art_data.prismatic_joints):
                    child_link_idx = joint.link.idx
                    if (
                        child_link_idx in self._stored_rigid_transforms
                        and env_idx in self._stored_rigid_transforms[child_link_idx]
                    ):
                        ad.prev_link_transforms[(idx, joint_idx, env_idx)] = self._stored_rigid_transforms[
                            child_link_idx
                        ][env_idx].copy()

    def _apply_base_link_velocity_from_ipc(self, entity):
        envs_vel = np.empty((self.sim._B, 6), dtype=gs.np_float)
        for env_idx in range(self.sim._B):
            abd_entry = self._abd_data_by_link[(entity.base_link_idx, env_idx)]
            envs_vel[env_idx, :3] = abd_entry.velocity[:3, 3]
            # omega_skew = dR/dt @ R^T
            omega_skew = abd_entry.velocity[:3, :3] @ abd_entry.transform[:3, :3].T
            envs_vel[env_idx, 3:] = (
                (omega_skew[2, 1] - omega_skew[1, 2]) / 2.0,
                (omega_skew[0, 2] - omega_skew[2, 0]) / 2.0,
                (omega_skew[1, 0] - omega_skew[0, 1]) / 2.0,
            )

        self.rigid_solver.set_dofs_velocity(
            envs_vel if self.sim.n_envs > 0 else envs_vel[0],
            dofs_idx=slice(entity.dof_start, entity.dof_start + 6),
            skip_forward=True,
        )

    # ============================================================
    # Section 4: IPC-Only Coupling
    # ============================================================

    def _post_advance_ipc_only(self):
        """
        Post-advance processing for 'ipc_only' entities.

        This method directly sets Genesis transforms from IPC results. It only handles rigid objects.
        """
        entity_indices = self._coupling_type_to_entities["ipc_only"]
        if not entity_indices:
            return

        envs_qpos = np.empty((self.sim._B, 7), dtype=gs.np_float)
        for i_e in entity_indices:
            entity = self.rigid_solver._entities[i_e]
            if entity.base_link.is_fixed:
                continue

            for env_idx in range(self.sim._B):
                abd_entry = self._abd_data_by_link[(entity.base_link_idx, env_idx)]
                envs_qpos[env_idx, :3], envs_qpos[env_idx, 3:7] = gu.T_to_trans_quat(abd_entry.transform)

            self.rigid_solver.set_qpos(
                envs_qpos if self.sim.n_envs > 0 else envs_qpos[0],
                qs_idx=slice(entity.q_start, entity.q_start + 7),
                skip_forward=True,
            )

            # FIXME: It is currently necessary to enforce zero velocity to avoid double time integration by Rigid solver
            # self._apply_base_link_velocity_from_ipc(entity)
            self.rigid_solver.set_dofs_velocity(
                velocity=None,
                dofs_idx=slice(entity.dof_start, entity.dof_start + 6),
                skip_forward=True,
            )

    # ============================================================
    # Section 5: FEM State Retrieval
    # ============================================================

    def _retrieve_fem_states(self):
        # IPC world advance/retrieve is handled at Scene level
        # This method handles both volumetric FEM (3D) and cloth (2D) post-processing

        if not self.fem_solver.is_active:
            return

        # Gather FEM states (both volumetric and cloth) using metadata filtering
        visitor = SceneVisitor(self._ipc_scene)

        # Collect FEM and cloth geometries using metadata
        fem_geo_by_entity = {}
        for geo_slot in visitor.geometries():
            if not isinstance(geo_slot, SimplicialComplexSlot):
                continue

            geo = geo_slot.geometry()
            if geo.dim() not in (2, 3):
                continue
            meta = read_ipc_geometry_metadata(geo)
            if meta is None:
                continue
            solver_type, env_idx, i_e = meta
            if solver_type not in ("fem", "cloth"):
                continue

            try:
                proc_geo = geo
                if geo.instances().size() >= 1:
                    proc_geo = merge(apply_transform(geo))
                pos = proc_geo.positions().view().reshape(-1, 3)
                fem_geo_by_entity.setdefault(i_e, {})[env_idx] = pos
            except (ValueError, RuntimeError) as e:
                gs.raise_exception_from(f"Failed to process FEM geometry for entity {i_e}, env {env_idx}.", e)

        # Update FEM entities using filtered geometries
        for i_e, env_positions in fem_geo_by_entity.items():
            if i_e < len(self.fem_solver._entities):
                entity = self.fem_solver._entities[i_e]
                per_env_positions = []

                for env_idx in range(self.sim._B):
                    if env_idx in env_positions:
                        per_env_positions.append(env_positions[env_idx])
                    else:
                        # Fallback for missing environment
                        per_env_positions.append(np.zeros((0, 3)))

                if per_env_positions:
                    envs_pos = np.stack(per_env_positions, axis=0, dtype=gs.np_float)
                    entity.set_pos(0, envs_pos)

    def _retrieve_rigid_states(self):
        """
        Retrieve ABD transforms/affine matrices after IPC step using AffineBodyStateAccessorFeature.

        O(num_rigid_bodies) instead of O(total_geometries).
        Also populates coupling_data arrays for force computation.
        """
        self._abd_data_by_link.clear()

        if self._ipc_scene is None:
            return

        if self._abd_state_feature is None or self._abd_state_geo is None:
            return

        # Single batch copy of ALL ABD states from IPC
        self._abd_state_feature.copy_to(self._abd_state_geo)

        # Get all transforms at once (array view)
        trans_attr = self._abd_state_geo.instances().find(builtin.transform)
        if trans_attr is None:
            return

        transforms = trans_attr.view()  # Shape: (num_bodies, 4, 4)

        # Get velocities (4x4 matrix representing transform derivative)
        vel_attr = self._abd_state_geo.instances().find(builtin.velocity)
        velocities = vel_attr.view() if vel_attr is not None else None  # Shape: (num_bodies, 4, 4)

        links_inertia = qd_to_numpy(self.rigid_solver.links_info.inertial_i)

        i_coupling = 0
        for abd_body_idx, (env_idx, link_idx, i_e) in self._abd_body_idx_to_link.items():
            if i_e not in self._rigid_retrieve_entity_set:
                continue

            aim_transform = self._stored_rigid_transforms[link_idx][env_idx]
            transform_matrix = transforms[abd_body_idx]

            self._abd_data_by_link[(link_idx, env_idx)] = ABDLinkEntry(
                transform=transform_matrix.copy(),
                aim_transform=aim_transform,
                velocity=velocities[abd_body_idx].copy(),
            )

            # Write directly to pre-allocated coupling_data buffers
            self.coupling_data.link_indices[i_coupling] = link_idx
            self.coupling_data.env_indices[i_coupling] = env_idx
            self.coupling_data.ipc_transforms[i_coupling] = transform_matrix
            self.coupling_data.aim_transforms[i_coupling] = aim_transform
            self.coupling_data.link_masses[i_coupling] = self.rigid_solver.links_info.inertial_mass[link_idx]
            self.coupling_data.inertia_tensors[i_coupling] = links_inertia[link_idx]
            i_coupling += 1

    def _apply_abd_coupling_forces(self):
        """
        Apply coupling forces from IPC ABD constraint to Genesis rigid bodies.

        Data has already been populated in coupling_data by _retrieve_rigid_states,
        so this function computes forces and applies the results.

        This ensures action-reaction force consistency:
        - IPC constraint force: G_ipc = M * (q_ipc^{n+1} - q_genesis^n)
        - Genesis reaction force: F_genesis = M * (q_ipc^{n+1} - q_genesis^n) = G_ipc
        """
        if not self._coupling_type_to_entities["two_way_soft_constraint"]:
            return

        if not self.options.two_way_coupling:
            return

        cd = self.coupling_data

        if len(cd.link_indices) == 0:
            return

        cd.out_forces, cd.out_torques = compute_coupling_forces(
            cd.ipc_transforms,
            cd.aim_transforms,
            cd.link_masses,
            cd.inertia_tensors,
            self.options.constraint_strength_translation,
            self.options.constraint_strength_rotation,
            self.sim._dt**2,
        )

        if np.isnan(cd.out_forces).any() or np.isnan(cd.out_torques).any():
            gs.raise_exception(
                "NaN detected in IPC coupling force/torque. "
                "This indicates numerical instability — consider decreasing the simulation timestep.",
            )

        # Group by environment and apply
        env_batches = {}
        for i in range(len(cd.link_indices)):
            env_idx = cd.env_indices[i]
            if env_idx not in env_batches:
                env_batches[env_idx] = ForceBatch()
            env_batches[env_idx].link_indices.append(cd.link_indices[i])
            env_batches[env_idx].forces.append(cd.out_forces[i])
            env_batches[env_idx].torques.append(cd.out_torques[i])

        self._apply_forces_to_rigid_links(env_batches)

    def couple_grad(self, f):
        """Gradient computation for coupling"""
        # IPC doesn't support gradients yet
        pass

    def reset(self, envs_idx=None):
        """Reset coupling state"""
        gs.logger.debug("Resetting IPC coupler state")
        self._ipc_world.recover(0)
        self._ipc_world.retrieve()

    # ============================================================
    # Section 6: Articulation IPC Setup & Geometry Lookup
    # ============================================================

    def _create_joint_geometry(
        self,
        joint,
        jtype,
        constitution,
        reverse_verts,
        i_e,
        i_b,
        joints_xanchor,
    ):
        """Create a single joint IPC geometry (revolute or prismatic) and return its slot."""
        child_link_idx = joint.link.idx
        parent_link_idx_original = joint.link.parent_idx if joint.link.parent_idx >= 0 else 0
        parent_link_idx = find_target_link_for_fixed_merge(self.rigid_solver, parent_link_idx_original)

        try:
            parent_abd_slot = self._abd_link_to_slot[(i_b, parent_link_idx)]
            child_abd_slot = self._abd_link_to_slot[(i_b, child_link_idx)]
        except KeyError as e:
            gs.raise_exception_from(
                f"Failed to build external_articulation in multi-env mode: "
                f"entity {i_e}, env {i_b}, joint '{joint.name}', "
                f"missing ABD slot(s) for parent={parent_link_idx}, child={child_link_idx}.",
                e,
            )

        joint_axis_local = joint.dofs_motion_ang[0] if jtype == "revolute" else joint.dofs_motion_vel[0]
        child_link = self.rigid_solver.links[child_link_idx]
        parent_link = self.rigid_solver.links[parent_link_idx]

        gs.logger.debug(f"\n--- Processing {jtype} joint: {joint.name} (env {i_b}) ---")
        gs.logger.debug(f"  Parent link: {parent_link_idx} ({parent_link.name})")
        gs.logger.debug(f"  Child link: {child_link_idx} ({child_link.name})")
        gs.logger.debug(f"  Joint axis (joint frame): {joint_axis_local}")

        child_rot_matrix = compute_link_init_world_rotation(self.rigid_solver, child_link_idx)
        joint_axis = child_rot_matrix @ joint_axis_local

        gs.logger.debug(f"  Child link rotation matrix:\n{child_rot_matrix}")
        gs.logger.debug(f"  Joint axis (world): {joint_axis}")

        joint_pos = joints_xanchor[joint.idx, i_b]
        gs.logger.debug(f"  Joint position (world): {joint_pos}")

        v1 = joint_pos - 0.5 * joint_axis
        v2 = joint_pos + 0.5 * joint_axis
        vertices = np.array([v2, v1] if reverse_verts else [v1, v2], dtype=np.float64)
        edges = np.array([[0, 1]], dtype=np.int32)
        mesh = linemesh(vertices, edges)

        if self._use_subscenes:
            self._ipc_scene_subscenes[i_b].apply_to(mesh)

        constitution.apply_to(mesh, [parent_abd_slot], [0], [child_abd_slot], [0], [100.0])

        joint_obj = self._ipc_objects.create(f"{jtype}_joint_{i_e}_{i_b}_{joint.name}")
        slot, _ = joint_obj.geometries().create(mesh)
        return slot

    def _add_articulation_entities_to_ipc(self):
        """
        Add articulated robot entities to IPC using ExternalArticulationConstraint.
        This enables joint-level coupling between Genesis and IPC.
        """
        self._ipc_eac = ExternalArticulationConstraint()
        self._ipc_constitution_tabular.insert(self._ipc_eac)

        # Initialize cache list for articulation entities with non-fixed base
        # (used by couple() to call _retrieve_rigid_states for base link transforms)

        # Process each rigid entity with external_articulation coupling type
        for i_e in range(len(self.rigid_solver._entities)):
            # Only process entities with external_articulation coupling type
            entity_coupling_type = self._coupling_types.get(i_e)
            if entity_coupling_type != "external_articulation":
                continue

            entity = self.rigid_solver._entities[i_e]

            # Extract joints from the entity
            joint_info = extract_articulated_joints(entity)

            # Skip entities without joints
            if joint_info["n_joints"] == 0:
                continue

            # Detect non-fixed base (for handling base link separately via SoftTransformConstraint)
            base_link = entity.links[0]
            has_free_base = not base_link.is_fixed
            base_link_idx = entity.base_link_idx

            gs.logger.debug(
                f"Adding articulated entity {i_e} with {joint_info['n_joints']} joints "
                f"({len(joint_info['revolute_joints'])} revolute, {len(joint_info['prismatic_joints'])} prismatic)"
            )

            # Create joint geometries and slots for libuipc
            joints_xanchor = qd_to_numpy(self.rigid_solver.joints_state.xanchor)
            abrj = AffineBodyRevoluteJoint()
            abpj = AffineBodyPrismaticJoint()
            n_joints = joint_info["n_joints"]
            joint_geo_slots_by_env = {}
            articulation_geos_by_env = {}
            articulation_slots_by_env = {}
            articulation_objects_by_env = {}
            mass_matrix = np.eye(n_joints, dtype=np.float64) * 1e4  # Default stiffness

            # Build one EA geometry set per environment.
            for i_b in range(self.sim._B):
                joint_geo_slots = []

                # Add revolute and prismatic joints (unified loop)
                joint_type_spec = [
                    ("revolute", joint_info["revolute_joints"], abrj, True),
                    ("prismatic", joint_info["prismatic_joints"], abpj, False),
                ]
                for jtype, joints, constitution, reverse_verts in joint_type_spec:
                    for joint in joints:
                        slot = self._create_joint_geometry(
                            joint,
                            jtype,
                            constitution,
                            reverse_verts,
                            i_e,
                            i_b,
                            joints_xanchor,
                        )
                        joint_geo_slots.append(slot)

                if len(joint_geo_slots) != n_joints:
                    raise RuntimeError(
                        "Failed to build external_articulation in multi-env mode: "
                        f"entity {i_e}, env {i_b}, expected {n_joints} joint slots, got {len(joint_geo_slots)}."
                    )

                indices = [0] * len(joint_geo_slots)
                articulation_geo = self._ipc_eac.create_geometry(joint_geo_slots, indices)

                if self._use_subscenes:
                    self._ipc_scene_subscenes[i_b].apply_to(articulation_geo)

                mass_attr = articulation_geo["joint_joint"].find("mass")
                mass_view = view(mass_attr)
                mass_view[:] = mass_matrix.ravel()  # Symmetric, so order doesn't matter

                articulation_object = self._ipc_objects.create(f"articulation_entity_{i_e}_{i_b}")
                articulation_slot, _ = articulation_object.geometries().create(articulation_geo)

                joint_geo_slots_by_env[i_b] = joint_geo_slots
                articulation_geos_by_env[i_b] = articulation_geo
                articulation_slots_by_env[i_b] = articulation_slot
                articulation_objects_by_env[i_b] = articulation_object

            # Store articulation data
            self._articulation_entities[i_e] = ArticulatedEntityData(
                entity=entity,
                revolute_joints=joint_info["revolute_joints"],
                prismatic_joints=joint_info["prismatic_joints"],
                joint_geo_slots_by_env=joint_geo_slots_by_env,
                articulation_geos_by_env=articulation_geos_by_env,
                articulation_slots_by_env=articulation_slots_by_env,
                articulation_objects_by_env=articulation_objects_by_env,
                n_joints=n_joints,
                ref_dof_prev=np.zeros(entity.n_dofs, dtype=np.float64),
                delta_theta_tilde=np.zeros(n_joints, dtype=np.float64),
                delta_theta=np.zeros(n_joints, dtype=np.float64),
                joint_qpos_indices=joint_info["joint_qpos_indices"],
                joint_dof_indices=joint_info["joint_dof_indices"],
                mass_matrix=mass_matrix,
                has_free_base=has_free_base,
                base_link_idx=base_link_idx,
            )

            # Add to cache list if non-fixed base (for _retrieve_rigid_states in couple())
            if has_free_base:
                self._articulation_non_fixed_base_entities.append(i_e)

            gs.logger.debug(f"Successfully added articulated entity {i_e} to IPC")
