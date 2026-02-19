import logging as _logging
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial
from genesis.options.solvers import IPCCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.array_class import V, V_VEC, V_MAT
from genesis.utils.misc import qd_to_numpy

from .ipc_array_class import (
    ArticulationState,
    IPCCouplingData,
    build_ipc_scene_config,
    build_link_transform_matrix,
    categorize_entities_by_coupling_type,
    compute_coupling_forces_kernel,
    compute_external_force_kernel,
    compute_link_contact_forces_kernel,
    compute_link_init_world_rotation,
    compute_link_to_link_transform,
    decompose_transform_matrix,
    extract_articulated_joints,
    find_target_link_for_fixed_merge,
    get_articulation_state,
    get_ipc_coupling_data,
    is_robot_entity,
    read_ipc_geometry_metadata,
)

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

try:
    import uipc

    UIPC_AVAILABLE = True
except ImportError:
    UIPC_AVAILABLE = False

if UIPC_AVAILABLE:
    from uipc import Logger as UIPCLogger, Quaternion, Transform, Vector3, builtin, view
    from uipc.backend import SceneVisitor
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
    from uipc.core import AffineBodyStateAccessorFeature, Engine, Scene, World
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


class IPCCoupler(RBC):
    """
    Coupler integrating the IPC (Incremental Potential Contact) solver via libuipc.

    Manages the communication between Genesis solvers and IPC, handling rigid bodies
    (as Affine Body Dynamics objects), FEM bodies, and cloth in a unified contact framework.

    Per-entity coupling mode is configured at entity-creation time via
    ``coupling_mode`` on ``gs.materials.Rigid``. Valid modes are:

    - ``None``: entity is ignored by IPC (default).
    - ``'two_way_soft_constraint'``: bidirectional soft-constraint coupling.
    - ``'external_articulation'``: joint-level coupling for articulated robots.
    - ``'ipc_only'``: IPC drives the entity; Genesis reads the resulting transforms.
    """

    def __init__(self, simulator: "Simulator", options: IPCCouplerOptions) -> None:
        if not UIPC_AVAILABLE:
            raise ImportError(
                "libuipc is required for IPC coupling but not found.\nPlease install via: pip install pyuipc"
            )

        self.sim = simulator
        self.options = options

        self.rigid_solver = self.sim.rigid_solver
        self.fem_solver = self.sim.fem_solver

        # IPC system components — initialized in build()
        self._ipc_engine = None
        self._ipc_world = None
        self._ipc_scene = None
        self._ipc_abd = None
        self._ipc_stk = None
        self._ipc_nks = None
        self._ipc_dsb = None
        self._ipc_eac = None
        self._ipc_ext_force = None
        self._ipc_stc = None
        self._ipc_animator = None
        self._ipc_scene_subscenes = {}
        self._use_subscenes = False

        # Contact elements (one per material category)
        self._ipc_abd_contact = None
        self._ipc_fem_contact = None
        self._ipc_cloth_contact = None
        self._ipc_ground_contact = None
        self._ipc_no_collision_contact = None

        # Per-entity coupling mode: entity_idx -> coupling_mode string
        # Populated from entity.material.coupling_mode at build() time.
        self._entity_coupling_types = {}

        # Per-entity link filter: entity_idx -> set of link_idx (None = all links)
        # Populated from entity.material.coupling_link_filter at build() time.
        self._ipc_link_filters = {}

        # Genesis rigid body states stored before each IPC advance step.
        # Maps link_idx -> {env_idx: 4×4 transform matrix}
        self._genesis_stored_states = {}

        # IPC ABD state retrieval (optimized path via AffineBodyStateAccessorFeature)
        self._abd_state_feature = None
        self._abd_state_geo = None
        # Maps ABD body index (from IPC) -> (env_idx, link_idx, entity_idx)
        self._abd_body_idx_to_link = {}

        # Primary ABD links (excludes merged fixed-joint aliases).
        # Ordered list of (env_idx, link_idx) in the order bodies were added to IPC.
        self._primary_abd_links = []

        # Link -> ABD geometry slot mapping: {(env_idx, link_idx): slot}
        self._link_to_abd_slot = {}

        # Articulated entities participating in joint-level coupling.
        # Maps entity_idx -> metadata dict (populated in _add_articulated_entities_to_ipc).
        self._articulated_entities = {}
        # Entities with non-fixed base (their base link also needs IPC-state retrieval).
        self._articulation_with_non_fixed_base = []

        # Link collision overrides: {entity_idx: {link_idx: bool}}
        self._link_collision_settings = {}

        # Previous link transforms for per-ABD ref_dof_prev synchronization.
        # Maps (entity_idx, joint_idx, env_idx) -> 4×4 transform matrix.
        self._prev_link_transforms = {}

        # Contact proxy state — only used when options.enable_contact_proxy is True.
        # Maps IPC global vertex index -> (link_idx, env_idx, local_vertex_idx).
        # Built during _add_rigid_geoms_to_ipc for two_way coupled links.
        self._vertex_to_link_mapping = {}
        # Running IPC global vertex index counter, updated during build.
        self._global_vertex_offset = 0
        # Per-step storage for contact forces: link_idx -> {env_idx: {'force', 'torque'}}
        self._ipc_contact_forces = {}

        # Outputs — available after each couple() call.
        # Maps link_idx -> {env_idx: {'transform': 4×4, 'aim_transform': 4×4, 'velocity': 4×4}}
        self.abd_data_by_link = {}

        # Exact-sized buffers — allocated at build() time after scene is populated.
        self.coupling_data: IPCCouplingData | None = None
        self.articulation_state: ArticulationState | None = None

        # Pre-allocated Quadrants fields for contact force pipeline.
        # Sizes are set in build() once the number of coupled links is known.
        self._contact_forces_ti = None
        self._contact_torques_ti = None
        self._abd_transforms_ti = None
        self._out_forces_ti = None
        self._link_indices_ti = None
        self._env_indices_ti = None
        self._link_contact_forces_out = None
        self._link_contact_torques_out = None
        self._vertex_force_gradients = None
        self._vertex_link_indices = None
        self._vertex_env_indices = None
        self._vertex_positions_world = None
        self._vertex_link_centers = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Set up the IPC scene, register all entities, and allocate runtime buffers."""
        # Populate coupling configuration from entity materials (before _init_ipc so
        # _entity_coupling_types is available when adding rigid geoms).
        self._collect_coupling_config_from_materials()

        self._init_ipc()
        self._add_objects_to_ipc()
        self._finalize_ipc()

    def _collect_coupling_config_from_materials(self):
        """Read coupling_mode and coupling_link_filter from each Rigid entity's material."""
        if not self.rigid_solver.is_active:
            return

        for entity in self.rigid_solver._entities:
            material = entity.material
            coupling_mode = getattr(material, "coupling_mode", None)
            if coupling_mode is None:
                continue

            entity_idx = entity._idx_in_solver

            if coupling_mode == "ipc_only" and entity.n_links != 1:
                gs.raise_exception(
                    f"'ipc_only' coupling mode requires a single-link entity. "
                    f"Entity {entity_idx} has {entity.n_links} links."
                )

            self._entity_coupling_types[entity_idx] = coupling_mode

            coupling_link_filter = getattr(material, "coupling_link_filter", None)
            if coupling_link_filter is not None:
                # Convert tuple of names to set of solver-level link indices.
                link_name_to_idx = {link.name: link.idx for link in entity.links}
                filter_set = set()
                for name in coupling_link_filter:
                    if name in link_name_to_idx:
                        filter_set.add(link_name_to_idx[name])
                    else:
                        gs.logger.warning(f"IPC link filter: link '{name}' not found in entity {entity_idx}.")
                self._ipc_link_filters[entity_idx] = filter_set

    def _init_ipc(self):
        """Create IPC engine, world, scene, constitutions, and contact model."""
        # Configure uipc logging level from Genesis logger
        if gs.logger.level >= _logging.ERROR:
            UIPCLogger.set_level(UIPCLogger.Level.Error)
        elif gs.logger.level >= _logging.WARNING:
            UIPCLogger.set_level(UIPCLogger.Level.Warn)
        elif gs.logger.level >= _logging.INFO:
            UIPCLogger.set_level(UIPCLogger.Level.Warn)
        elif gs.logger.level >= _logging.DEBUG:
            UIPCLogger.set_level(UIPCLogger.Level.Info)

        workspace = os.path.join(tempfile.gettempdir(), f"genesis_ipc_workspace_{os.getpid()}")
        os.makedirs(workspace, exist_ok=True)

        self._ipc_engine = Engine("cuda", workspace)
        self._ipc_world = World(self._ipc_engine)

        config = build_ipc_scene_config(self.options, self.sim.options)
        self._ipc_scene = Scene(config)

        self._ipc_abd = AffineBodyConstitution()
        self._ipc_stk = StableNeoHookean()
        self._ipc_nks = StrainLimitingBaraffWitkinShell()
        self._ipc_dsb = DiscreteShellBending()
        self._ipc_eac = ExternalArticulationConstraint()

        self._ipc_scene.constitution_tabular().insert(self._ipc_abd)
        self._ipc_scene.constitution_tabular().insert(self._ipc_stk)
        self._ipc_scene.constitution_tabular().insert(self._ipc_eac)

        # Compute representative friction values from materials.
        rigid_friction = self._compute_representative_rigid_friction()
        fem_friction = self._compute_representative_fem_friction()

        self._ipc_scene.contact_tabular().default_model(rigid_friction, self.options.contact_resistance)

        self._ipc_abd_contact = self._ipc_scene.contact_tabular().create("abd_contact")
        self._ipc_fem_contact = self._ipc_scene.contact_tabular().create("fem_contact")
        self._ipc_cloth_contact = self._ipc_scene.contact_tabular().create("cloth_contact")
        self._ipc_ground_contact = self._ipc_scene.contact_tabular().create("ground_contact")
        self._ipc_no_collision_contact = self._ipc_scene.contact_tabular().create("no_collision_contact")

        res = self.options.contact_resistance

        # FEM-FEM
        self._ipc_scene.contact_tabular().insert(self._ipc_fem_contact, self._ipc_fem_contact, fem_friction, res, True)
        # FEM-ABD
        self._ipc_scene.contact_tabular().insert(
            self._ipc_fem_contact, self._ipc_abd_contact, rigid_friction, res, True
        )
        # ABD-ABD (controlled by rigid solver's self-collision setting)
        self._ipc_scene.contact_tabular().insert(
            self._ipc_abd_contact, self._ipc_abd_contact, rigid_friction, res, self.rigid_solver._enable_self_collision
        )
        # Cloth-Cloth (controlled by option)
        self._ipc_scene.contact_tabular().insert(
            self._ipc_cloth_contact, self._ipc_cloth_contact, fem_friction, res, self.options.enable_cloth_self_contact
        )
        # Cloth-FEM
        self._ipc_scene.contact_tabular().insert(
            self._ipc_cloth_contact, self._ipc_fem_contact, fem_friction, res, True
        )
        # Cloth-ABD
        self._ipc_scene.contact_tabular().insert(
            self._ipc_cloth_contact, self._ipc_abd_contact, rigid_friction, res, True
        )
        # Ground-ABD
        self._ipc_scene.contact_tabular().insert(
            self._ipc_ground_contact, self._ipc_abd_contact, rigid_friction, res, self.options.enable_ground_contact
        )
        # Ground-FEM
        self._ipc_scene.contact_tabular().insert(
            self._ipc_ground_contact, self._ipc_fem_contact, rigid_friction, res, True
        )
        # Ground-Cloth
        self._ipc_scene.contact_tabular().insert(
            self._ipc_ground_contact, self._ipc_cloth_contact, rigid_friction, res, True
        )
        # no_collision vs everything
        for other in [
            self._ipc_abd_contact,
            self._ipc_fem_contact,
            self._ipc_cloth_contact,
            self._ipc_ground_contact,
            self._ipc_no_collision_contact,
        ]:
            self._ipc_scene.contact_tabular().insert(self._ipc_no_collision_contact, other, rigid_friction, res, False)

        # Subscenes for multi-environment isolation.
        B = self.sim._B
        self._use_subscenes = B > 1
        if self._use_subscenes:
            for i in range(B):
                self._ipc_scene_subscenes[i] = self._ipc_scene.subscene_tabular().create(f"subscene{i}")
            for i in range(B):
                for j in range(B):
                    if i != j:
                        self._ipc_scene.subscene_tabular().insert(
                            self._ipc_scene_subscenes[i], self._ipc_scene_subscenes[j], False
                        )

        self.abd_data_by_link = {}

    def _compute_representative_rigid_friction(self):
        """Return a representative friction value from registered rigid entities (default 0.5)."""
        frictions = []
        for entity in self.rigid_solver._entities:
            f = getattr(entity.material, "friction", None)
            if f is not None:
                frictions.append(float(f))
        return float(np.mean(frictions)) if frictions else 0.5

    def _compute_representative_fem_friction(self):
        """Return a representative friction value from registered FEM entities (default 0.5)."""
        frictions = []
        for entity in self.fem_solver._entities:
            f = getattr(entity.material, "friction_mu", None)
            if f is not None:
                frictions.append(float(f))
        return float(np.mean(frictions)) if frictions else 0.5

    def _add_objects_to_ipc(self):
        """Add all solver entities to the IPC scene."""
        if self.fem_solver.is_active:
            self._add_fem_entities_to_ipc()

        if self.rigid_solver.is_active:
            self._add_rigid_geoms_to_ipc()

        has_articulation = any(ct == "external_articulation" for ct in self._entity_coupling_types.values())
        if has_articulation and self.rigid_solver.is_active:
            self._add_articulated_entities_to_ipc()

    def _add_fem_entities_to_ipc(self):
        """Register FEM and cloth entities in the IPC scene."""
        fem_solver = self.fem_solver
        scene = self._ipc_scene

        fem_solver._mesh_handles = {}
        fem_solver.list_env_obj = []
        fem_solver.list_env_mesh = []

        for i_b in range(self.sim._B):
            fem_solver.list_env_obj.append([])
            fem_solver.list_env_mesh.append([])
            for i_e, entity in enumerate(fem_solver._entities):
                is_cloth = isinstance(entity.material, ClothMaterial)

                obj_name = f"cloth_{i_b}_{i_e}" if is_cloth else f"fem_{i_b}_{i_e}"
                fem_solver.list_env_obj[i_b].append(scene.objects().create(obj_name))

                if is_cloth:
                    verts = entity.init_positions.cpu().numpy().astype(np.float64)
                    faces = entity.surface_triangles.astype(np.int32)
                    mesh = uipc_trimesh(verts, faces)
                else:
                    mesh = tetmesh(entity.init_positions.cpu().numpy(), entity.elems)

                fem_solver.list_env_mesh[i_b].append(mesh)

                if self._use_subscenes:
                    self._ipc_scene_subscenes[i_b].apply_to(mesh)

                if is_cloth:
                    self._ipc_cloth_contact.apply_to(mesh)
                else:
                    self._ipc_fem_contact.apply_to(mesh)

                label_surface(mesh)

                if is_cloth:
                    moduli = ElasticModuli2D.youngs_poisson(entity.material.E, entity.material.nu)
                    self._ipc_nks.apply_to(
                        mesh, moduli=moduli, mass_density=entity.material.rho, thickness=entity.material.thickness
                    )
                    if entity.material.bending_stiffness is not None:
                        self._ipc_dsb.apply_to(mesh, bending_stiffness=entity.material.bending_stiffness)
                else:
                    moduli = ElasticModuli.youngs_poisson(entity.material.E, entity.material.nu)
                    self._ipc_stk.apply_to(mesh, moduli, mass_density=entity.material.rho)

                meta_attrs = mesh.meta()
                meta_attrs.create("solver_type", "cloth" if is_cloth else "fem")
                meta_attrs.create("env_idx", str(i_b))
                meta_attrs.create("entity_idx", str(i_e))

                fem_solver.list_env_obj[i_b][i_e].geometries().create(mesh)
                fem_solver._mesh_handles[f"gs_ipc_{i_b}_{i_e}"] = mesh

                # FEM vertices occupy IPC global index space before rigid vertices.
                self._global_vertex_offset += mesh.vertices().size()

    def _add_rigid_geoms_to_ipc(self):
        """Register rigid geoms in the IPC scene as ABD objects, merging fixed-joint children."""
        rigid_solver = self.rigid_solver
        scene = self._ipc_scene

        if not hasattr(self, "_ipc_ext_force") or self._ipc_ext_force is None:
            self._ipc_ext_force = AffineBodyExternalBodyForce()
            scene.constitution_tabular().insert(self._ipc_ext_force)

        rigid_solver.list_env_obj = []
        rigid_solver.list_env_mesh = []
        rigid_solver._mesh_handles = {}
        rigid_solver._abd_transforms = {}

        for i_b in range(self.sim._B):
            rigid_solver.list_env_obj.append([])
            rigid_solver.list_env_mesh.append([])

            # --- Planes ---
            for i_g in range(rigid_solver.n_geoms_):
                geom_type = rigid_solver.geoms_info.type[i_g]
                if geom_type != gs.GEOM_TYPE.PLANE:
                    continue
                pos = rigid_solver.geoms_info.pos[i_g].to_numpy()
                normal = np.array([0.0, 0.0, 1.0])
                height = float(np.dot(pos, normal))
                plane_geom = ground(height, normal)
                plane_obj = scene.objects().create(f"rigid_plane_{i_b}_{i_g}")
                rigid_solver.list_env_obj[i_b].append(plane_obj)
                rigid_solver.list_env_mesh[i_b].append(None)
                self._ipc_ground_contact.apply_to(plane_geom)
                plane_obj.geometries().create(plane_geom)
                rigid_solver._mesh_handles[f"rigid_plane_{i_b}_{i_g}"] = plane_geom

            # --- Non-plane geoms: group by merged target link ---
            # link_geoms: target_link_idx -> {meshes, link_world_pos, link_world_quat, entity_idx, original_to_target}
            link_geoms = {}

            for i_g in range(rigid_solver.n_geoms_):
                geom_type = rigid_solver.geoms_info.type[i_g]
                if geom_type == gs.GEOM_TYPE.PLANE:
                    continue

                link_idx = rigid_solver.geoms_info.link_idx[i_g]
                entity_idx = rigid_solver.links_info.entity_idx[link_idx]
                coupling_type = self._entity_coupling_types.get(entity_idx)
                if coupling_type is None:
                    continue

                # Apply link filter for two_way mode.
                if coupling_type == "two_way_soft_constraint" and entity_idx in self._ipc_link_filters:
                    link_filter = self._ipc_link_filters[entity_idx]
                    if link_filter is not None and link_idx not in link_filter:
                        continue

                target_link_idx = find_target_link_for_fixed_merge(rigid_solver, link_idx)

                if target_link_idx not in link_geoms:
                    link_geoms[target_link_idx] = {
                        "meshes": [],
                        "link_world_pos": None,
                        "link_world_quat": None,
                        "entity_idx": entity_idx,
                        "original_to_target": {},
                    }

                if link_idx != target_link_idx and link_idx not in link_geoms[target_link_idx]["original_to_target"]:
                    R_rel, t_rel = compute_link_to_link_transform(rigid_solver, link_idx, target_link_idx)
                    link_geoms[target_link_idx]["original_to_target"][link_idx] = (R_rel, t_rel)

                vert_num = rigid_solver.geoms_info.vert_num[i_g]
                if vert_num == 0:
                    continue

                try:
                    vert_start = rigid_solver.geoms_info.vert_start[i_g]
                    vert_end = rigid_solver.geoms_info.vert_end[i_g]
                    face_start = rigid_solver.geoms_info.face_start[i_g]
                    face_end = rigid_solver.geoms_info.face_end[i_g]

                    geom_verts = rigid_solver.verts_info.init_pos.to_numpy()[vert_start:vert_end]
                    geom_faces = rigid_solver.faces_info.verts_idx.to_numpy()[face_start:face_end]
                    geom_faces = geom_faces - vert_start

                    geom_rel_pos = rigid_solver.geoms_info.pos[i_g].to_numpy()
                    geom_rel_quat = rigid_solver.geoms_info.quat[i_g].to_numpy()
                    geom_rot_mat = gu.quat_to_R(geom_rel_quat)
                    transformed_verts = geom_verts @ geom_rot_mat.T + geom_rel_pos

                    if link_idx != target_link_idx:
                        R_rel, t_rel = link_geoms[target_link_idx]["original_to_target"][link_idx]
                        transformed_verts = transformed_verts @ R_rel.T + t_rel

                    rigid_mesh = uipc_trimesh(transformed_verts.astype(np.float64), geom_faces.astype(np.int32))
                    link_geoms[target_link_idx]["meshes"].append((i_g, rigid_mesh))

                    if link_geoms[target_link_idx]["link_world_pos"] is None:
                        link_geoms[target_link_idx]["link_world_pos"] = rigid_solver.links_state.pos[
                            target_link_idx, i_b
                        ]
                        link_geoms[target_link_idx]["link_world_quat"] = rigid_solver.links_state.quat[
                            target_link_idx, i_b
                        ]
                except Exception as e:
                    gs.logger.warning(f"Failed to process geom {i_g}: {e}")

            # Second pass: merge geoms per link and create IPC ABD objects.
            for target_link_idx, link_data in link_geoms.items():
                if not link_data["meshes"]:
                    continue

                try:
                    if len(link_data["meshes"]) == 1:
                        _, merged_mesh = link_data["meshes"][0]
                    else:
                        meshes_to_merge = [m for _, m in link_data["meshes"]]
                        merged_mesh = merge(meshes_to_merge)

                    label_surface(merged_mesh)

                    link_world_pos = link_data["link_world_pos"]
                    link_world_quat = link_data["link_world_quat"]

                    if link_world_pos is not None:
                        trans_view = view(merged_mesh.transforms())
                        t = Transform.Identity()
                        t.translate(
                            Vector3.Values(
                                (float(link_world_pos[0]), float(link_world_pos[1]), float(link_world_pos[2]))
                            )
                        )
                        uipc_quat = Quaternion(link_world_quat)
                        t.rotate(uipc_quat)
                        trans_view[0] = t.matrix()

                    entity_idx = link_data["entity_idx"]
                    # Contact element: per-link collision setting override if set.
                    entity_collision_settings = self._link_collision_settings.get(entity_idx, {})
                    link_collision_enabled = entity_collision_settings.get(target_link_idx, True)
                    contact_elem = self._ipc_abd_contact if link_collision_enabled else self._ipc_no_collision_contact
                    contact_elem.apply_to(merged_mesh)

                    if self._use_subscenes:
                        self._ipc_scene_subscenes[i_b].apply_to(merged_mesh)

                    # Apply ABD constitution with material mass/inertia.
                    link_mass = float(rigid_solver.links_info.inertial_mass[target_link_idx])
                    self._ipc_abd.apply_to(merged_mesh, mass_density=link_mass)

                    # Apply external force constitution.
                    self._ipc_ext_force.apply_to(merged_mesh)

                    # Set is_fixed to mark fixed links as static in IPC.
                    link = rigid_solver.links[target_link_idx]
                    is_fixed_attr = merged_mesh.instances().find(builtin.is_fixed)
                    if is_fixed_attr is not None:
                        view(is_fixed_attr)[0] = 1 if link.is_fixed else 0

                    # Determine whether this link's base is free and driven by IPC.
                    coupling_type = self._entity_coupling_types.get(entity_idx)
                    entity_obj = rigid_solver._entities[entity_idx]
                    is_free_base = (
                        coupling_type == "external_articulation"
                        and not entity_obj.links[0].is_fixed
                        and target_link_idx == entity_obj.base_link_idx
                    )
                    ipc_driven_base = is_free_base and self.options.free_base_driven_by_ipc

                    # Mark ABD bodies as externally driven (Genesis controls their motion),
                    # except for free bases explicitly driven by IPC physics.
                    external_kinetic_attr = merged_mesh.instances().find(builtin.external_kinetic)
                    if external_kinetic_attr is not None:
                        view(external_kinetic_attr)[:] = 0 if ipc_driven_base else 1

                    # SoftTransformConstraint: pulls IPC ABD body toward Genesis target transform.
                    # Required for two_way links and the non-fixed base of external_articulation entities
                    # (unless free_base_driven_by_ipc is True).
                    needs_stc = coupling_type == "two_way_soft_constraint" or (is_free_base and not ipc_driven_base)
                    if needs_stc:
                        if self._ipc_stc is None:
                            self._ipc_stc = SoftTransformConstraint()
                            scene.constitution_tabular().insert(self._ipc_stc)
                        self._ipc_stc.apply_to(
                            merged_mesh,
                            np.array(
                                [
                                    self.options.constraint_strength_translation,
                                    self.options.constraint_strength_rotation,
                                ]
                            ),
                        )

                    # Add metadata for geometry identification.
                    meta_attrs = merged_mesh.meta()
                    meta_attrs.create("solver_type", "rigid")
                    meta_attrs.create("env_idx", str(i_b))
                    meta_attrs.create("link_idx", str(target_link_idx))

                    obj_name = f"rigid_link_{i_b}_{target_link_idx}"
                    link_obj = scene.objects().create(obj_name)
                    rigid_solver.list_env_obj[i_b].append(link_obj)
                    rigid_solver.list_env_mesh[i_b].append(merged_mesh)
                    geo_slot, _ = link_obj.geometries().create(merged_mesh)
                    rigid_solver._mesh_handles[obj_name] = merged_mesh
                    self._link_to_abd_slot[(i_b, target_link_idx)] = geo_slot

                    # IPC Animator: feeds Genesis stored states to IPC as aim_transform each step.
                    if needs_stc:
                        if self._ipc_animator is None:
                            self._ipc_animator = scene.animator()

                        def _make_animate_cb(env_i, link_i, coupler_ref):
                            def _animate(info):
                                geo_slots = info.geo_slots()
                                if not geo_slots:
                                    return
                                geo = geo_slots[0].geometry()
                                stored = coupler_ref._genesis_stored_states
                                if link_i not in stored or env_i not in stored[link_i]:
                                    return
                                transform_matrix = stored[link_i][env_i]
                                is_constrained_attr = geo.instances().find(builtin.is_constrained)
                                aim_transform_attr = geo.instances().find(builtin.aim_transform)
                                if is_constrained_attr is not None and aim_transform_attr is not None:
                                    view(is_constrained_attr)[0] = 1
                                    view(aim_transform_attr)[:] = transform_matrix

                            return _animate

                        self._ipc_animator.insert(link_obj, _make_animate_cb(i_b, target_link_idx, self))

                    # Track primary ABD links (for AffineBodyStateAccessorFeature index mapping).
                    if i_b == 0 or not any(lk == target_link_idx for _, lk in self._primary_abd_links):
                        self._primary_abd_links.append((i_b, target_link_idx))

                    # Contact proxy: map IPC global vertex indices to their parent link.
                    # Only two_way links contribute contact gradient forces to Genesis.
                    if self.options.enable_contact_proxy and coupling_type == "two_way_soft_constraint":
                        n_verts = merged_mesh.vertices().size()
                        for local_idx in range(n_verts):
                            self._vertex_to_link_mapping[self._global_vertex_offset + local_idx] = (
                                target_link_idx,
                                i_b,
                                local_idx,
                            )
                    self._global_vertex_offset += merged_mesh.vertices().size()

                except Exception as e:
                    gs.logger.warning(f"Failed to add rigid link {target_link_idx} to IPC: {e}")

        # Allocate coupling_data with exact size (number of primary ABD links).
        n_coupled_links = len(self._primary_abd_links)
        if n_coupled_links > 0:
            self.coupling_data = get_ipc_coupling_data(n_coupled_links)

        # Pre-allocate Quadrants contact force fields with exact sizes.
        n_links_total = rigid_solver.n_links_ if rigid_solver.is_active else 0
        n_envs = self.sim._B
        if n_coupled_links > 0:
            self._contact_forces_ti = V_VEC(3, dtype=gs.qd_float, shape=n_coupled_links)
            self._contact_torques_ti = V_VEC(3, dtype=gs.qd_float, shape=n_coupled_links)
            self._abd_transforms_ti = V_MAT(n=4, m=4, dtype=gs.qd_float, shape=n_coupled_links)
            self._out_forces_ti = V_VEC(12, dtype=gs.qd_float, shape=n_coupled_links)
            self._link_indices_ti = V(dtype=gs.qd_int, shape=n_coupled_links)
            self._env_indices_ti = V(dtype=gs.qd_int, shape=n_coupled_links)
        if n_links_total > 0 and n_envs > 0:
            self._link_contact_forces_out = V_VEC(3, dtype=gs.qd_float, shape=(n_links_total, n_envs))
            self._link_contact_torques_out = V_VEC(3, dtype=gs.qd_float, shape=(n_links_total, n_envs))

        # Allocate exact-size per-vertex proxy buffers for contact gradient mapping.
        n_proxy_verts = len(self._vertex_to_link_mapping)
        if n_proxy_verts > 0:
            self._vertex_force_gradients = V_VEC(3, dtype=gs.qd_float, shape=n_proxy_verts)
            self._vertex_link_indices = V(dtype=gs.qd_int, shape=n_proxy_verts)
            self._vertex_env_indices = V(dtype=gs.qd_int, shape=n_proxy_verts)
            self._vertex_positions_world = V_VEC(3, dtype=gs.qd_float, shape=n_proxy_verts)
            self._vertex_link_centers = V_VEC(3, dtype=gs.qd_float, shape=n_proxy_verts)

    def _add_articulated_entities_to_ipc(self):
        """Register articulated robots in IPC using ExternalArticulationConstraint."""
        rigid_solver = self.rigid_solver
        scene = self._ipc_scene

        total_articu_qs = 0
        total_articu_joints = 0

        for entity_idx, coupling_type in self._entity_coupling_types.items():
            if coupling_type != "external_articulation":
                continue

            entity = rigid_solver._entities[entity_idx]
            joint_info = extract_articulated_joints(entity)

            if joint_info["n_joints"] == 0:
                continue

            base_link = entity.links[0]
            has_non_fixed_base = not base_link.is_fixed
            base_link_idx = entity.base_link_idx

            abrj = AffineBodyRevoluteJoint()
            abpj = AffineBodyPrismaticJoint()

            joint_geo_slots = []
            joint_objects = []

            for joint in joint_info["revolute_joints"]:
                child_link_idx = joint.link.idx
                parent_link_idx_orig = joint.link.parent_idx if joint.link.parent_idx >= 0 else 0
                parent_link_idx = find_target_link_for_fixed_merge(rigid_solver, parent_link_idx_orig)

                parent_slot = self._find_abd_geometry_slot_by_link(parent_link_idx, env_idx=0)
                child_slot = self._find_abd_geometry_slot_by_link(child_link_idx, env_idx=0)

                if parent_slot is None or child_slot is None:
                    gs.logger.warning(f"Skipping revolute joint {joint.name}: ABD slots not found.")
                    continue

                joint_axis_local = joint.dofs_motion_ang[0]
                child_rot = compute_link_init_world_rotation(rigid_solver, child_link_idx)
                joint_axis = child_rot @ joint_axis_local
                joint_pos = rigid_solver.joints_state.xanchor.to_numpy()[joint.idx, 0]

                axis_length = 1.0
                v1 = joint_pos - (axis_length / 2) * joint_axis
                v2 = joint_pos + (axis_length / 2) * joint_axis
                vertices = np.array([v2, v1], dtype=np.float64)
                edges = np.array([[0, 1]], dtype=np.int32)
                revolute_mesh = linemesh(vertices, edges)
                abrj.apply_to(revolute_mesh, [parent_slot], [0], [child_slot], [0], [100.0])

                joint_obj = scene.objects().create(f"revolute_joint_{entity_idx}_{joint.name}")
                revolute_slot, _ = joint_obj.geometries().create(revolute_mesh)
                joint_geo_slots.append(revolute_slot)
                joint_objects.append(joint_obj)

            for joint in joint_info["prismatic_joints"]:
                child_link_idx = joint.link.idx
                parent_link_idx_orig = joint.link.parent_idx if joint.link.parent_idx >= 0 else 0
                parent_link_idx = find_target_link_for_fixed_merge(rigid_solver, parent_link_idx_orig)

                parent_slot = self._find_abd_geometry_slot_by_link(parent_link_idx, env_idx=0)
                child_slot = self._find_abd_geometry_slot_by_link(child_link_idx, env_idx=0)

                if parent_slot is None or child_slot is None:
                    gs.logger.warning(f"Skipping prismatic joint {joint.name}: ABD slots not found.")
                    continue

                joint_axis_local = joint.dofs_motion_vel[0]
                child_rot = compute_link_init_world_rotation(rigid_solver, child_link_idx)
                joint_axis = child_rot @ joint_axis_local
                joint_pos = rigid_solver.joints_state.xanchor.to_numpy()[joint.idx, 0]

                axis_length = 1.0
                v1 = joint_pos - (axis_length / 2) * joint_axis
                v2 = joint_pos + (axis_length / 2) * joint_axis
                vertices = np.array([v1, v2], dtype=np.float64)
                edges = np.array([[0, 1]], dtype=np.int32)
                prismatic_mesh = linemesh(vertices, edges)
                abpj.apply_to(prismatic_mesh, [parent_slot], [0], [child_slot], [0], [100.0])

                joint_obj = scene.objects().create(f"prismatic_joint_{entity_idx}_{joint.name}")
                prismatic_slot, _ = joint_obj.geometries().create(prismatic_mesh)
                joint_geo_slots.append(prismatic_slot)
                joint_objects.append(joint_obj)

            if not joint_geo_slots:
                gs.logger.warning(f"Entity {entity_idx}: no valid joint geometry slots created.")
                continue

            # Create ExternalArticulationConstraint geometry.
            n_joints = joint_info["n_joints"]
            n_dofs = entity.n_qs

            indices = [0] * len(joint_geo_slots)
            articulation_geo = self._ipc_eac.create_geometry(joint_geo_slots, indices)

            mass_matrix = np.eye(n_joints, dtype=np.float64) * 1e4
            mass_attr = articulation_geo["joint_joint"].find("mass")
            mass_view = view(mass_attr)
            mass_view[:] = mass_matrix.T.flatten()  # Column-major

            art_obj = scene.objects().create(f"articulation_{entity_idx}")
            articulation_slot, _ = art_obj.geometries().create(articulation_geo)

            qs_start = total_articu_qs
            joint_start = total_articu_joints
            total_articu_qs += n_dofs
            total_articu_joints += n_joints

            art_data = {
                "entity": entity,
                "entity_idx": entity_idx,
                "n_dofs": n_dofs,
                "n_joints": n_joints,
                "qs_start": qs_start,
                "joint_start": joint_start,
                "q_start": entity._q_start,
                "joint_qpos_indices": joint_info["joint_qpos_indices"],
                "joint_dof_indices": joint_info["joint_dof_indices"],
                "revolute_joints": joint_info["revolute_joints"],
                "prismatic_joints": joint_info["prismatic_joints"],
                "articulation_geo": articulation_geo,
                "articulation_slot": articulation_slot,
                "articulation_object": art_obj,
                "has_non_fixed_base": has_non_fixed_base,
                "base_link_idx": base_link_idx,
            }
            self._articulated_entities[entity_idx] = art_data

            if has_non_fixed_base:
                self._articulation_with_non_fixed_base.append(entity_idx)

            gs.logger.info(f"Added articulated entity {entity_idx} with {n_joints} joints.")

        # Allocate articulation state with exact sizes.
        if total_articu_qs > 0:
            self.articulation_state = get_articulation_state(total_articu_qs, total_articu_joints, self.sim._B)

    def _find_abd_geometry_slot_by_link(self, link_idx, env_idx=0):
        """Return the ABD geometry slot for the given link, or None if not found."""
        return self._link_to_abd_slot.get((env_idx, link_idx))

    def _finalize_ipc(self):
        """Initialize the IPC world and set up optimized ABD state retrieval."""
        self._ipc_world.init(self._ipc_scene)
        self._ipc_world.dump()
        gs.logger.info("IPC world initialized.")
        self._init_abd_state_accessor()

    def _init_abd_state_accessor(self):
        """Set up AffineBodyStateAccessorFeature for efficient batch ABD state retrieval."""
        try:
            self._abd_state_feature = self._ipc_world.features().find(AffineBodyStateAccessorFeature)
            if self._abd_state_feature is None:
                return

            body_count = self._abd_state_feature.body_count()
            if body_count == 0:
                self._abd_state_feature = None
                return

            identity_matrix = np.eye(4, dtype=np.float64)
            self._abd_state_geo = self._abd_state_feature.create_geometry()
            self._abd_state_geo.instances().create(builtin.transform, identity_matrix)
            self._abd_state_geo.instances().create(builtin.velocity, identity_matrix)

            self._abd_body_idx_to_link = {}
            for abd_body_idx, (env_idx, link_idx) in enumerate(self._primary_abd_links):
                entity_idx = int(self.rigid_solver.links_info.entity_idx[link_idx])
                self._abd_body_idx_to_link[abd_body_idx] = (env_idx, link_idx, entity_idx)

            gs.logger.info(f"AffineBodyStateAccessorFeature initialized ({body_count} bodies).")

        except Exception as e:
            gs.logger.warning(f"AffineBodyStateAccessorFeature not available: {e}. Using legacy path.")
            self._abd_state_feature = None
            self._abd_state_geo = None

    # ------------------------------------------------------------------
    # Properties / configuration API
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True if the IPC world has been initialized."""
        return self._ipc_world is not None

    @property
    def has_rigid_coupling(self) -> bool:
        """True if at least one rigid entity is coupled to IPC."""
        return bool(self._entity_coupling_types)

    def has_coupling_type(self, coupling_type: str) -> bool:
        """Return True if any entity has the given coupling mode."""
        return coupling_type in self._entity_coupling_types.values()

    def set_link_ipc_collision(self, entity, enabled: bool, link_names=None, link_indices=None):
        """Enable or disable IPC collision detection for specific links of an entity."""
        entity_idx = entity._idx_in_solver
        if entity_idx not in self._link_collision_settings:
            self._link_collision_settings[entity_idx] = {}

        target_links = set()
        if link_names is not None:
            for name in link_names:
                try:
                    target_links.add(entity.get_link(name=name).idx)
                except Exception:
                    gs.logger.warning(f"Link '{name}' not found in entity {entity_idx}.")
        if link_indices is not None:
            for local_idx in link_indices:
                target_links.add(local_idx + entity._link_start)

        if not target_links:
            for link in entity.links:
                self._link_collision_settings[entity_idx][link.idx] = enabled
        else:
            for idx in target_links:
                self._link_collision_settings[entity_idx][idx] = enabled

    # ------------------------------------------------------------------
    # Coupling step
    # ------------------------------------------------------------------

    def preprocess(self, f):
        pass

    def couple(self, f):
        """Execute one IPC coupling step."""
        if not self.is_active:
            return

        entities_by_type = categorize_entities_by_coupling_type(self._entity_coupling_types)
        two_way_entities = entities_by_type["two_way_soft_constraint"]
        articulation_entities = entities_by_type["external_articulation"]
        ipc_only_entities = entities_by_type["ipc_only"]

        self._store_genesis_rigid_states()

        if articulation_entities:
            self._pre_advance_external_articulation(articulation_entities)

        self._ipc_world.advance()
        self._ipc_world.retrieve()

        self._retrieve_fem_states(f)

        rigid_entities = set(two_way_entities + ipc_only_entities + self._articulation_with_non_fixed_base)
        if rigid_entities:
            self._retrieve_rigid_states(f, rigid_entities)

        if two_way_entities:
            if self.options.two_way_coupling:
                self._apply_abd_coupling_forces(set(two_way_entities))
            if self.options.enable_contact_proxy:
                self._record_ipc_contact_forces()
                self._apply_ipc_contact_forces()

        if articulation_entities:
            self._post_advance_external_articulation(articulation_entities)

        if ipc_only_entities:
            self._post_advance_ipc_only(ipc_only_entities)

    def couple_grad(self, f):
        """Gradient coupling — not supported by IPC."""
        pass

    def reset(self, envs_idx=None):
        """Reset IPC state to initial condition."""
        self._ipc_world.recover(0)
        self._ipc_world.retrieve()

    # ------------------------------------------------------------------
    # Store Genesis states before IPC advance
    # ------------------------------------------------------------------

    def _store_genesis_rigid_states(self):
        """
        Cache Genesis rigid body transforms before IPC advance.

        Results are stored in ``self._genesis_stored_states``:
        ``link_idx -> {env_idx: 4×4 transform matrix}``.
        """
        if not self.rigid_solver.is_active:
            return

        rigid_solver = self.rigid_solver
        self._genesis_stored_states.clear()
        is_parallelized = self.sim._scene.n_envs > 0

        for env_idx in range(self.sim._B):
            if is_parallelized:
                all_pos = rigid_solver.get_links_pos(envs_idx=env_idx).detach().cpu().numpy()
                all_quat = rigid_solver.get_links_quat(envs_idx=env_idx).detach().cpu().numpy()
            else:
                all_pos = rigid_solver.get_links_pos().detach().cpu().numpy()
                all_quat = rigid_solver.get_links_quat().detach().cpu().numpy()

            if all_pos.ndim == 3:
                all_pos = all_pos[0]
            if all_quat.ndim == 3:
                all_quat = all_quat[0]

            for link_idx in range(all_pos.shape[0]):
                transform = build_link_transform_matrix(all_pos[link_idx], all_quat[link_idx])
                if link_idx not in self._genesis_stored_states:
                    self._genesis_stored_states[link_idx] = {}
                self._genesis_stored_states[link_idx][env_idx] = transform

    # ------------------------------------------------------------------
    # Retrieve states after IPC advance
    # ------------------------------------------------------------------

    def _retrieve_rigid_states(self, f, entity_set=None):
        """Populate ``self.abd_data_by_link`` from IPC ABD state after advance."""
        if not hasattr(self, "_ipc_scene") or not hasattr(self.rigid_solver, "list_env_mesh"):
            return

        if self._abd_state_feature is not None and self._abd_state_geo is not None:
            try:
                self.abd_data_by_link = self._retrieve_rigid_states_optimized(entity_set)
                return
            except Exception as e:
                gs.logger.warning(f"Optimized ABD retrieval failed: {e}. Falling back to legacy path.")

        self.abd_data_by_link = self._retrieve_rigid_states_legacy(entity_set)

    def _retrieve_rigid_states_optimized(self, entity_set=None):
        abd_data_by_link = {}
        self._abd_state_feature.copy_to(self._abd_state_geo)

        trans_attr = self._abd_state_geo.instances().find(builtin.transform)
        if trans_attr is None:
            return abd_data_by_link

        transforms = trans_attr.view()
        vel_attr = self._abd_state_geo.instances().find(builtin.velocity)
        velocities = vel_attr.view() if vel_attr is not None else None

        cd = self.coupling_data
        n_items = 0

        for abd_body_idx, (env_idx, link_idx, entity_idx) in self._abd_body_idx_to_link.items():
            if entity_set is not None and entity_idx not in entity_set:
                continue

            if link_idx not in self._genesis_stored_states or env_idx not in self._genesis_stored_states[link_idx]:
                continue

            aim_transform = self._genesis_stored_states[link_idx][env_idx]
            transform_matrix = transforms[abd_body_idx]

            if link_idx not in abd_data_by_link:
                abd_data_by_link[link_idx] = {}

            entry = {"transform": transform_matrix.copy(), "aim_transform": aim_transform}
            if velocities is not None:
                entry["velocity"] = velocities[abd_body_idx].copy()
            abd_data_by_link[link_idx][env_idx] = entry

            if cd is not None and n_items < cd.link_indices.shape[0]:
                cd.link_indices[n_items] = link_idx
                cd.env_indices[n_items] = env_idx
                cd.ipc_transforms[n_items] = transform_matrix
                cd.aim_transforms[n_items] = aim_transform
                cd.link_masses[n_items] = float(self.rigid_solver.links_info.inertial_mass[link_idx])
                cd.inertia_tensors[n_items] = self.rigid_solver.links_info.inertial_i[link_idx].to_numpy()
                n_items += 1

        if cd is not None:
            cd.n_items = n_items

        return abd_data_by_link

    def _retrieve_rigid_states_legacy(self, entity_set=None):
        rigid_solver = self.rigid_solver
        visitor = SceneVisitor(self._ipc_scene)
        abd_data_by_link = {}

        for geo_slot in visitor.geometries():
            if not isinstance(geo_slot, SimplicialComplexSlot):
                continue
            geo = geo_slot.geometry()
            if geo.dim() not in (2, 3):
                continue
            meta = read_ipc_geometry_metadata(geo)
            if meta is None:
                continue
            solver_type, env_idx, link_idx = meta
            if solver_type != "rigid":
                continue

            if entity_set is not None:
                entity_idx = int(self.rigid_solver.links_info.entity_idx[link_idx])
                if entity_idx not in entity_set:
                    continue

            try:
                transforms = geo.transforms()
                transform_matrix = view(transforms)[0].copy() if transforms.size() > 0 else None
                aim_transform = self._genesis_stored_states.get(link_idx, {}).get(env_idx)
                if link_idx not in abd_data_by_link:
                    abd_data_by_link[link_idx] = {}
                abd_data_by_link[link_idx][env_idx] = {
                    "transform": transform_matrix,
                    "aim_transform": aim_transform,
                }
            except Exception as e:
                gs.logger.warning(f"Failed to retrieve ABD geometry data: {e}")

        return abd_data_by_link

    def _retrieve_fem_states(self, f):
        """Write IPC vertex positions back to Genesis FEM entities."""
        if not self.fem_solver.is_active:
            return

        visitor = SceneVisitor(self._ipc_scene)
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
            solver_type, env_idx, entity_idx = meta
            if solver_type not in ("fem", "cloth"):
                continue

            try:
                if entity_idx not in fem_geo_by_entity:
                    fem_geo_by_entity[entity_idx] = {}
                proc_geo = merge(apply_transform(geo)) if geo.instances().size() >= 1 else geo
                pos = proc_geo.positions().view().reshape(-1, 3)
                fem_geo_by_entity[entity_idx][env_idx] = pos
            except Exception:
                continue

        for entity_idx, env_positions in fem_geo_by_entity.items():
            if entity_idx >= len(self.fem_solver._entities):
                continue
            entity = self.fem_solver._entities[entity_idx]
            env_pos_list = []
            for env_idx in range(self.sim._B):
                env_pos_list.append(env_positions.get(env_idx, np.zeros((0, 3))))
            if env_pos_list:
                all_env_pos = np.stack(env_pos_list, axis=0, dtype=gs.np_float)
                entity.set_pos(0, all_env_pos)

    # ------------------------------------------------------------------
    # Two-way coupling: apply ABD soft-constraint forces
    # ------------------------------------------------------------------

    def _record_ipc_contact_forces(self):
        """
        Extract per-link contact forces from IPC contact gradients.

        Reads vertex force gradients from IPC's ``contact_system`` feature, maps them to
        Genesis rigid links via ``_vertex_to_link_mapping``, accumulates per-link forces
        and torques using ``compute_link_contact_forces_kernel``, and stores results in
        ``_ipc_contact_forces``.
        """
        self._ipc_contact_forces.clear()

        if not self._vertex_to_link_mapping:
            return

        contact_feature = self._ipc_world.features().find("core/contact_system")
        if contact_feature is None:
            return

        dt2 = float(self.sim._dt) ** 2
        total_force_dict = {}  # {global_vertex_idx: force_gradient_3d}

        for prim_type in contact_feature.contact_primitive_types():
            vert_grad = Geometry()
            contact_feature.contact_gradient(prim_type, vert_grad)
            instances = vert_grad.instances()
            i_attr = instances.find("i")
            grad_attr = instances.find("grad")
            if i_attr is None or grad_attr is None:
                continue
            indices = view(i_attr)
            gradients = view(grad_attr)
            if len(indices) == 0 or gradients.size == 0:
                continue
            # Gradient shape is (n, 3) for vectors or (n, 3, 3) for matrices; take first 3.
            if gradients.ndim == 3:
                scaled_grads = gradients.reshape(len(gradients), -1)[:, :3] / dt2
            else:
                scaled_grads = gradients[:, :3] / dt2
            for k, idx in enumerate(indices):
                key = int(idx)
                if key in total_force_dict:
                    total_force_dict[key] += scaled_grads[k]
                else:
                    total_force_dict[key] = scaled_grads[k].copy()

        if not total_force_dict:
            return

        # Collect world-space vertex positions for two_way links from the IPC scene.
        vertex_to_link = self._vertex_to_link_mapping
        link_vertex_positions = {}  # {(link_idx, env_idx): [world_pos, ...]}
        global_vert_offset = 0

        visitor = SceneVisitor(self._ipc_scene)
        for geo_slot in visitor.geometries():
            if not isinstance(geo_slot, SimplicialComplexSlot):
                continue
            geo = geo_slot.geometry()
            if geo.dim() not in (2, 3):
                continue
            n_verts = geo.vertices().size()
            meta = read_ipc_geometry_metadata(geo)
            if meta is not None and meta[0] == "rigid" and global_vert_offset in vertex_to_link:
                _, env_idx, link_idx = meta
                transforms = geo.transforms()
                if transforms.size() > 0:
                    T = view(transforms)[0]
                    positions = view(geo.positions())
                    pos_list = []
                    for local_idx in range(n_verts):
                        p = np.array(positions[local_idx]).flatten()[:3]
                        pos_list.append((T @ np.append(p, 1.0))[:3])
                    link_vertex_positions[(link_idx, env_idx)] = pos_list
            global_vert_offset += n_verts

        if not link_vertex_positions:
            return

        link_centers_dict = {key: np.mean(verts, axis=0) for key, verts in link_vertex_positions.items()}

        # Fill per-vertex buffers for kernel call (only vertices with actual contact forces).
        n_entries = 0
        max_entries = len(vertex_to_link)
        for vert_idx, force_grad in total_force_dict.items():
            if vert_idx not in vertex_to_link or n_entries >= max_entries:
                continue
            link_idx, env_idx, local_idx = vertex_to_link[vert_idx]
            if (link_idx, env_idx) not in link_vertex_positions:
                continue
            center_pos = link_centers_dict.get((link_idx, env_idx))
            if center_pos is None:
                continue
            self._vertex_force_gradients[n_entries] = force_grad
            self._vertex_link_indices[n_entries] = link_idx
            self._vertex_env_indices[n_entries] = env_idx
            self._vertex_positions_world[n_entries] = link_vertex_positions[(link_idx, env_idx)][local_idx]
            self._vertex_link_centers[n_entries] = center_pos
            n_entries += 1

        if n_entries == 0:
            return

        self._link_contact_forces_out.fill(0.0)
        self._link_contact_torques_out.fill(0.0)

        compute_link_contact_forces_kernel(
            n_entries,
            self._vertex_force_gradients,
            self._vertex_link_indices,
            self._vertex_env_indices,
            self._vertex_positions_world,
            self._vertex_link_centers,
            self._link_contact_forces_out,
            self._link_contact_torques_out,
        )

        # Extract results once and store per-link.
        forces_all = qd_to_numpy(self._link_contact_forces_out, transpose=True)
        torques_all = qd_to_numpy(self._link_contact_torques_out, transpose=True)

        for link_idx, env_idx in link_centers_dict:
            force = forces_all[env_idx, link_idx]
            torque = torques_all[env_idx, link_idx]
            if np.any(force != 0.0) or np.any(torque != 0.0):
                if link_idx not in self._ipc_contact_forces:
                    self._ipc_contact_forces[link_idx] = {}
                self._ipc_contact_forces[link_idx][env_idx] = {"force": force, "torque": torque}

    def _apply_ipc_contact_forces(self):
        """Apply contact forces from ``_ipc_contact_forces`` to Genesis rigid bodies."""
        if not self._ipc_contact_forces:
            return

        rigid_solver = self.rigid_solver
        is_parallelized = self.sim._scene.n_envs > 0

        if is_parallelized:
            env_batches = {}
            for link_idx, env_data in self._ipc_contact_forces.items():
                for env_idx, force_data in env_data.items():
                    if env_idx not in env_batches:
                        env_batches[env_idx] = {"link_indices": [], "forces": [], "torques": []}
                    env_batches[env_idx]["link_indices"].append(link_idx)
                    env_batches[env_idx]["forces"].append(force_data["force"] * 0.5)
                    env_batches[env_idx]["torques"].append(force_data["torque"] * 0.5)

            for env_idx, batch in env_batches.items():
                if not batch["link_indices"]:
                    continue
                rigid_solver.apply_links_external_force(
                    force=batch["forces"], links_idx=batch["link_indices"], envs_idx=env_idx, local=False
                )
                rigid_solver.apply_links_external_torque(
                    torque=batch["torques"], links_idx=batch["link_indices"], envs_idx=env_idx, local=False
                )
        else:
            all_link_indices = []
            all_forces = []
            all_torques = []
            for link_idx, env_data in self._ipc_contact_forces.items():
                for force_data in env_data.values():
                    all_link_indices.append(link_idx)
                    all_forces.append(force_data["force"] * 0.5)
                    all_torques.append(force_data["torque"] * 0.5)
            if all_link_indices:
                rigid_solver.apply_links_external_force(
                    force=all_forces, links_idx=all_link_indices, envs_idx=None, local=False
                )
                rigid_solver.apply_links_external_torque(
                    torque=all_torques, links_idx=all_link_indices, envs_idx=None, local=False
                )

    def _apply_abd_coupling_forces(self, entity_set=None):
        """Apply soft-constraint coupling forces from IPC to Genesis rigid bodies."""
        cd = self.coupling_data
        if cd is None or cd.n_items == 0:
            return

        n_items = cd.n_items
        translation_strength = float(self.options.constraint_strength_translation)
        rotation_strength = float(self.options.constraint_strength_rotation)
        dt = float(self.sim._dt)
        dt2 = dt * dt

        compute_coupling_forces_kernel(
            n_items,
            cd.ipc_transforms[:n_items],
            cd.aim_transforms[:n_items],
            cd.link_masses[:n_items],
            cd.inertia_tensors[:n_items],
            translation_strength,
            rotation_strength,
            dt2,
            cd.out_forces[:n_items],
            cd.out_torques[:n_items],
        )

        rigid_solver = self.rigid_solver
        is_parallelized = self.sim._scene.n_envs > 0

        if is_parallelized:
            env_batches = {}
            for i in range(n_items):
                env_idx = int(cd.env_indices[i])
                if env_idx not in env_batches:
                    env_batches[env_idx] = {"link_indices": [], "forces": [], "torques": []}
                env_batches[env_idx]["link_indices"].append(int(cd.link_indices[i]))
                env_batches[env_idx]["forces"].append(cd.out_forces[i])
                env_batches[env_idx]["torques"].append(cd.out_torques[i])

            for env_idx, batch in env_batches.items():
                for j, link_idx in enumerate(batch["link_indices"]):
                    force = batch["forces"][j]
                    torque = batch["torques"][j]
                    if np.isnan(force).any() or np.isnan(torque).any():
                        gs.logger.warning(f"NaN in coupling force for link {link_idx}, env {env_idx}. Skipping.")
                        continue
                    try:
                        rigid_solver.apply_links_external_force(
                            force=force.reshape(1, 1, 3), links_idx=link_idx, envs_idx=env_idx
                        )
                        rigid_solver.apply_links_external_torque(
                            torque=torque.reshape(1, 1, 3), links_idx=link_idx, envs_idx=env_idx
                        )
                    except Exception as e:
                        gs.logger.warning(f"Failed to apply ABD coupling force: {e}")
        else:
            for i in range(n_items):
                link_idx = int(cd.link_indices[i])
                force = cd.out_forces[i]
                torque = cd.out_torques[i]
                if np.isnan(force).any() or np.isnan(torque).any():
                    gs.logger.warning(f"NaN in coupling force for link {link_idx}. Skipping.")
                    continue
                try:
                    rigid_solver.apply_links_external_force(
                        force=force.reshape(1, 3), links_idx=link_idx, envs_idx=None
                    )
                    rigid_solver.apply_links_external_torque(
                        torque=torque.reshape(1, 3), links_idx=link_idx, envs_idx=None
                    )
                except Exception as e:
                    gs.logger.warning(f"Failed to apply ABD coupling force: {e}")

    # ------------------------------------------------------------------
    # External articulation coupling
    # ------------------------------------------------------------------

    def _pre_advance_external_articulation(self, entity_indices):
        """
        Pre-advance: read Genesis qpos, compute joint deltas, send to IPC.
        Updates ref_dof_prev snapshot and sends delta_theta_tilde to EAC geometry.
        """
        if not self._articulated_entities:
            return

        art_state = self.articulation_state
        rigid_solver = self.rigid_solver
        is_parallelized = self.sim._scene.n_envs > 0

        for entity_idx in entity_indices:
            if entity_idx not in self._articulated_entities:
                continue
            art = self._articulated_entities[entity_idx]
            entity = art["entity"]
            qs_start = art["qs_start"]
            joint_start = art["joint_start"]
            n_dofs = art["n_dofs"]
            n_joints = art["n_joints"]
            joint_qpos_indices = art["joint_qpos_indices"]
            joint_dof_indices = art["joint_dof_indices"]
            q_start = art["q_start"]  # start of this entity's qpos in global rigid solver qpos
            articulation_slot = art["articulation_slot"]
            env_idx = 0  # EAC operates on env 0 for geometry

            # Get current qpos for all envs at once via torch.
            qpos_np = qd_to_numpy(rigid_solver.global_info.qpos, transpose=True)
            # qpos_np shape: [n_envs, n_total_qs] (after transpose)

            for i_env in range(self.sim._B):
                qpos_entity = qpos_np[i_env, q_start : q_start + n_dofs]

                # Compute delta_theta_tilde = qpos_current - ref_dof_prev (per joint).
                for j_idx, qpos_idx in enumerate(joint_qpos_indices):
                    ref = float(art_state.ref_dof_prev[qs_start + qpos_idx, i_env])
                    cur = float(qpos_entity[qpos_idx])
                    delta = cur - ref
                    art_state.delta_theta_ipc[joint_start + j_idx, i_env] = delta

            # Update IPC EAC geometry: set ref_dof_prev on instances.
            articulation_geo = art["articulation_geo"]
            ref_dof_attr = articulation_geo["joint"].find("ref_dof_prev")
            if ref_dof_attr is not None:
                ref_view = view(ref_dof_attr)
                for j_idx, qpos_idx in enumerate(joint_qpos_indices):
                    ref_val = float(art_state.ref_dof_prev[qs_start + qpos_idx, env_idx])
                    ref_view[j_idx] = ref_val

            # Set delta_theta_tilde on IPC geometry.
            delta_attr = articulation_geo["joint"].find("delta_theta_tilde")
            if delta_attr is not None:
                delta_view = view(delta_attr)
                for j_idx in range(n_joints):
                    delta_view[j_idx] = float(art_state.delta_theta_ipc[joint_start + j_idx, env_idx])

            # Extract and transfer mass matrix (from rigid solver to IPC EAC).
            mass_mat_np = qd_to_numpy(rigid_solver.global_info.mass_mat, transpose=True)
            # mass_mat_np shape: [n_envs, n_dofs, n_dofs]
            mass_mat_entity = mass_mat_np[env_idx, q_start : q_start + n_dofs, q_start : q_start + n_dofs]

            mass_attr = articulation_geo["joint_joint"].find("mass")
            if mass_attr is not None:
                mass_view = view(mass_attr)
                for i in range(n_joints):
                    dof_i = joint_dof_indices[i]
                    for j in range(n_joints):
                        dof_j = joint_dof_indices[j]
                        # column-major order for IPC
                        mass_view[j * n_joints + i] = float(mass_mat_entity[dof_i, dof_j])

            # Synchronize per-ABD-body ref_dof_prev from stored Genesis transforms.
            for joint_idx, joint in enumerate(art["revolute_joints"] + art["prismatic_joints"]):
                child_link_idx = joint.link.idx
                abd_geo_slot = self._find_abd_geometry_slot_by_link(child_link_idx, env_idx=0)
                if abd_geo_slot is None:
                    continue
                abd_geo = abd_geo_slot.geometry()
                if abd_geo is None:
                    continue
                ref_dof_prev_attr = abd_geo.instances().find("ref_dof_prev")
                if ref_dof_prev_attr is None:
                    continue
                ref_dof_view = view(ref_dof_prev_attr)
                key = (entity_idx, joint_idx, 0)
                if key in self._prev_link_transforms:
                    q = affine_body.transform_to_q(self._prev_link_transforms[key])
                    ref_dof_view[0] = q
                elif child_link_idx in self._genesis_stored_states and 0 in self._genesis_stored_states[child_link_idx]:
                    q = affine_body.transform_to_q(self._genesis_stored_states[child_link_idx][0])
                    ref_dof_view[0] = q

    def _post_advance_external_articulation(self, entity_indices):
        """
        Post-advance: read IPC's delta_theta output, compute new qpos, write to Genesis.
        """
        if not self._articulated_entities:
            return

        art_state = self.articulation_state
        rigid_solver = self.rigid_solver
        is_parallelized = self.sim._scene.n_envs > 0
        qpos_np = qd_to_numpy(rigid_solver.global_info.qpos, transpose=True)
        # qpos_np shape: [n_envs, n_total_qs]

        for entity_idx in entity_indices:
            if entity_idx not in self._articulated_entities:
                continue
            art = self._articulated_entities[entity_idx]
            entity = art["entity"]
            qs_start = art["qs_start"]
            joint_start = art["joint_start"]
            n_dofs = art["n_dofs"]
            n_joints = art["n_joints"]
            joint_qpos_indices = art["joint_qpos_indices"]
            q_start = art["q_start"]
            articulation_slot = art["articulation_slot"]

            # Read delta_theta from IPC EAC geometry.
            scene_art_geo = articulation_slot.geometry()
            delta_theta_attr = scene_art_geo["joint"].find("delta_theta")
            if delta_theta_attr is not None:
                delta_theta_view = view(delta_theta_attr)
                env_idx = 0  # EAC is env 0
                for j_idx in range(n_joints):
                    art_state.delta_theta_ipc[joint_start + j_idx, env_idx] = float(delta_theta_view[j_idx])

            # Apply delta to all environments.
            for i_env in range(self.sim._B):
                qpos_entity = qpos_np[i_env, q_start : q_start + n_dofs].copy()

                # For each joint DOF, apply delta on top of ref_dof_prev.
                for j_idx, qpos_idx in enumerate(joint_qpos_indices):
                    ref = float(art_state.ref_dof_prev[qs_start + qpos_idx, i_env])
                    delta = float(art_state.delta_theta_ipc[joint_start + j_idx, 0])
                    qpos_entity[qpos_idx] = ref + delta

                qpos_new_np = qpos_entity.astype(gs.np_float)
                if self.sim._B > 1:
                    entity.set_qpos(qpos_new_np, envs_idx=i_env, zero_velocity=False)
                else:
                    entity.set_qpos(qpos_new_np, zero_velocity=False)

            # Update ref_dof_prev = qpos after IPC step.
            qpos_np_updated = qd_to_numpy(rigid_solver.global_info.qpos, transpose=True)
            for i_env in range(self.sim._B):
                for dof_idx in range(n_dofs):
                    art_state.ref_dof_prev[qs_start + dof_idx, i_env] = float(qpos_np_updated[i_env, q_start + dof_idx])

            # For non-fixed base: apply IPC base link transform from abd_data_by_link.
            if art["has_non_fixed_base"]:
                base_link_idx = art["base_link_idx"]
                for i_env in range(self.sim._B):
                    env_data = self.abd_data_by_link.get(base_link_idx, {})
                    ipc_transform = env_data.get(i_env, {}).get("transform")
                    if ipc_transform is None:
                        continue
                    pos, quat_wxyz = decompose_transform_matrix(ipc_transform)
                    if self.sim._B > 1:
                        rigid_solver.set_base_links_pos(pos, [base_link_idx], envs_idx=i_env, relative=False)
                        rigid_solver.set_base_links_quat(quat_wxyz, [base_link_idx], envs_idx=i_env, relative=False)
                    else:
                        rigid_solver.set_base_links_pos(pos, [base_link_idx], envs_idx=None, relative=False)
                        rigid_solver.set_base_links_quat(quat_wxyz, [base_link_idx], envs_idx=None, relative=False)

                    ipc_velocity = env_data.get(i_env, {}).get("velocity")
                    if ipc_velocity is not None:
                        linear_vel = ipc_velocity[:3, 3]
                        R_curr = ipc_transform[:3, :3]
                        dR_dt = ipc_velocity[:3, :3]
                        omega_skew = dR_dt @ R_curr.T
                        angular_vel = np.array(
                            [
                                (omega_skew[2, 1] - omega_skew[1, 2]) / 2.0,
                                (omega_skew[0, 2] - omega_skew[2, 0]) / 2.0,
                                (omega_skew[1, 0] - omega_skew[0, 1]) / 2.0,
                            ]
                        )
                        base_vel = np.concatenate([linear_vel, angular_vel])
                        base_dofs_local = list(range(6))
                        if self.sim._B > 1:
                            entity.set_dofs_velocity(base_vel, dofs_idx_local=base_dofs_local, envs_idx=i_env)
                        else:
                            entity.set_dofs_velocity(base_vel, dofs_idx_local=base_dofs_local)

            # Store current link transforms for next step's per-ABD ref_dof_prev sync.
            for joint_idx, joint in enumerate(art["revolute_joints"] + art["prismatic_joints"]):
                child_link_idx = joint.link.idx
                if child_link_idx in self._genesis_stored_states and 0 in self._genesis_stored_states[child_link_idx]:
                    key = (entity_idx, joint_idx, 0)
                    self._prev_link_transforms[key] = self._genesis_stored_states[child_link_idx][0].copy()

    # ------------------------------------------------------------------
    # IPC-only coupling
    # ------------------------------------------------------------------

    def _post_advance_ipc_only(self, entity_indices):
        """Set Genesis transforms from IPC results for ipc_only entities."""
        rigid_solver = self.rigid_solver
        is_parallelized = self.sim._scene.n_envs > 0

        for i_env in range(self.sim._B):
            for entity_idx in entity_indices:
                entity = rigid_solver._entities[entity_idx]
                link_idx = entity.base_link_idx

                env_data = self.abd_data_by_link.get(link_idx, {})
                ipc_transform = env_data.get(i_env, {}).get("transform")
                if ipc_transform is None:
                    continue

                pos, quat_wxyz = decompose_transform_matrix(ipc_transform)

                if is_robot_entity(entity) and entity.n_qs >= 7:
                    if is_parallelized:
                        qpos_current = entity.get_qpos(envs_idx=i_env).detach().cpu().numpy()
                    else:
                        qpos_current = entity.get_qpos().detach().cpu().numpy()
                    if qpos_current.ndim > 1:
                        qpos_current = qpos_current[0]
                    qpos_new = qpos_current.copy()
                    qpos_new[:3] = pos
                    qpos_new[3:7] = quat_wxyz
                    if is_parallelized:
                        entity.set_qpos(qpos_new, envs_idx=i_env, zero_velocity=True, skip_forward=True)
                    else:
                        entity.set_qpos(qpos_new, zero_velocity=True, skip_forward=True)
                else:
                    if is_parallelized:
                        rigid_solver.set_base_links_pos(pos, [link_idx], envs_idx=i_env, relative=False)
                        rigid_solver.set_base_links_quat(quat_wxyz, [link_idx], envs_idx=i_env, relative=False)
                        entity.zero_all_dofs_velocity(envs_idx=i_env)
                    else:
                        rigid_solver.set_base_links_pos(pos, [link_idx], envs_idx=None, relative=False)
                        rigid_solver.set_base_links_quat(quat_wxyz, [link_idx], envs_idx=None, relative=False)
                        entity.zero_all_dofs_velocity(envs_idx=None)
