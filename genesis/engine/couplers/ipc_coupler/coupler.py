import logging
import os
import tempfile
import weakref
from functools import partial
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.materials.FEM.cloth import Cloth
from genesis.options.solvers import IPCCouplerOptions, RigidOptions
from genesis.repr_base import RBC
from genesis.utils.mesh import are_meshes_overlapping
from genesis.utils.misc import geometric_mean, harmonic_mean, qd_to_numpy, qd_to_torch, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.entities import FEMEntity, RigidEntity
    from genesis.engine.entities.rigid_entity import RigidJoint, RigidLink
    from genesis.engine.simulator import Simulator
    from genesis.engine.solvers import FEMSolver, RigidSolver

# Check if libuipc is available
try:
    import uipc

    UIPC_AVAILABLE = True
except ImportError:
    UIPC_AVAILABLE = False

if TYPE_CHECKING or UIPC_AVAILABLE:
    import polyscope as ps
    from uipc.backend import SceneVisitor
    from uipc.constitution import (
        AffineBodyConstitution,
        AffineBodyPrismaticJoint,
        AffineBodyRevoluteJoint,
        DiscreteShellBending,
        ElasticModuli,
        ElasticModuli2D,
        ExternalArticulationConstraint,
        SoftTransformConstraint,
        StableNeoHookean,
        StrainLimitingBaraffWitkinShell,
    )
    from uipc.core import Engine, World, Scene, AffineBodyStateAccessorFeature, ContactElement, SubsceneElement
    from uipc.geometry import GeometrySlot, SimplicialComplex, SimplicialComplexSlot
    from uipc.gui import SceneGUI

    from .data import COUPLING_TYPE, ABDLinkData, ArticulatedEntityData
    from .utils import (
        build_ipc_scene_config,
        compute_link_to_link_transform,
        find_target_link_for_fixed_merge,
        read_ipc_geometry_metadata,
    )


# Affine body stiffness in MPa
ABD_KAPPA = 100.0
# TODO: consider deriving from Genesis joint properties instead of hardcoding.
JOINT_STRENGTH_RATIO = 100.0


class IPCCoupler(RBC):
    """
    Coupler class for handling Incremental Potential Contact (IPC) simulation coupling.

    This coupler manages the communication between Genesis solvers and the IPC system,
    including rigid bodies (as ABD objects) and FEM bodies in a unified contact framework.
    """

    def __init__(self, simulator: "Simulator", options: IPCCouplerOptions) -> None:
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
                "Python module 'uipc' is required by IPCCoupler but is not installed. Please install it via "
                "`pip install pyuipc`."
            )

        self.sim = simulator
        self.options = options

        assert gs.use_zerocopy, (
            "IPC coupler requires zero-copy, which is not supported on this platform. "
            "Make sure Torch and Quadrants are sharing the same device."
        )

        # Define some proxies for convenience
        self.rigid_solver: "RigidSolver" = self.sim.rigid_solver
        self.fem_solver: "FEMSolver" = self.sim.fem_solver

        # ==== IPC System Infrastructure ====
        self._ipc_engine: Engine | None = None
        self._ipc_world: World | None = None
        self._ipc_scene = Scene(build_ipc_scene_config(self.options, self.sim.options))
        self._ipc_subscenes: list[SubsceneElement] = []
        self._ipc_constitution_tabular = self._ipc_scene.constitution_tabular()
        self._ipc_contact_tabular = self._ipc_scene.contact_tabular()
        self._ipc_subscene_tabular = self._ipc_scene.subscene_tabular()
        self._ipc_objects = self._ipc_scene.objects()
        self._ipc_animator = self._ipc_scene.animator()

        # ==== IPC Constitutions ====
        self._ipc_abd: AffineBodyConstitution | None = None
        self._ipc_stk: StableNeoHookean | None = None
        self._ipc_stc: SoftTransformConstraint | None = None
        self._ipc_nks: StrainLimitingBaraffWitkinShell | None = None
        self._ipc_dsb: DiscreteShellBending | None = None
        self._ipc_eac: ExternalArticulationConstraint | None = None

        # ==== IPC Contact Elements ====
        self._ipc_no_collision_contact: ContactElement = self._ipc_contact_tabular.create("no_collision_contact")
        self._ipc_fems_contact: dict["FEMEntity", ContactElement] = {}
        self._ipc_clothes_contact: dict["FEMEntity", ContactElement] = {}
        self._ipc_abd_links_contact: dict["RigidLink", ContactElement] = {}
        self._ipc_grounds_contact: dict["RigidEntity", ContactElement] = {}

        # ==== Entity Coupling Configuration ====
        self._coup_type_by_entity: dict["RigidEntity", COUPLING_TYPE] = {}
        # Link filter for two_way_soft_constraint coupling
        self._coup_links: dict["RigidEntity", set["RigidLink"]] = {}
        self._coupling_collision_settings: dict["RigidEntity", dict["RigidLink", bool]] = {}
        self._entities_by_coup_type: dict[COUPLING_TYPE, list["RigidEntity"]] = {}

        # ==== ABD Geometry & State ====
        # Cached merged world-frame trimesh per link for neutral-pose overlap check
        self._abd_merged_meshes: dict["RigidLink", trimesh.Trimesh] = {}
        self._abd_state_feature: AffineBodyStateAccessorFeature | None = None
        self._abd_state_geom: SimplicialComplex | None = None
        # Set to True when set_qpos/set_dofs_position is called; triggers IPC state sync before next advance
        self._is_abd_updated: bool = False

        # ==== Input/Output Data ====
        self._abd_data_by_link: dict["RigidLink", ABDLinkData] = {}
        self._articulation_data_by_entity: dict["RigidEntity", ArticulatedEntityData] = {}

        # ==== GUI ====
        self._ipc_gui: SceneGUI | None = None

    # ============================================================
    # Section 1: Configuration API
    # ============================================================

    def build(self) -> None:
        """Build IPC system"""
        # IPC coupler builds a single IPC scene shared across all envs, so it requires
        # identical geometry topology (links, joints, geoms) across environments.
        # Batched info options allow per-env topology which is incompatible.
        if self.rigid_solver.is_active:
            rigid_options = cast(RigidOptions, self.rigid_solver._options)
            if rigid_options.batch_links_info or rigid_options.batch_dofs_info or rigid_options.batch_joints_info:
                gs.raise_exception(
                    "IPC coupler does not support batched rigid info (batch_links_info, batch_dofs_info, "
                    "batch_joints_info). Please disable these options when using IPC coupling."
                )

        self._B = self.sim._B

        self._init_ipc()
        self._setup_coupling_config()
        self._add_objects_to_ipc()
        self._finalize_ipc()
        self._init_accessors()

        if os.environ.get("GS_ENABLE_IPC_GUI", "0") == "1":
            self._init_ipc_gui()

    def _setup_coupling_config(self):
        """Read coup_type, coup_links, and collision settings from entity materials."""
        assert gs.logger is not None

        entity: "RigidEntity"
        for i_e, entity in enumerate(cast(list["RigidEntity"], self.rigid_solver.entities)):
            if not entity.material.needs_coup:
                continue
            coup_type = entity.material.coup_type
            is_robot = any(j.type not in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.FIXED) for j in entity.joints)
            if coup_type is None:
                # Auto-select: robots get articulation coupling, objects get ipc_only
                if is_robot:
                    coup_type = "external_articulation" if entity.base_link.is_fixed else "two_way_soft_constraint"
                else:
                    coup_type = "ipc_only"

            self._coup_type_by_entity[entity] = coup_type = getattr(COUPLING_TYPE, coup_type.upper())
            if coup_type == COUPLING_TYPE.EXTERNAL_ARTICULATION:
                if not entity.base_link.is_fixed:
                    gs.raise_exception(
                        f"Rigid entity {i_e} has a non-fixed base. "
                        f"Use 'two_way_soft_constraint' instead of 'external_articulation'."
                    )
                if not is_robot:
                    gs.raise_exception(
                        f"Rigid entity {i_e} has no articulated joints. Use 'ipc_only' instead of "
                        "'external_articulation'."
                    )
            elif coup_type == COUPLING_TYPE.IPC_ONLY:
                if is_robot:
                    gs.raise_exception(
                        f"Rigid entity {i_e} has articulated joints. Use 'external_articulation' instead of 'ipc_only'."
                    )
            gs.logger.debug(f"Rigid entity {i_e}: coupling type '{coup_type.name.lower()}'")

            # Resolve link filter from material
            link_filter_names = entity.material.coup_links
            if link_filter_names is not None:
                self._coup_links[entity] = set(map(entity.get_link, link_filter_names))
                gs.logger.debug(f"Rigid entity {i_e}: IPC link filter set to {len(link_filter_names)} link(s)")

            # Resolve collision settings from material
            if not entity.material.enable_coup_collision:
                # Disable collision for all links
                self._coupling_collision_settings[entity] = {link: False for link in entity.links}
                gs.logger.debug(f"Rigid entity {i_e}: IPC collision disabled for all links")
            elif entity.material.coup_collision_links is not None:
                # Positive filter: only named links get collision, others disabled
                allowed = set(entity.material.coup_collision_links)
                self._coupling_collision_settings[entity] = {
                    link: False for link in entity.links if link.name not in allowed
                }
                gs.logger.debug(f"Rigid entity {i_e}: IPC collision limited to {allowed}")

        # Categorize entities by coupling type
        for entity, coup_type in self._coup_type_by_entity.items():
            self._entities_by_coup_type.setdefault(coup_type, []).append(entity)

    def _init_ipc(self) -> None:
        """Initialize IPC system components"""
        assert gs.logger is not None

        if gs.logger.level <= logging.DEBUG:
            uipc.Logger.set_level(uipc.Logger.Level.Info)
            uipc.Timer.enable_all()
        else:
            uipc.Logger.set_level(uipc.Logger.Level.Error)
            uipc.Timer.disable_all()

        # Create workspace directory for IPC output, named after scene UID.
        workspace = os.path.join(tempfile.gettempdir(), f"genesis_ipc_{self.sim.scene.uid.full()}")
        os.makedirs(workspace, exist_ok=False)

        # Note: gpu_device option may need to be set via CUDA environment variables (CUDA_VISIBLE_DEVICES)
        # before Genesis initialization, as libuipc Engine does not expose device selection in constructor
        self._ipc_engine = Engine("cuda", workspace)
        self._ipc_world = World(self._ipc_engine)

        # Set up sub-scenes for multi-environment to isolate per-environment contacts if batched
        for env_idx in range(self._B):
            ipc_subscene = self._ipc_subscene_tabular.create(f"subscene_{env_idx}")
            for other_ipc_subscene in self._ipc_subscenes:
                self._ipc_subscene_tabular.insert(other_ipc_subscene, ipc_subscene, False)
            self._ipc_subscenes.append(ipc_subscene)

    def _add_objects_to_ipc(self) -> None:
        """Add objects from solvers to IPC system"""
        # Add FEM entities to IPC
        if self.fem_solver.is_active:
            self._add_fem_entities_to_ipc()

        # Add rigid geoms and articulated entities to IPC based on per-entity coupling types
        if self.rigid_solver.is_active:
            self._add_rigid_geoms_to_ipc()
            self._add_articulation_entities_to_ipc()

        # Register all per-entity contact pair models with per-material friction
        self._register_contact_pairs()

    def _add_fem_entities_to_ipc(self) -> None:
        """Add FEM entities to the existing IPC scene (includes both volumetric FEM and cloth)"""

        entity: "FEMEntity"
        for i_e, entity in enumerate(cast(list["FEMEntity"], self.fem_solver.entities)):
            is_cloth = isinstance(entity.material, Cloth)
            solver_type = "cloth" if is_cloth else "fem"

            # ---- Create mesh (env-independent geometry) ----
            # trimesh for cloth (2D shell), tetmesh for volumetric FEM (3D)
            if is_cloth:
                verts = tensor_to_array(entity.init_positions).astype(np.float64, copy=False)
                faces = entity.surface_triangles.astype(np.int32, copy=False)
                mesh = uipc.geometry.trimesh(verts, faces)
            else:
                mesh = uipc.geometry.tetmesh(tensor_to_array(entity.init_positions), entity.elems)
            uipc.geometry.label_surface(mesh)

            # ---- Apply constitutions (env-independent) ----
            # Apply per-entity contact element
            if is_cloth:
                self._ipc_clothes_contact[entity] = self._ipc_contact_tabular.create(f"cloth_contact_{i_e}")
                self._ipc_clothes_contact[entity].apply_to(mesh)
            else:
                self._ipc_fems_contact[entity] = self._ipc_contact_tabular.create(f"fem_contact_{i_e}")
                self._ipc_fems_contact[entity].apply_to(mesh)

            # Apply material constitution based on type
            if is_cloth:
                if self._ipc_nks is None:
                    self._ipc_nks = StrainLimitingBaraffWitkinShell()
                    self._ipc_constitution_tabular.insert(self._ipc_nks)

                moduli = ElasticModuli2D.youngs_poisson(entity.material.E, entity.material.nu)
                self._ipc_nks.apply_to(
                    mesh, moduli=moduli, mass_density=entity.material.rho, thickness=entity.material.thickness
                )

                if entity.material.bending_stiffness is not None:
                    if self._ipc_dsb is None:
                        self._ipc_dsb = DiscreteShellBending()
                        self._ipc_constitution_tabular.insert(self._ipc_dsb)

                    self._ipc_dsb.apply_to(mesh, bending_stiffness=entity.material.bending_stiffness)
            else:
                if self._ipc_stk is None:
                    self._ipc_stk = StableNeoHookean()
                    self._ipc_constitution_tabular.insert(self._ipc_stk)

                moduli = ElasticModuli.youngs_poisson(entity.material.E, entity.material.nu)
                self._ipc_stk.apply_to(mesh, moduli, mass_density=entity.material.rho)

            # ---- Per-environment: create IPC objects, then set per-env attrs on slot geometry ----
            for env_idx in range(self._B):
                fem_obj = self._ipc_objects.create(f"{solver_type}_{i_e}_{env_idx}")
                fem_geom_slot, _ = fem_obj.geometries().create(mesh)

                # All per-env writes go on the slot's own geometry (deep-copied)
                slot_geom = fem_geom_slot.geometry()
                if self._B > 1:
                    self._ipc_subscenes[env_idx].apply_to(slot_geom)
                slot_meta = slot_geom.meta()
                slot_meta.create("solver_type", solver_type)
                slot_meta.create("entity_idx", str(i_e))
                slot_meta.create("env_idx", str(env_idx))

    def _add_rigid_geoms_to_ipc(self) -> None:
        """Add rigid geoms to the IPC scene as ABD objects, merging geoms by link."""
        assert gs.logger is not None

        gs.logger.debug(f"Registered entity coupling types: {set(self._coup_type_by_entity.values())}")

        # ========== Pre-compute link groups (env-independent) ==========
        # Group links by fixed-joint merge target, matching mjcf.py behavior where geoms from fixed-joint children are
        # merged into the parent body's mesh.
        # target_link -> [source_links that merge into it via fixed joints]
        target_groups: dict["RigidLink", list["RigidLink"]] = {}
        merge_transforms: dict["RigidLink", tuple[np.ndarray, np.ndarray]] = {
            # source_link_idx -> (R, t) relative to target frame
        }
        for link in self.rigid_solver.links:
            entity = link.entity

            coup_type = self._coup_type_by_entity.get(entity)
            if coup_type is None:
                continue

            # Link filter for two_way_soft_constraint
            if coup_type == COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT:
                link_filter = self._coup_links.get(entity)
                if link_filter is not None and link not in link_filter:
                    continue

            target_link = find_target_link_for_fixed_merge(link)
            target_groups.setdefault(target_link, []).append(link)

            if target_link is not link:
                merge_transforms[link] = compute_link_to_link_transform(link, target_link)
                gs.logger.debug(f"Fixed-merge: link {link.idx} ({link.name}) -> {target_link.idx} ({target_link.name})")

        # ========== Process each link across environments ==========
        links_pos = qd_to_numpy(self.rigid_solver.links_state.pos, transpose=True)
        links_quat = qd_to_numpy(self.rigid_solver.links_state.quat, transpose=True)

        for target_link, source_links in target_groups.items():
            entity = target_link.entity
            entity_coup_type = self._coup_type_by_entity[entity]
            i_e = entity._idx_in_solver

            # ---- Collect geom meshes (env-independent local-frame geometry) ----
            meshes = []
            for source_link in source_links:
                for geom in source_link.geoms:
                    if geom.type == gs.GEOM_TYPE.PLANE:
                        local_normal = geom.data[:3].astype(np.float64, copy=False)
                        normal = gu.transform_by_quat(local_normal, geom.init_quat)
                        normal = normal / np.linalg.norm(normal)
                        height = np.dot(geom.init_pos, normal)
                        plane_geom = uipc.geometry.ground(height, normal)

                        if entity not in self._ipc_grounds_contact:
                            plane_contact = self._ipc_contact_tabular.create(f"ground_contact_{i_e}")
                            self._ipc_grounds_contact[entity] = plane_contact
                        self._ipc_grounds_contact[entity].apply_to(plane_geom)

                        for env_idx in range(self._B):
                            plane_obj = self._ipc_objects.create(f"rigid_plane_{geom.idx}_{env_idx}")
                            if self._B > 1:
                                self._ipc_subscenes[env_idx].apply_to(plane_geom)
                            plane_obj.geometries().create(plane_geom)
                    elif geom.n_verts:
                        # Apply geom transform to vertices
                        geom_verts = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)

                        # Apply additional transform for fixed joint merging
                        if source_link is not target_link:
                            geom_verts = gu.transform_by_trans_quat(geom_verts, *merge_transforms[source_link])

                        try:
                            mesh = uipc.geometry.trimesh(
                                geom_verts.astype(np.float64, copy=False),
                                geom.init_faces.astype(np.int32, copy=False),
                            )
                        except RuntimeError as e:
                            gs.raise_exception_from(f"Failed to process geom {geom.idx} for IPC.", e)

                        meshes.append(mesh)

            if not meshes:
                continue

            # ---- Merge meshes ----
            rigid_link_geom = meshes[0] if len(meshes) == 1 else uipc.geometry.merge(meshes)
            uipc.geometry.label_surface(rigid_link_geom)
            is_open_mesh = not uipc.geometry.is_trimesh_closed(rigid_link_geom)

            # Cache merged world-frame trimesh for env 0 (used by neutral overlap check)
            link_T_0 = gu.trans_quat_to_T(links_pos[0, target_link.idx], links_quat[0, target_link.idx])
            local_verts = np.asarray(rigid_link_geom.positions().view())[..., 0]
            world_verts = (link_T_0[:3, :3] @ local_verts.T).T + link_T_0[:3, 3]
            faces = rigid_link_geom.triangles().topo().view()[..., 0]
            # Shrink 0.1% toward centroid to match rigid collider's neutral overlap check
            centroid = world_verts.mean(axis=0, keepdims=True)
            world_verts = centroid + (1.0 - 1e-3) * (world_verts - centroid)
            self._abd_merged_meshes[target_link] = trimesh.Trimesh(vertices=world_verts, faces=faces, process=False)

            # ---- Determine coupling behavior ----
            is_ipc_only = entity_coup_type == COUPLING_TYPE.IPC_ONLY
            is_soft_constraint_target = entity_coup_type == COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT

            # ---- Apply constitutions (env-independent, once per link) ----

            # Apply per-link contact element or no-collision marker
            if self._coupling_collision_settings.get(entity, {}).get(target_link, True):
                if target_link not in self._ipc_abd_links_contact:
                    abd_contact = self._ipc_contact_tabular.create(f"abd_link_contact_{target_link.idx}")
                    self._ipc_abd_links_contact[target_link] = abd_contact
                self._ipc_abd_links_contact[target_link].apply_to(rigid_link_geom)
            else:
                self._ipc_no_collision_contact.apply_to(rigid_link_geom)

            # Apply ABD constitution
            if self._ipc_abd is None:
                self._ipc_abd = AffineBodyConstitution()
                self._ipc_constitution_tabular.insert(self._ipc_abd)
            self._ipc_abd.apply_to(rigid_link_geom, kappa=ABD_KAPPA * uipc.unit.MPa, mass_density=entity.material.rho)

            # Apply SoftTransformConstraint for coupled links
            if is_soft_constraint_target:
                if self._ipc_stc is None:
                    self._ipc_stc = SoftTransformConstraint()
                    self._ipc_constitution_tabular.insert(self._ipc_stc)

                constraint_strength = np.array(entity.material.coup_stiffness)
                self._ipc_stc.apply_to(rigid_link_geom, constraint_strength)

            # Set geometry attributes (env-independent)
            # external_kinetic: 1 = driven by rigid solver, 0 = IPC-only
            external_kinetic_attr = rigid_link_geom.instances().find(uipc.builtin.external_kinetic)
            uipc.view(external_kinetic_attr)[:] = int(not is_ipc_only)

            is_fixed_attr = rigid_link_geom.instances().find(uipc.builtin.is_fixed)
            uipc.view(is_fixed_attr)[:] = int(target_link.is_fixed)

            # ---- Per-environment: create IPC objects, then set per-env attrs on slot geometry ----
            abd_geom_slots: list[GeometrySlot] = []
            for env_idx in range(self._B):
                abd_obj = self._ipc_objects.create(f"rigid_link_{target_link.idx}_{env_idx}")
                abd_geom_slot, _ = abd_obj.geometries().create(rigid_link_geom)

                # All per-env writes go on the slot's own geometry (deep-copied)
                slot_geom = abd_geom_slot.geometry()
                uipc.view(slot_geom.transforms())[0] = gu.trans_quat_to_T(
                    links_pos[env_idx, target_link.idx], links_quat[env_idx, target_link.idx]
                )
                if self._B > 1:
                    self._ipc_subscenes[env_idx].apply_to(slot_geom)
                slot_meta = slot_geom.meta()
                slot_meta.create("solver_type", "rigid")
                slot_meta.create("link_idx", str(target_link.idx))
                slot_meta.create("env_idx", str(env_idx))
                abd_geom_slots.append(abd_geom_slot)

                # Register animator for coupled links (env-specific: needs abd_obj and env_idx)
                if is_soft_constraint_target:
                    self._ipc_animator.insert(
                        abd_obj, partial(self._animate_rigid_link, weakref.ref(self), target_link, env_idx)
                    )

            # ---- Store link data ----
            needs_ipc_state = is_ipc_only or is_soft_constraint_target
            self._abd_data_by_link[target_link] = ABDLinkData(
                slots=abd_geom_slots,
                aim_transforms=np.tile(np.eye(4, dtype=gs.np_float), (self._B, 1, 1)),
                ipc_transforms=np.tile(np.eye(4, dtype=gs.np_float), (self._B, 1, 1)) if needs_ipc_state else None,
                ipc_velocities=np.zeros((self._B, 4, 4), dtype=gs.np_float) if needs_ipc_state else None,
            )

    def _add_articulation_entities_to_ipc(self) -> None:
        """
        Add articulated robot entities to IPC using ExternalArticulationConstraint.

        This enables joint-level coupling between Genesis and IPC.
        """
        assert gs.logger is not None

        if COUPLING_TYPE.EXTERNAL_ARTICULATION not in self._coup_type_by_entity.values():
            return

        self._ipc_eac = ExternalArticulationConstraint()
        self._ipc_constitution_tabular.insert(self._ipc_eac)

        joints_xaxis = qd_to_numpy(self.rigid_solver.joints_state.xaxis, transpose=True)
        joints_xanchor = qd_to_numpy(self.rigid_solver.joints_state.xanchor, transpose=True)

        # Process each rigid entity with external_articulation coupling type
        for i_e, entity in enumerate(cast(list["RigidEntity"], self.rigid_solver.entities)):
            # Only process entities with external_articulation coupling type
            if self._coup_type_by_entity.get(entity) != COUPLING_TYPE.EXTERNAL_ARTICULATION:
                continue

            gs.logger.debug(f"Adding articulated entity {i_e} with {entity.n_joints} joints")

            # ---- Collect joint info (env-independent) ----
            joints: list[tuple["RigidJoint", type, bool, "RigidLink", "RigidLink"]] = []
            for joint in entity.joints:
                if joint.type == gs.JOINT_TYPE.FIXED:
                    continue
                elif joint.type == gs.constants.JOINT_TYPE.REVOLUTE:
                    joint_constitution = AffineBodyRevoluteJoint
                    reverse_verts = True
                elif joint.type == gs.constants.JOINT_TYPE.PRISMATIC:
                    joint_constitution = AffineBodyPrismaticJoint
                    reverse_verts = False
                else:
                    gs.raise_exception(f"Unsupported joint type: {joint.type}")

                child_link = joint.link
                parent_link = entity.links[max(joint.link.parent_idx, 0) - entity.link_start]
                parent_link = find_target_link_for_fixed_merge(parent_link)
                if parent_link not in self._abd_data_by_link or child_link not in self._abd_data_by_link:
                    gs.raise_exception(
                        "Rigid link has no collision geometry. Coupling type 'external_articulation' is not supported."
                    )
                joints.append((joint, joint_constitution, reverse_verts, parent_link, child_link))

            # ---- Create joint geometries per environment ----
            articulation_geom_slots: list[GeometrySlot] = []
            for env_idx in range(self._B):
                joint_geom_slots: list[GeometrySlot] = []
                for joint, joint_constitution, reverse_verts, parent_link, child_link in joints:
                    joint_axis = joints_xaxis[env_idx, joint.idx]
                    joint_pos = joints_xanchor[env_idx, joint.idx]

                    v1 = joint_pos - 0.5 * joint_axis
                    v2 = joint_pos + 0.5 * joint_axis
                    vertices = np.array([v2, v1] if reverse_verts else [v1, v2], dtype=np.float64)
                    edges = np.array([[0, 1]], dtype=np.int32)
                    joint_geom = uipc.geometry.linemesh(vertices, edges)
                    if self._B > 1:
                        self._ipc_subscenes[env_idx].apply_to(joint_geom)

                    parent_abd_slot = self._abd_data_by_link[parent_link].slots[env_idx]
                    child_abd_slot = self._abd_data_by_link[child_link].slots[env_idx]
                    joint_constitution().apply_to(
                        joint_geom, [parent_abd_slot], [0], [child_abd_slot], [0], [JOINT_STRENGTH_RATIO]
                    )

                    joint_obj = self._ipc_objects.create(f"joint_{joint.idx}_{env_idx}")
                    joint_geom_slot, _ = joint_obj.geometries().create(joint_geom)
                    joint_geom_slots.append(joint_geom_slot)

                articulation_geom = self._ipc_eac.create_geometry(joint_geom_slots, [0] * len(joint_geom_slots))
                if self._B > 1:
                    self._ipc_subscenes[env_idx].apply_to(articulation_geom)

                articulation_obj = self._ipc_objects.create(f"articulation_entity_{i_e}_{env_idx}")
                articulation_geom_slot, _ = articulation_obj.geometries().create(articulation_geom)
                articulation_geom_slots.append(articulation_geom_slot)

            # Store articulation data with pre-allocated per-step arrays
            n_joints = len(joints)
            self._articulation_data_by_entity[entity] = ArticulatedEntityData(
                slots=articulation_geom_slots,
                q_slice=slice(entity.q_start, entity.q_end),
                dof_slice=slice(entity.dof_start, entity.dof_end),
                joints_child_link=[j.link for j, *_ in joints],
                joints_qs_idx_local=[j.qs_idx_local[0] for j, *_ in joints],
                delta_theta_tilde=np.zeros((self._B, n_joints), dtype=np.float64),
                prev_qpos=np.zeros((self._B, entity.n_qs), dtype=np.float64),
                mass_matrix=np.zeros((self._B, entity.n_dofs, entity.n_dofs), dtype=np.float64),
                ipc_qpos=np.zeros((self._B, entity.n_qs), dtype=gs.np_float),
            )

            gs.logger.debug(f"Successfully added articulated rigid entity {i_e} to IPC.")

    def _register_contact_pairs(self) -> None:
        """Register pairwise contact models for all contact elements.

        Friction is combined by geometric mean, resistance by harmonic mean (series spring).
        Rigid link self-collision filtering mirrors the RigidSolver collider:
        ``enable_self_collision``, ``enable_adjacent_collision``, ``enable_neutral_collision``.
        """
        from genesis.engine.solvers.rigid.collider.collider import are_links_adjacent

        assert gs.logger is not None

        enable_self_collision = self.rigid_solver._enable_self_collision
        enable_adjacent_collision = self.rigid_solver._enable_adjacent_collision
        enable_neutral_collision = self.rigid_solver._enable_neutral_collision

        # Collect non-ABD contact infos (FEM, cloth)
        non_abd_infos: list[tuple[ContactElement, float, float]] = []
        for entity, elem in (*self._ipc_clothes_contact.items(), *self._ipc_fems_contact.items()):
            friction = entity.material.friction_mu
            resistance = entity.material.contact_resistance or self.options.contact_resistance
            non_abd_infos.append((elem, friction, resistance))

        # Collect ABD link contact infos
        abd_link_infos: list[tuple[ContactElement, "RigidLink", float, float]] = []
        for link, elem in self._ipc_abd_links_contact.items():
            friction = link.entity.material.coup_friction
            resistance = link.entity.material.contact_resistance or self.options.contact_resistance
            abd_link_infos.append((elem, link, friction, resistance))

        # ---- Non-ABD × Non-ABD pairs ----
        for i, (elem_i, friction_i, resistance_i) in enumerate(non_abd_infos):
            for elem_j, friction_j, resistance_j in non_abd_infos[i:]:
                self._ipc_contact_tabular.insert(
                    elem_i,
                    elem_j,
                    geometric_mean(friction_i, friction_j),
                    harmonic_mean(resistance_i, resistance_j),
                    True,
                )

        # ---- Non-ABD × ABD link pairs ----
        for elem_na, friction_na, resistance_na in non_abd_infos:
            for elem_abd, _, friction_abd, resistance_abd in abd_link_infos:
                self._ipc_contact_tabular.insert(
                    elem_na,
                    elem_abd,
                    geometric_mean(friction_na, friction_abd),
                    harmonic_mean(resistance_na, resistance_abd),
                    True,
                )

        # ---- ABD link × ABD link pairs (with self-collision filtering) ----
        for i, (elem_i, link_i, friction_i, resistance_i) in enumerate(abd_link_infos):
            for elem_j, link_j, friction_j, resistance_j in abd_link_infos[i:]:
                friction_ij = geometric_mean(friction_i, friction_j)
                resistance_ij = harmonic_mean(resistance_i, resistance_j)

                if not self.options.enable_rigid_rigid_contact:
                    self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, False)
                    continue

                # Same-entity self-collision filtering (mirrors RigidSolver collider)
                if link_i.entity is link_j.entity and link_i is not link_j:
                    if not enable_self_collision:
                        gs.logger.debug(
                            f"IPC: disable self-coll {link_i.name} <-> {link_j.name} (self_collision=False)"
                        )
                        self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, False)
                        continue
                    if not enable_adjacent_collision and are_links_adjacent(link_i, link_j):
                        gs.logger.debug(f"IPC: disable adjacent-coll {link_i.name} <-> {link_j.name}")
                        self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, False)
                        continue
                    mesh_i = self._abd_merged_meshes.get(link_i)
                    mesh_j = self._abd_merged_meshes.get(link_j)
                    if (
                        not enable_neutral_collision
                        and mesh_i is not None
                        and mesh_j is not None
                        and are_meshes_overlapping(mesh_i, mesh_j)
                    ):
                        gs.logger.debug(f"IPC: disable neutral-coll {link_i.name} <-> {link_j.name}")
                        self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, False)
                        continue

                self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, True)

        # ---- All contact elements (for ground and no-collision registration) ----
        all_contact_infos: list[tuple[ContactElement, float, float, bool]] = []
        for elem, friction, resistance in non_abd_infos:
            all_contact_infos.append((elem, friction, resistance, False))
        for elem, _, friction, resistance in abd_link_infos:
            all_contact_infos.append((elem, friction, resistance, True))

        # Register per-plane ground contact pairs
        for entity, ground_elem in self._ipc_grounds_contact.items():
            plane_friction = entity.material.coup_friction
            plane_resistance = entity.material.contact_resistance or self.options.contact_resistance
            for elem, friction, resistance, is_abd in all_contact_infos:
                friction_ground = geometric_mean(friction, plane_friction)
                resistance_ground = harmonic_mean(resistance, plane_resistance)
                enabled = not is_abd or self.options.enable_rigid_ground_contact
                self._ipc_contact_tabular.insert(ground_elem, elem, friction_ground, resistance_ground, enabled)
            self._ipc_contact_tabular.insert(self._ipc_no_collision_contact, ground_elem, 0.0, 0.0, False)

        # Register no_collision pairs (always disabled)
        for elem, *_ in all_contact_infos:
            self._ipc_contact_tabular.insert(self._ipc_no_collision_contact, elem, 0.0, 0.0, False)
        self._ipc_contact_tabular.insert(
            self._ipc_no_collision_contact, self._ipc_no_collision_contact, 0.0, 0.0, False
        )

    def _finalize_ipc(self):
        """Finalize IPC setup and initialize AffineBodyStateAccessorFeature"""
        assert gs.logger is not None
        assert self._ipc_world is not None
        self._ipc_world.init(self._ipc_scene)
        # Checkpoint frame 0 so that recover(0) works in reset().
        self._ipc_world.dump()
        gs.logger.info("IPC world initialized successfully")

    def _init_accessors(self):
        assert gs.logger is not None
        assert self._ipc_world is not None

        # No ABD bodies, feature not needed
        if not self._abd_data_by_link:
            return

        abd_links = list(self._abd_data_by_link.keys())
        n_abd_links = len(abd_links)

        self._abd_state_feature = cast(
            AffineBodyStateAccessorFeature, self._ipc_world.features().find(AffineBodyStateAccessorFeature)
        )
        body_count = self._abd_state_feature.body_count()

        # Verify the count matches IPC's ABD body count
        if body_count != n_abd_links * self._B:
            gs.raise_exception(f"ABD body count mismatch: got {body_count}.")

        # Create state geometry for batch data transfer
        self._abd_state_geom = self._abd_state_feature.create_geometry()
        self._abd_state_geom.instances().create(uipc.builtin.transform, np.eye(4, dtype=np.float64))
        self._abd_state_geom.instances().create(uipc.builtin.velocity, np.zeros((4, 4), dtype=np.float64))

    def _init_ipc_gui(self):
        """Initialize polyscope-based IPC GUI viewer."""
        try:
            if not ps.is_initialized():
                ps.init()
            self._ipc_gui = SceneGUI(self._ipc_scene, "split")
            # Also sets up_dir and ground_plane_height from scene
            self._ipc_gui.register()

            # Match polyscope camera to Genesis viewer options
            viewer_opts = self.sim.scene.viewer_options
            if viewer_opts is not None:
                cam_pos = np.asarray(viewer_opts.camera_pos, dtype=np.float64)
                cam_lookat = np.asarray(viewer_opts.camera_lookat, dtype=np.float64)
                ps.look_at(cam_pos, cam_lookat)

            ps.show(forFrames=1)
            gs.logger.info("IPC GUI initialized successfully")
        except Exception as e:
            gs.logger.warning(f"IPC GUI unavailable: {e}. Continuing without IPC GUI.")
            self._ipc_gui = None

    # ============================================================
    # Section 2: Core implementation
    # ============================================================

    def preprocess(self, f):
        """Preprocessing step before coupling"""
        pass

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
        assert self._ipc_world is not None

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
        self._retrieve_ipc_fem_states()
        self._retrieve_ipc_rigid_states()

        # Step 5: Post-advance — write IPC-resolved state to qpos
        self._post_advance_write_qpos()

        # Step 6: Update GUI if enabled
        if self._ipc_gui is not None:
            ps.frame_tick()
            self._ipc_gui.update()

    def couple_grad(self, f):
        """Gradient computation for coupling"""
        gs.raise_exception("couple_grad is not available for IPCCoupler. Please use LegacyCoupler instead.")

    def reset(self, envs_idx=None):
        """Reset coupling state. Per-env reset is not supported by libuipc; envs_idx must cover all envs."""
        assert gs.logger is not None
        assert self._ipc_world is not None
        if envs_idx is not None:
            all_envs = set(range(max(self._B, 1)))
            envs_set = set(int(x) for x in envs_idx) if hasattr(envs_idx, "__iter__") else {int(envs_idx)}
            assert envs_set == all_envs, f"IPC coupler only supports full reset, got envs_idx={envs_idx}"

        gs.logger.debug("Resetting IPC coupler state")
        self._is_abd_updated = False
        self._ipc_world.recover(0)
        self._ipc_world.retrieve()

    def mark_is_abd_updated(self):
        """Mark all coupled entities as needing IPC ABD state sync after position changes."""
        self._is_abd_updated = True

    def cache_pre_prediction_transforms(self):
        """
        Sync IPC ABD body transforms from current (pre-prediction) link poses.

        Called by RigidSolver before kernel_predict_integrate. At this point
        links_state reflects actual poses (including any set_qpos changes) before
        prediction overwrites them. ABD bodies are set to these poses so IPC can
        resolve collisions on the path toward the predicted target.
        """
        if not self._is_abd_updated or self._abd_state_feature is None:
            return

        assert self._abd_state_geom is not None

        links_pos = qd_to_numpy(self.rigid_solver.links_state.pos, transpose=True)
        links_quat = qd_to_numpy(self.rigid_solver.links_state.quat, transpose=True)
        links_transform = gu.trans_quat_to_T(links_pos, links_quat)

        self._abd_state_feature.copy_to(self._abd_state_geom)
        trans_attr = self._abd_state_geom.instances().find(uipc.builtin.transform)
        transforms = trans_attr.view()

        for i_link, link in enumerate(self._abd_data_by_link.keys()):
            for env_idx in range(self._B):
                abd_body_idx = i_link * self._B + env_idx
                transforms[abd_body_idx] = links_transform[env_idx, link.idx]

        self._abd_state_feature.copy_from(self._abd_state_geom)
        self._is_abd_updated = False

    @property
    def is_active(self) -> bool:
        """Check if IPC coupling is active"""
        return self._ipc_world is not None

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
        return bool(self._coup_type_by_entity)

    # ============================================================
    # Section 3: Helpers
    # ============================================================
    @staticmethod
    def _animate_rigid_link(coupler_ref, link, env_idx, info):
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

        # Enable constraint and set target transform (q_genesis^n)
        is_constrained_attr = geom.instances().find(uipc.builtin.is_constrained)
        aim_transform_attr = geom.instances().find(uipc.builtin.aim_transform)
        assert is_constrained_attr and aim_transform_attr
        uipc.view(is_constrained_attr)[0] = 1
        uipc.view(aim_transform_attr)[:] = coupler._abd_data_by_link[link].aim_transforms[env_idx]

    def _retrieve_ipc_fem_states(self):
        # IPC world advance/retrieve is handled at Scene level
        # This method handles both volumetric FEM (3D) and cloth (2D) post-processing

        if not self.fem_solver.is_active:
            return

        # Gather FEM states (both volumetric and cloth) using metadata filtering
        visitor = SceneVisitor(self._ipc_scene)

        # Collect FEM and cloth geometries using metadata
        fem_entities = cast(list["FEMEntity"], self.fem_solver.entities)
        fem_positions_by_entity: dict["FEMEntity", list[np.ndarray]] = {
            entity: [np.array([]) for _ in range(self._B)] for entity in fem_entities
        }
        for fem_geom_slot in visitor.geometries():
            if not isinstance(fem_geom_slot, SimplicialComplexSlot):
                continue

            fem_geom = fem_geom_slot.geometry()
            if fem_geom.dim() not in (2, 3):
                continue
            meta = read_ipc_geometry_metadata(fem_geom)
            if meta is None:
                continue
            solver_type, env_idx, i_e = meta
            if solver_type not in ("fem", "cloth"):
                continue

            entity = cast("FEMEntity", self.fem_solver.entities[i_e])
            (transformed_geom,) = uipc.geometry.apply_transform(fem_geom)
            fem_positions_by_entity[entity][env_idx] = transformed_geom.positions().view().reshape(-1, 3)

        # Update FEM entities using filtered geometries
        for entity, geom_positions in fem_positions_by_entity.items():
            geom_positions = np.stack(geom_positions, axis=0, dtype=gs.np_float)
            entity.set_pos(0, geom_positions)

    def _retrieve_ipc_rigid_states(self):
        """
        Retrieve ABD transforms/affine matrices after IPC step using AffineBodyStateAccessorFeature.

        O(num_rigid_bodies) instead of O(total_geometries).
        Also populates data arrays for force computation.
        """
        if self._abd_state_feature is None:
            return

        # Single batch copy of ALL ABD states from IPC
        assert self._abd_state_geom is not None
        self._abd_state_feature.copy_to(self._abd_state_geom)

        # Get all transforms at once (array view)
        trans_attr = self._abd_state_geom.instances().find(uipc.builtin.transform)
        # Shape: (num_bodies, 4, 4)
        transforms = trans_attr.view()

        # Get velocities (4x4 matrix representing transform derivative)
        vel_attr = self._abd_state_geom.instances().find(uipc.builtin.velocity)
        # Shape: (num_bodies, 4, 4)
        velocities = vel_attr.view()

        for i_link, (link, abd_data) in enumerate(self._abd_data_by_link.items()):
            if abd_data.ipc_transforms is None:
                continue
            for env_idx in range(self._B):
                abd_body_idx = i_link * self._B + env_idx
                abd_data.ipc_transforms[env_idx] = transforms[abd_body_idx]
                abd_data.ipc_velocities[env_idx] = velocities[abd_body_idx]

    def _store_gs_rigid_states(self):
        """
        Store predicted Genesis rigid body states before IPC advance.

        After kernel_predict_integrate + FK, qpos and links_state contain predicted values.
        These are cached so that _pre_advance/_post_advance methods don't need to
        reach back into the rigid solver for reads.

        Note: IPC-only entities have no animator (external_kinetic=0), so stored
        transforms are unused by IPC for them.
        """
        if not self.rigid_solver.is_active:
            return

        # Cache per-entity qpos slices for external articulation
        if self._articulation_data_by_entity:
            qpos = qd_to_numpy(self.rigid_solver.qpos, transpose=True)
            qpos_prev = qd_to_numpy(self.rigid_solver.qpos_prev, transpose=True)
            mass_matrix = qd_to_numpy(self.rigid_solver.mass_mat, transpose=True)

            for ad in self._articulation_data_by_entity.values():
                entity_qpos = qpos[..., ad.q_slice]
                entity_qpos_prev = qpos_prev[..., ad.q_slice]
                ad.delta_theta_tilde[:] = (
                    entity_qpos[..., ad.joints_qs_idx_local] - entity_qpos_prev[..., ad.joints_qs_idx_local]
                )
                ad.prev_qpos[:] = entity_qpos_prev
                ad.mass_matrix[:] = mass_matrix[:, ad.dof_slice, ad.dof_slice]

        # Store transforms for all rigid links
        links_pos = qd_to_numpy(self.rigid_solver.links_state.pos, transpose=True)
        links_quat = qd_to_numpy(self.rigid_solver.links_state.quat, transpose=True)
        links_transform = gu.trans_quat_to_T(links_pos, links_quat)
        for link, abd_data in self._abd_data_by_link.items():
            abd_data.aim_transforms[:] = links_transform[:, link.idx]

    def _pre_advance_external_articulation(self):
        """
        Pre-advance processing for external_articulation entities.
        Prepares articulation data and updates IPC geometry before advance().
        """
        if COUPLING_TYPE.EXTERNAL_ARTICULATION not in self._entities_by_coup_type:
            return

        for ad in self._articulation_data_by_entity.values():
            # Update IPC geometry for each articulated entity
            for env_idx in range(self._B):
                articulation_geom = ad.slots[env_idx].geometry()

                delta_theta_tilde_attr = articulation_geom["joint"].find("delta_theta_tilde")
                uipc.view(delta_theta_tilde_attr)[:] = ad.delta_theta_tilde[env_idx]

                mass_matrix_attr = articulation_geom["joint_joint"].find("mass")
                uipc.view(mass_matrix_attr).flat[:] = ad.mass_matrix[env_idx]

    def _post_advance_write_qpos(self):
        """
        Write IPC-resolved state back into rigid_global_info.qpos (predicted).

        kernel_restore_integrate will back-compute velocity/acceleration for step_2 to land
        on IPC's target positions.

        For two_way_soft_constraint, non-fixed base links get their IPC-resolved transform
        written to qpos[0:7], and child link joint angles are back-computed from IPC transforms.
        For external_articulation (fixed base only), joint qpos comes from IPC delta_theta.
        """
        if not self._coup_type_by_entity:
            return

        qpos_tc = qd_to_torch(self.rigid_solver.qpos, transpose=True, copy=False)

        # ---- Step 1: Non-fixed base links — write IPC transform to qpos[0:7] ----
        for link, abd_data in self._abd_data_by_link.items():
            if abd_data.ipc_transforms is None:
                continue
            entity = link.entity
            if link is not entity.base_link or entity.base_link.is_fixed:
                continue

            q_start = entity.q_start
            envs_qpos = np.empty((self._B, 7), dtype=gs.np_float)
            for env_idx in range(self._B):
                envs_qpos[env_idx, :3], envs_qpos[env_idx, 3:7] = gu.T_to_trans_quat(abd_data.ipc_transforms[env_idx])
            qpos_tc[:, q_start : q_start + 7] = torch.from_numpy(envs_qpos).to(qpos_tc.device)

        # ---- Step 2a: Two-way child links — back-compute joint angles from IPC transforms ----
        if COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT in self._entities_by_coup_type:
            qpos0 = qd_to_numpy(self.rigid_solver.qpos0, transpose=True)
            links_pos = qd_to_numpy(self.rigid_solver.links_state.pos, transpose=True)
            links_quat = qd_to_numpy(self.rigid_solver.links_state.quat, transpose=True)

            for link, abd_data in self._abd_data_by_link.items():
                if abd_data.ipc_transforms is None:
                    continue
                entity = link.entity
                if self._coup_type_by_entity.get(entity) != COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT:
                    continue
                if link is entity.base_link or link.parent_idx == -1:
                    continue

                parent_link = entity.links[link.parent_idx - entity.link_start]
                joint = link.joints[0]
                if joint.type not in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                    continue
                q_idx = joint.q_start
                envs_q = np.empty((self._B, 1), dtype=gs.np_float)
                for env_idx in range(self._B):
                    parent_abd = self._abd_data_by_link.get(parent_link)
                    if parent_abd is not None and parent_abd.ipc_transforms is not None:
                        parent_T = parent_abd.ipc_transforms[env_idx]
                        parent_quat = gu.T_to_trans_quat(parent_T)[1]
                    else:
                        parent_T = gu.trans_quat_to_T(
                            links_pos[env_idx, parent_link.idx], links_quat[env_idx, parent_link.idx]
                        )
                        parent_quat = links_quat[env_idx, parent_link.idx]
                    child_T = abd_data.ipc_transforms[env_idx]
                    child_quat_pre = gu.transform_quat_by_quat(
                        np.asarray(link.quat, dtype=parent_quat.dtype), parent_quat
                    )
                    if joint.type == gs.JOINT_TYPE.REVOLUTE:
                        child_quat = gu.T_to_trans_quat(child_T)[1]
                        qloc = gu.transform_quat_by_quat(child_quat, gu.inv_quat(child_quat_pre))
                        rotvec = gu.quat_to_rotvec(qloc)
                        axis = np.asarray(joint._dofs_motion_ang[0], dtype=rotvec.dtype)
                        angle_ipc = float(np.dot(rotvec, axis))
                    else:  # PRISMATIC
                        child_pos = child_T[:3, 3]
                        parent_pos = parent_T[:3, 3]
                        link_offset_pos = np.asarray(link.pos, dtype=parent_pos.dtype)
                        pos_pre = parent_pos + gu.transform_by_quat(link_offset_pos, parent_quat)
                        axis = np.asarray(joint._dofs_motion_vel[0], dtype=pos_pre.dtype)
                        xaxis = gu.transform_by_quat(axis, child_quat_pre)
                        angle_ipc = float(np.dot(child_pos - pos_pre, xaxis))
                    envs_q[env_idx, 0] = qpos0[env_idx, q_idx] + angle_ipc
                qpos_tc[:, q_idx : q_idx + 1] = torch.from_numpy(envs_q).to(qpos_tc.device)

        # ---- Step 2b: External articulation — read delta_theta, write joint qpos ----
        for ad in self._articulation_data_by_entity.values():
            delta_theta_ipc = np.empty((self._B, len(ad.joints_qs_idx_local)), dtype=np.float64)
            for env_idx in range(self._B):
                articulation_geom = ad.slots[env_idx].geometry()
                delta_theta_attr = articulation_geom["joint"].find("delta_theta")
                delta_theta_ipc[env_idx] = delta_theta_attr.view()

            np.copyto(ad.ipc_qpos, ad.prev_qpos, casting="same_kind")
            ad.ipc_qpos[..., ad.joints_qs_idx_local] += delta_theta_ipc
            # Base link qpos[0:7] already handled in Step 1 for non-fixed base;
            # only write joint DOFs here.
            global_qs = [ad.q_slice.start + qi for qi in ad.joints_qs_idx_local]
            qpos_tc[:, global_qs] = torch.from_numpy(ad.ipc_qpos[..., ad.joints_qs_idx_local]).to(qpos_tc.device)
