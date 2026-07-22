import inspect
import math
import os
from itertools import chain
from typing import TYPE_CHECKING, Literal, Any, Hashable
from functools import wraps

import quadrants as qd
import numpy as np
import torch
import trimesh

import genesis as gs
from genesis.engine.materials.base import Material
from genesis.engine.mesh import InertialProperties
from genesis.options.morphs import Morph
from genesis.options.surfaces import Surface
from genesis.utils import array_class
from genesis.utils import geom as gu
from genesis.utils import mesh as mu
from genesis.utils import mjcf as mju
from genesis.utils import terrain as tu
from genesis.utils import urdf as uu
from genesis.utils.misc import DeprecationError, broadcast_tensor, qd_to_numpy, qd_to_torch, tensor_to_array
from genesis.typing import UnitVec4FType, Vec3FType
from genesis.engine.states.entities import RigidEntityState

from ..base_entity import Entity
from .rigid_equality import RigidEquality
from .rigid_geom import RigidGeom
from .rigid_joint import RigidJoint
from .rigid_link import (
    RHO_MUJOCO,
    RHO_OBJECT,
    RHO_ROBOT,
    GeomInertialInfo,
    KinematicLink,
    LinkInertial,
    LinkInertialInfo,
    RigidLink,
    compose_inertial_from_g_infos,
    compose_inertial_properties,
    finalize_inertial,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
    from genesis.engine.solvers.kinematic_solver import KinematicSolver


# Wrapper to track the arguments of a function and save them in the target buffer
def tracked(fun):
    sig = inspect.signature(fun)

    @wraps(fun)
    def wrapper(self, *args, **kwargs):
        if self._update_tgt_while_set:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            args_dict = dict(tuple(bound.arguments.items())[1:])
            # Key the slot by (method, dofs subset, envs subset) so same-step calls on distinct subsets (e.g. arm
            # and gripper force control, or per-environment-group commands) each keep their own entry and gradient
            # path; a coarser key would let the second call evict the first from the tape. Slices key directly when
            # hashable (Python 3.12 onward) and resolve against their dimension size otherwise.
            key = [fun.__name__]
            for indices, n in (
                (args_dict.get("dofs_idx_local"), self.n_dofs),
                (args_dict.get("envs_idx"), self._solver._B),
            ):
                if indices is None:
                    subset = None
                elif isinstance(indices, slice):
                    subset = indices if isinstance(indices, Hashable) else tuple(range(*indices.indices(n)))
                elif isinstance(indices, torch.Tensor):
                    subset = tuple(tensor_to_array(indices).reshape(-1).tolist())
                else:
                    subset = tuple(np.asarray(indices).reshape(-1).tolist())
                key.append(subset)
            self._update_tgt(tuple(key), args_dict)
        return fun(self, *args, **kwargs)

    return wrapper


@qd.data_oriented
class KinematicEntity(Entity):
    """
    Base entity class for articulated rigid-body systems (morphology, FK, Jacobian, IK).

    Used directly by KinematicSolver for visualization-only kinematic entities,
    and subclassed by RigidEntity as a type marker for physics-enabled entities.
    """

    # override typing
    _solver: "KinematicSolver"

    def __init__(
        self,
        scene: "Scene",
        solver: "KinematicSolver",
        material: Material,
        morph: Morph,
        surface: Surface,
        idx: int,
        idx_in_solver,
        link_start: int,
        joint_start: int,
        q_start: int,
        dof_start: int,
        vgeom_start: int,
        vvert_start: int,
        vface_start: int,
        custom_vvert_start: int,
        custom_vface_start: int,
        morph_heterogeneous: list[Morph] | None = None,
        name: str | None = None,
    ):
        # Set heterogeneous support before super().__init__() because _get_morph_identifier() needs it
        self._morph_heterogeneous = morph_heterogeneous if morph_heterogeneous is not None else []
        self._enable_heterogeneous = bool(self._morph_heterogeneous)

        super().__init__(idx, scene, morph, solver, material, surface, name=name)

        self._idx_in_solver = idx_in_solver
        self._link_start: int = link_start
        self._joint_start: int = joint_start
        self._q_start = q_start
        self._dof_start = dof_start
        self._vgeom_start = vgeom_start
        self._vvert_start = vvert_start
        self._vface_start = vface_start
        self._custom_vvert_start = custom_vvert_start
        self._custom_vface_start = custom_vface_start

        self._is_built: bool = False
        self._is_attached: bool = False
        self._variant_init_qpos: list[np.ndarray] | None = None

        # Per-link world<-user pose offset (morph 'offset_pos'/'offset_quat' composed with the link's inertial
        # alignment), keyed by local link index; only root links carry one (children stay identity), so a multi-root
        # entity keeps a distinct offset per root. The base link's is mirrored into '_offset_pos'/'_offset_quat' for the
        # heterogeneous-variant seed; the solver gathers them into per-link tensors at build.
        self._links_offset_pos: dict[int, np.ndarray] = {}
        self._links_offset_quat: dict[int, np.ndarray] = {}
        self._offset_pos = np.array(self._morph.offset_pos, dtype=gs.np_float)
        self._offset_quat = np.array(self._morph.offset_quat, dtype=gs.np_float)

        # Transient per-link, per-variant load-time inertial info (see 'LinkInertialInfo'), computed at load from
        # the (then-available) parsed geometry, consumed by 'RigidLink._build' (geometry hint) and the align anchor.
        # Dropped after '_align_free_roots'; never persisted, since the geometry it derives from is irrelevant to a
        # kinematic entity once built. Indexed by link-local position (link.idx - link_start), matching the link order
        # and the '_links_offset_*' arrays: '_align_link' appends one entry per link in build order, the heterogeneous
        # loop appends variants.
        self._links_inertial_info: list[list[LinkInertialInfo]] = []

        # Per-variant base-link offset for heterogeneous entities, primary first and aligned with '_variant_init_qpos'
        self._variant_offset_pos: list[np.ndarray] | None = None
        self._variant_offset_quat: list[np.ndarray] | None = None

        self._load_model()

        # Initialize target variables and checkpoint
        self._tgt_keys = (
            "set_pos",
            "set_quat",
            "set_dofs_velocity",
            "control_dofs_force",
            "control_dofs_velocity",
            "control_dofs_position",
            "control_dofs_position_velocity",
        )
        self._tgt = dict()
        self._tgt_buffer = list()
        self._ckpt = dict()
        self._update_tgt_while_set = self._solver._requires_grad

    def _update_tgt(self, key, value):
        # Set [self._tgt] value while keeping the insertion order between keys. When a new key is inserted or an existing
        # key is updated, the new element should be inserted at the end of the dict. This is because we need to keep
        # the insertion order to correctly pass the gradients in the backward pass.
        self._tgt.pop(key, None)
        self._tgt[key] = value

    def init_ckpt(self):
        pass

    def _load_morph(self, morph: Morph):
        """Load a single morph into the entity."""
        if isinstance(morph, gs.morphs.Mesh):
            self._load_mesh(morph, self._surface)
        elif isinstance(morph, (gs.morphs.MJCF, gs.morphs.URDF, gs.morphs.Drone, gs.morphs.USD)):
            self._load_scene(morph, self._surface)
        elif isinstance(morph, gs.morphs.Primitive):
            self._load_primitive(morph, self._surface)
        elif isinstance(morph, gs.morphs.Terrain):
            self._load_terrain(morph, self._surface)
        else:
            gs.raise_exception(f"Unsupported morph: {morph}.")

        # Load heterogeneous variants (if any)
        self._load_heterogeneous_morphs()

    def _load_heterogeneous_morphs(self):
        """Load heterogeneous morphs (additional geometry variants for parallel environments).

        Each variant is loaded as additional geoms/vgeoms attached to links.
        Variant tracking (geom/vgeom ranges, inertial) is stored on the Link itself.
        """
        if not self._enable_heterogeneous:
            return

        # The per-variant offset and inertial alignment are tracked for a single root only; a multi-root entity (one
        # MJCF/URDF with several free root bodies) would cross-contaminate the roots' offsets and init poses.
        if sum(link.parent_idx == -1 for link in self._links) > 1:
            gs.raise_exception("Heterogeneous morphs are not supported on multi-root entities.")

        # Init variant tracking on ALL links
        for link in self._links:
            link._init_variant_tracking()

        # Track per-variant init_qpos and base-link offset for per-environment dispatch (primary first). Each variant
        # accumulates its own below so an asymmetric variant is aligned exactly like its homogeneous equivalent.
        self._variant_init_qpos = [self.init_qpos]
        self._variant_offset_pos = [self._offset_pos]
        self._variant_offset_quat = [self._offset_quat]

        n_links = len(self._links)

        # Load additional heterogeneous variants
        for morph in self._morph_heterogeneous:
            if isinstance(morph, (gs.morphs.URDF, gs.morphs.MJCF)):
                # Parse variant scene file
                morph._enable_mujoco_compatibility = self._morph._enable_mujoco_compatibility
                v_l_infos, v_links_j_infos, v_links_g_infos, _ = self._parse_scene(morph, self._surface)

                # Validate that the variant has the same joint structure as the primary
                if len(v_l_infos) != n_links:
                    gs.raise_exception(
                        f"Heterogeneous variant has {len(v_l_infos)} links, "
                        f"but primary has {n_links}. All variants must have the same link count."
                    )
                for i_l, (link, v_j_infos) in enumerate(zip(self._links, v_links_j_infos)):
                    primary_joints = link.joints
                    if len(v_j_infos) != len(primary_joints):
                        gs.raise_exception(
                            f"Heterogeneous variant link {i_l} has {len(v_j_infos)} joints, "
                            f"but primary has {len(primary_joints)}."
                        )
                    for p_joint, v_j_info in zip(primary_joints, v_j_infos):
                        if p_joint.name != v_j_info["name"]:
                            gs.raise_exception(
                                f"Joint name mismatch at link {i_l}: primary has '{p_joint.name}', "
                                f"variant has '{v_j_info['name']}'. All variants must have the same joint names."
                            )
                        if p_joint.type != v_j_info["type"]:
                            gs.raise_exception(
                                f"Joint type mismatch for '{p_joint.name}': primary has {p_joint.type}, "
                                f"variant has {v_j_info['type']}."
                            )
                        if p_joint.n_dofs != v_j_info["n_dofs"]:
                            gs.raise_exception(
                                f"DoF count mismatch for joint '{p_joint.name}': primary has {p_joint.n_dofs}, "
                                f"variant has {v_j_info['n_dofs']}."
                            )

                # Post-process each link's geoms. The COM/principal-axis anchoring of the floating base is deferred to
                # '_align_free_roots' after build (where the finalized per-variant composite inertia is known); here
                # only the morph pose offset is composed into the variant's init_qpos and offset.
                offset_pos = np.array(morph.offset_pos, dtype=gs.np_float)
                offset_quat = np.array(morph.offset_quat, dtype=gs.np_float)
                cg_vg_infos = []
                for v_l_info, v_j_infos, v_g_infos in zip(v_l_infos, v_links_j_infos, v_links_g_infos):
                    is_robot = v_l_info.get("is_robot", np.array(False, dtype=np.bool_))
                    cg_infos, vg_infos = self._postprocess_geoms_info(morph, v_g_infos, is_robot)
                    cg_vg_infos.append((cg_infos, vg_infos))

                # Extract variant's init_qpos from parsed joint infos, composing the morph offset into the free joint so
                # relative getters report the variant's user frame.
                variant_init_qpos_parts = []
                for v_l_info, v_j_infos in zip(v_l_infos, v_links_j_infos):
                    is_root = v_l_info["parent_idx"] == -1
                    for j_info in v_j_infos:
                        qpos = j_info["init_qpos"]
                        if is_root and j_info["type"] == gs.JOINT_TYPE.FREE:
                            init_pos, init_quat = gu.transform_pos_quat_by_trans_quat(
                                np.array(morph.offset_pos, dtype=gs.np_float),
                                np.array(morph.offset_quat, dtype=gs.np_float),
                                qpos[:3],
                                qpos[3:7],
                            )
                            qpos = np.concatenate([init_pos, init_quat])
                        variant_init_qpos_parts.append(qpos)
                if variant_init_qpos_parts:
                    self._variant_init_qpos.append(np.concatenate(variant_init_qpos_parts))
                else:
                    self._variant_init_qpos.append(np.array([]))
                self._variant_offset_pos.append(offset_pos)
                self._variant_offset_quat.append(offset_quat)

                # Add geoms per link and stash this variant's finalized inertia for the align anchor.
                recompute = morph.recompute_inertia
                for i_link, (link, v_l_info, (cg_infos, vg_infos)) in enumerate(
                    zip(self._links, v_l_infos, cg_vg_infos)
                ):
                    self._add_heterogeneous_variant(link, cg_infos, vg_infos)
                    self._on_heterogeneous_scene_variant_loaded(link, morph, v_l_info)
                    self._links_inertial_info[i_link].append(
                        self._finalize_inertial(
                            None if recompute else v_l_info.get("inertial_mass"),
                            None if recompute else v_l_info.get("inertial_pos"),
                            None if recompute else v_l_info.get("inertial_quat"),
                            None if recompute else v_l_info.get("inertial_i"),
                            cg_infos,
                            vg_infos,
                            v_l_info.get("is_robot", False),
                        )
                    )

            elif isinstance(morph, (gs.morphs.Mesh, gs.morphs.Primitive)):
                if isinstance(morph, gs.morphs.Mesh):
                    g_infos = self._load_mesh(morph, self._surface, load_geom_only_for_heterogeneous=True)
                else:
                    g_infos = self._load_primitive(morph, self._surface, load_geom_only_for_heterogeneous=True)
                if morph.fixed != self._morph.fixed:
                    gs.raise_exception("Mixing fixed and non-fixed morphs in heterogeneous entities is not supported.")
                cg_infos, vg_infos = self._postprocess_geoms_info(morph, g_infos, is_robot=False)

                # The COM/principal-axis anchoring is deferred to '_align_free_roots' after build; compose only the
                # morph pose offset here.
                offset_pos = np.array(morph.offset_pos, dtype=gs.np_float)
                offset_quat = np.array(morph.offset_quat, dtype=gs.np_float)

                self._add_heterogeneous_variant(self._links[0], cg_infos, vg_infos)
                # Mesh/Primitive variants have no explicit inertial; the anchor inertia comes from their geometry.
                self._links_inertial_info[0].append(
                    self._finalize_inertial(None, None, None, None, cg_infos, vg_infos, is_robot=False)
                )

                if morph.fixed:
                    init_qpos = np.array((), dtype=gs.np_float)
                else:
                    init_pos, init_quat = gu.transform_pos_quat_by_trans_quat(
                        np.array(morph.offset_pos, dtype=gs.np_float),
                        np.array(morph.offset_quat, dtype=gs.np_float),
                        np.array(morph.pos, dtype=gs.np_float),
                        np.array(morph.quat, dtype=gs.np_float),
                    )
                    init_qpos = np.concatenate([init_pos, init_quat])
                self._variant_init_qpos.append(init_qpos)
                self._variant_offset_pos.append(offset_pos)
                self._variant_offset_quat.append(offset_quat)
            else:
                gs.raise_exception(
                    f"Heterogeneous morphs only support URDF, MJCF, Primitive, and Mesh, got: {type(morph).__name__}."
                )

        # For multi-link entities, reassign indices and recompute variant ranges
        if len(self._links) > 1:
            self._reassign_heterogeneous_indices()

    def _add_heterogeneous_variant(self, link, cg_infos, vg_infos):
        """Add a heterogeneous variant's visual geoms to a link.

        RigidEntity overrides to additionally add collision geoms.
        """
        for g_info in vg_infos:
            link._add_vgeom(
                vmesh=g_info["vmesh"],
                init_pos=g_info.get("pos", gu.zero_pos()),
                init_quat=g_info.get("quat", gu.identity_quat()),
            )
        link._record_variant_vgeom_range(len(vg_infos))

    def _on_heterogeneous_scene_variant_loaded(self, link, morph, v_l_info):
        """Hook for subclasses after a scene variant's geoms have been added to a link."""

    def _reassign_heterogeneous_indices(self):
        """Reassign vgeom indices for multi-link heterogeneous entities.

        RigidEntity overrides to additionally handle collision geom indices.
        """
        running_vgeom_idx = self._vgeom_start
        running_vvert = self._vvert_start
        running_vface = self._vface_start

        for link in self._links:
            for vgeom in link.vgeoms:
                vgeom._idx = running_vgeom_idx
                vgeom._vvert_start = running_vvert
                vgeom._vface_start = running_vface
                running_vgeom_idx += 1
                running_vvert += vgeom.n_vverts
                running_vface += vgeom.n_vfaces

        for link in self._links:
            if link._variant_vgeom_ranges is None:
                continue
            vgeom_counts = [end - start for start, end in link._variant_vgeom_ranges]
            vgeom_cursor = link.vgeoms[0].idx if link.vgeoms else 0
            link._variant_vgeom_ranges = []
            for count in vgeom_counts:
                link._variant_vgeom_ranges.append((vgeom_cursor, vgeom_cursor + count))
                vgeom_cursor += count

    def _load_model(self):
        self._links = gs.List()
        self._joints = gs.List()

        self._load_morph(self._morph)

    def _load_primitive(self, morph, surface, load_geom_only_for_heterogeneous=False):
        if morph.fixed:
            joint_type = gs.JOINT_TYPE.FIXED
            n_qs = 0
            n_dofs = 0
            init_qpos = np.array([])
        else:
            joint_type = gs.JOINT_TYPE.FREE
            n_qs = 7
            n_dofs = 6
            init_qpos = np.concatenate([morph.pos, morph.quat])

        metadata: dict[str, Any] = {"texture_path": None}

        if isinstance(morph, gs.options.morphs.Box):
            extents = np.array(morph.size)
            tmesh = mu.create_box(extents=extents)
            cmesh = tmesh
            geom_data = extents
            geom_type = gs.GEOM_TYPE.BOX
            link_name_prefix = "box"
        elif isinstance(morph, gs.options.morphs.Sphere):
            tmesh = mu.create_sphere(radius=morph.radius)
            cmesh = tmesh
            geom_data = np.array([morph.radius])
            geom_type = gs.GEOM_TYPE.SPHERE
            link_name_prefix = "sphere"
        elif isinstance(morph, gs.options.morphs.Cylinder):
            tmesh = mu.create_cylinder(radius=morph.radius, height=morph.height)
            cmesh = tmesh
            geom_data = np.array([morph.radius, morph.height])
            geom_type = gs.GEOM_TYPE.CYLINDER
            link_name_prefix = "cylinder"
        elif isinstance(morph, gs.options.morphs.Plane):
            metadata["texture_path"] = mu.DEFAULT_PLANE_TEXTURE_PATH
            tmesh, cmesh = mu.create_plane(
                normal=morph.normal,
                plane_size=morph.plane_size,
                tile_size=morph.tile_size,
                color_or_texture=metadata["texture_path"],
            )
            geom_data = np.array(morph.normal)
            geom_type = gs.GEOM_TYPE.PLANE
            link_name_prefix = "plane"
        else:
            gs.raise_exception("Unsupported primitive shape")

        # contains one visual geom (vgeom) and one collision geom (geom)
        g_infos = []
        if morph.visualization:
            g_infos.append(
                dict(contype=0, conaffinity=0, vmesh=gs.Mesh.from_trimesh(tmesh, surface=surface, metadata=metadata))
            )
        if (morph.contype or morph.conaffinity) and morph.collision:
            g_infos.append(
                dict(
                    contype=morph.contype,
                    conaffinity=morph.conaffinity,
                    mesh=gs.Mesh.from_trimesh(cmesh, surface=gs.surfaces.Collision()),
                    type=geom_type,
                    data=geom_data,
                    sol_params=gu.default_solver_params(),
                )
            )

        # For heterogeneous simulation, only return geometry info without creating link/joint
        if load_geom_only_for_heterogeneous:
            return g_infos

        self._add_by_info(
            l_info=dict(
                is_robot=False,
                name=f"{link_name_prefix}_baselink",
                pos=np.array(morph.pos),
                quat=np.array(morph.quat),
                inertial_pos=None,  # we will compute the COM later based on the geometry
                inertial_quat=gu.identity_quat(),
                parent_idx=-1,
            ),
            j_infos=[
                dict(
                    name=f"{link_name_prefix}_baselink_joint",
                    n_qs=n_qs,
                    n_dofs=n_dofs,
                    type=joint_type,
                    init_qpos=init_qpos,
                )
            ],
            g_infos=g_infos,
            morph=morph,
            surface=surface,
        )
        return g_infos

    def _load_mesh(self, morph, surface, load_geom_only_for_heterogeneous=False):
        # Load meshes
        meshes = gs.Mesh.from_morph_surface(morph, surface)

        link_pos, link_quat = map(np.array, (morph.pos, morph.quat))

        if morph.fixed:
            joint_type = gs.JOINT_TYPE.FIXED
            n_qs = 0
            n_dofs = 0
            init_qpos = np.array([])
        else:
            joint_type = gs.JOINT_TYPE.FREE
            n_qs = 7
            n_dofs = 6
            init_qpos = np.concatenate([link_pos, link_quat])

        g_infos = []
        if morph.visualization:
            for mesh in meshes:
                g_infos.append(dict(contype=0, conaffinity=0, vmesh=mesh, pos=gu.zero_pos(), quat=gu.identity_quat()))
        if morph.collision:
            if morph.merge_submeshes_for_collision and len(meshes) > 1:
                # Merge every submesh into a single collision geom if requested.
                collision_groups = [list(meshes)]
            else:
                # A source mesh node split into several visual materials is one physical body, so its submeshes are
                # merged into a single collision geom rather than split per material. Pieces meant to collide
                # separately must be authored as separate nodes. Meshes with no source node are each their own body.
                collision_groups = []
                groups_by_node = {}
                for mesh in meshes:
                    node_index = mesh.metadata.get("node_index")
                    if node_index is None:
                        collision_groups.append([mesh])
                        continue
                    group = groups_by_node.get(node_index)
                    if group is None:
                        group = groups_by_node[node_index] = []
                        collision_groups.append(group)
                    group.append(mesh)

            for group in collision_groups:
                if len(group) == 1:
                    mesh = group[0]
                else:
                    tmesh = trimesh.util.concatenate([submesh.trimesh for submesh in group])
                    mesh = gs.Mesh.from_trimesh(mesh=tmesh, surface=gs.surfaces.Collision())
                g_infos.append(
                    dict(
                        contype=morph.contype,
                        conaffinity=morph.conaffinity,
                        mesh=mesh,
                        type=gs.GEOM_TYPE.MESH,
                        sol_params=gu.default_solver_params(),
                        pos=gu.zero_pos(),
                        quat=gu.identity_quat(),
                    )
                )

        # For heterogeneous simulation, only return geometry info without creating link/joint
        if load_geom_only_for_heterogeneous:
            return g_infos

        link_name = os.path.basename(morph.file).replace(".", "_")

        self._add_by_info(
            l_info=dict(is_robot=False, name=f"{link_name}_baselink", pos=link_pos, quat=link_quat, parent_idx=-1),
            j_infos=[
                dict(name=f"{link_name}_baselink_joint", n_qs=n_qs, n_dofs=n_dofs, type=joint_type, init_qpos=init_qpos)
            ],
            g_infos=g_infos,
            morph=morph,
            surface=surface,
        )
        return g_infos

    def _load_terrain(self, morph, surface):
        vmesh, mesh, self.terrain_hf = tu.parse_terrain(morph, surface)
        self.terrain_scale = np.array((morph.horizontal_scale, morph.vertical_scale), dtype=gs.np_float)

        g_infos = []
        if morph.visualization:
            g_infos.append(dict(contype=0, conaffinity=0, vmesh=vmesh))
        if morph.collision:
            g_infos.append(
                dict(
                    contype=1,
                    conaffinity=1,
                    mesh=mesh,
                    type=gs.GEOM_TYPE.TERRAIN,
                    sol_params=gu.default_solver_params(),
                )
            )

        self._add_by_info(
            l_info=dict(
                is_robot=False,
                name="baselink",
                pos=np.array(morph.pos),
                quat=np.array(morph.quat),
                inertial_pos=None,
                inertial_quat=gu.identity_quat(),
                inertial_i=None,
                inertial_mass=None,
                parent_idx=-1,
                invweight=None,
            ),
            j_infos=[dict(name="joint_baselink", n_qs=0, n_dofs=0, type=gs.JOINT_TYPE.FIXED)],
            g_infos=g_infos,
            morph=morph,
            surface=surface,
        )

    def _parse_scene(self, morph, surface):
        # Keep track of whether parsed inertia can be considered valid
        is_inertia_invalid = True

        # Mujoco's unified MJCF+URDF parser is not good enough for now to be used for loading both MJCF and URDF files.
        # First, it would happen when loading visual meshes having supported format (i.e. Collada files '.dae').
        # Second, it does not take into account URDF 'mimic' joint constraints. However, it does a better job at
        # initialized undetermined physics parameters.
        if isinstance(morph, gs.morphs.MJCF):
            # Mujoco's unified MJCF+URDF parser systematically for MJCF files
            l_infos, links_j_infos, links_g_infos, eqs_info = mju.parse_xml(morph, surface, self._solver._options)
        elif isinstance(morph, (gs.morphs.URDF, gs.morphs.Drone)):
            # Custom "legacy" URDF parser for loading geometries (visual and collision) and equality constraints.
            # This is necessary because Mujoco cannot parse visual geometries (meshes) reliably for URDF.
            l_infos, links_j_infos, links_g_infos, eqs_info = uu.parse_urdf(morph, surface)

            # Mujoco's unified MJCF+URDF parser for only link, joints, and collision geometries properties
            morph_ = morph.model_copy(update=dict(visualization=False))
            try:
                # Mujoco's unified MJCF+URDF parser for URDF files.
                # Note that Mujoco URDF parser completely ignores equality constraints.
                l_infos_mj, links_j_infos_mj, links_g_infos_mj, _ = mju.parse_xml(morph_, surface)

                # Unset link inertial properties that are actually undefined to force recomputation by genesis
                if not morph._enable_mujoco_compatibility:
                    for l_info_gs in l_infos:
                        for l_info_mj in l_infos_mj:
                            if l_info_gs["name"] == l_info_mj["name"]:
                                for key, value in l_info_gs.items():
                                    if value is None:
                                        l_info_mj[key] = None
                                        is_inertia_invalid = False
                                break
                l_infos = l_infos_mj

                # Mujoco is not parsing actuators properties
                for j_info_gs in chain.from_iterable(links_j_infos):
                    for j_info_mj in chain.from_iterable(links_j_infos_mj):
                        if j_info_mj["name"] == j_info_gs["name"]:
                            for name in ("dofs_force_range", "dofs_armature", "dofs_act_gain", "dofs_act_bias"):
                                j_info_mj[name] = j_info_gs[name]
                            break
                links_j_infos = links_j_infos_mj

                # Must invalidate invweight if default rotor armature inertia has been specified
                if morph.default_armature is not None:
                    for link_j_infos in links_j_infos:
                        for j_info in link_j_infos:
                            if j_info["type"] not in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.FIXED):
                                is_inertia_invalid = False
                                break

                # Take into account 'world' body if it was added automatically for our legacy URDF parser
                if len(links_g_infos_mj) == len(links_g_infos) + 1:
                    assert not links_g_infos_mj[0]
                    links_g_infos.insert(0, [])
                assert len(links_g_infos_mj) == len(links_g_infos)

                # Update collision geometries, ignoring fake" visual geometries returned by Mujoco, (which is using
                # collision as visual to avoid loading mesh files), and keeping the true visual geometries provided
                # by our custom legacy URDF parser.
                # Note that the Kinematic tree ordering is stable between Mujoco and Genesis (Hopefully!).
                for link_g_infos, link_g_infos_mj in zip(links_g_infos, links_g_infos_mj):
                    # Remove collision geometries from our legacy URDF parser
                    for i_g, g_info in tuple(enumerate(link_g_infos))[::-1]:
                        is_col = g_info["contype"] or g_info["conaffinity"]
                        if is_col:
                            del link_g_infos[i_g]

                    # Add visual geometries from Mujoco's unified MJCF+URDF parser
                    for g_info in link_g_infos_mj:
                        is_col = g_info["contype"] or g_info["conaffinity"]
                        if is_col:
                            link_g_infos.append(g_info)
            except (ValueError, AssertionError) as e:
                gs.logger.warning(
                    "Falling back to legacy URDF parser. Default values of physics properties may be off:\n"
                    + str(e).replace("\n", " - ")
                )
        elif isinstance(morph, gs.morphs.USD):
            from genesis.utils.usd import parse_usd_rigid_entity

            # Unified parser handles both articulations and rigid bodies
            l_infos, links_j_infos, links_g_infos, eqs_info = parse_usd_rigid_entity(morph, surface)

        # Make sure that the inertia matrix of all links is valid
        if not morph.recompute_inertia:
            for l_info in l_infos:
                inertia_i = l_info.get("inertial_i")
                if inertia_i is None:
                    continue

                # Compute eigenvalues of inertia matrix after enforcing symmetry
                inertia_diag, Q = np.linalg.eigh(0.5 * (inertia_i + inertia_i.T))

                # Make sure that all eigenvalues are positive, ignoring rounding errors
                if (inertia_diag < -gs.EPS).any():
                    gs.raise_exception(
                        f"Inertia matrix of link '{l_info['name']}' not positive definite (eigenvalues: {inertia_diag})."
                    )

                # Make sure that the inertia matrix is physically valid (nothing to do with numerical conditioning)
                if any(
                    inertia_diag[i] + inertia_diag[(i + 1) % 3] < inertia_diag[(i + 2) % 3] * (1.0 - 1e-6) - 1e-9
                    for i in range(3)
                ):
                    gs.raise_exception(
                        f"Inertia matrix of link '{l_info['name']}' does not satisfy A+B>=C for all permutations "
                        f"(eigenvalues: {inertia_diag}). Please fix manually you morph file '{morph.file}' or specify "
                        "`recompute_inertia=True`."
                    )

                # Make sure that the inertia matrix is symmetric with positive eigenvalues
                l_info["inertial_i"] = Q @ np.diag(np.maximum(inertia_diag, 0.0)) @ Q.T

        # Remove any "virtual" root link that was not present in the original file morph.
        # Mujoco unified parser and our legacy parser have different behaviors.
        # * Mujoco unified parser always adds a root 'world' link if it does not exist, and fuse all fixed links from
        #   root to first articulated body.
        # * Our legacy parser adds a root 'world' link if the root joint is not a fixed joint in file morph.
        # Remove this virtual world link if the child has a free joint (the free joint absorbs the full pose into
        # 'init_qpos' regardless of pos/quat), or if the child has an identity transform.
        base_j_info, base_g_info = links_j_infos[0], links_g_infos[0]
        if len(l_infos) > 1 and (sum(j_info["n_dofs"] for j_info in base_j_info) == 0) and not base_g_info:
            child_has_freejoint = any(j_info["type"] == gs.JOINT_TYPE.FREE for j_info in links_j_infos[1])
            child_is_identity = (np.abs(l_infos[1]["pos"]) < gs.EPS).all() and (
                np.abs(l_infos[1]["quat"] - (1, 0, 0, 0)) < gs.EPS
            ).all()
            if child_has_freejoint or child_is_identity:
                del l_infos[0], links_j_infos[0], links_g_infos[0]
                for l_info in l_infos:
                    l_info["parent_idx"] = max(l_info["parent_idx"] - 1, -1)
                    if "root_idx" in l_info:
                        l_info["root_idx"] = max(l_info["root_idx"] - 1, -1)

        # URDF is a robot description file so all links have same root_idx
        if isinstance(morph, gs.morphs.URDF) and not morph._enable_mujoco_compatibility:
            for l_info in l_infos:
                l_info["root_idx"] = 0

        # Genesis requires links associated with free joints to be attached to the world directly
        for l_info, link_j_infos in zip(l_infos, links_j_infos):
            if all(j_info["type"] == gs.JOINT_TYPE.FREE for j_info in link_j_infos):
                l_info["parent_idx"] = -1

        # Add free floating joint at root if necessary
        if (
            (isinstance(morph, gs.morphs.Drone) or (isinstance(morph, gs.morphs.URDF) and not morph.fixed))
            and links_j_infos
            and sum(j_info["n_dofs"] for j_info in links_j_infos[0]) == 0
        ):
            # Define free joint
            j_info = dict()
            j_info["name"] = "root_joint"
            j_info["type"] = gs.JOINT_TYPE.FREE
            j_info["n_qs"] = 7
            j_info["n_dofs"] = 6
            j_info["init_qpos"] = np.concatenate([gu.zero_pos(), gu.identity_quat()])
            j_info["pos"] = gu.zero_pos()
            j_info["quat"] = gu.identity_quat()
            j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
            j_info["dofs_motion_vel"] = np.eye(6, 3)
            j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
            j_info["dofs_stiffness"] = np.zeros(6)
            j_info["dofs_invweight"] = np.zeros(6)
            j_info["dofs_frictionloss"] = np.zeros(6)
            j_info["dofs_damping"] = np.zeros(6)
            if isinstance(morph, gs.morphs.Drone):
                # FIXME: This pattern not ideal because the inertial mass may be unknown at this point.
                mass_tot = sum(l_info.get("inertial_mass") or 0.0 for l_info in l_infos)
                j_info["dofs_damping"][3:] = mass_tot * morph.default_base_ang_damping_scale
            j_info["dofs_armature"] = np.zeros(6)
            j_info["dofs_act_gain"] = np.zeros((6,), dtype=gs.np_float)
            j_info["dofs_act_bias"] = np.zeros((6, 3), dtype=gs.np_float)
            j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (6, 1))
            links_j_infos[0] = [j_info]

            # The base link is now moving, which merges every kinematic tree connected to it into a single tree rooted
            # at it. Parser-provided root indices assume a base welded to the world, so recompute them by propagating
            # each parent's root (parents precede children); links with no parent root their own tree.
            for i_l, l_info in enumerate(l_infos):
                if "root_idx" in l_info:
                    parent_idx = l_info["parent_idx"]
                    l_info["root_idx"] = l_infos[parent_idx]["root_idx"] if parent_idx != -1 else i_l

            # Must invalidate invweight for all child links and joints because the root joint was fixed when it was
            # initially computed. Re-initialize it to some strictly negative value to trigger recomputation in solver.
            for i_l in range(len(l_infos)):
                l_infos[i_l]["invweight"] = np.full((2,), fill_value=-1.0)
                for j_info in links_j_infos[i_l]:
                    j_info["dofs_invweight"] = np.full((j_info["n_dofs"],), fill_value=-1.0)

        # Force recomputing inertial information based on geometry if ill-defined for some reason.
        # A moving link needs a well-defined inertia only for its own rigid body; a rigidly-attached (fixed-joint) child
        # folds its mass into the parent's composite-rigid-body inertia.
        has_links_subtree_mass = [
            bool(link_g_infos) or (l_info.get("inertial_mass") or 0.0) > 0.0
            for link_g_infos, l_info in zip(links_g_infos, l_infos)
        ]
        for i in reversed(range(len(l_infos))):
            parent_idx = l_infos[i]["parent_idx"]
            if parent_idx >= 0 and all(j_info["type"] == gs.JOINT_TYPE.FIXED for j_info in links_j_infos[i]):
                has_links_subtree_mass[parent_idx] |= has_links_subtree_mass[i]

        is_inertia_invalid = False
        for i, (l_info, link_g_infos, link_j_infos, has_link_subtree_mass) in enumerate(
            zip(l_infos, links_g_infos, links_j_infos, has_links_subtree_mass)
        ):
            # Fixed links are subsumed into their parent's composite; only moving links need a well-defined inertia.
            if all(j_info["type"] == gs.JOINT_TYPE.FIXED for j_info in link_j_infos):
                continue
            if not (
                (l_info.get("inertial_mass") is None or l_info["inertial_mass"] <= 0.0)
                or (l_info.get("inertial_i") is None or (np.diag(l_info["inertial_i"]) <= 0.0).any())
            ):
                continue

            # The own inertia is degenerate, so the parsed inverse weight (derived from it) must be recomputed
            # regardless of how the mass is resolved below.
            is_inertia_invalid = True

            # A geometry-less moving link whose rigidly-attached (fixed-joint) subtree carries the mass keeps its
            # (near-)zero own inertia: the composite-rigid-body inertia is finite, so leave it as parsed. Otherwise
            # recompute from its own geometry, warning only when nothing in the rigid subtree provides mass.
            if not link_g_infos and has_link_subtree_mass:
                continue
            if not link_g_infos:
                gs.logger.warning(
                    f"Moving link '{l_info['name']}' has no mass, inertia or geometry, and no rigidly-attached "
                    "child provides any. Setting its mass to 'gs.EPS'."
                )
            elif l_info.get("inertial_mass") is not None or l_info.get("inertial_i") is not None:
                gs.logger.debug(
                    f"Invalid or undefined inertia for link '{l_info['name']}'. Force recomputing it based on geometry."
                )
            l_info["inertial_i"] = None
        if is_inertia_invalid:
            for l_info, link_j_infos in zip(l_infos, links_j_infos):
                l_info["invweight"] = np.full((2,), fill_value=-1.0)
                for j_info in link_j_infos:
                    j_info["dofs_invweight"] = np.full((j_info["n_dofs"],), fill_value=-1.0)

        # Check if there is something weird with the options
        non_physical_fieldnames = ("dofs_frictionloss", "dofs_damping", "dofs_armature")
        for j_info in (
            j_info for link_j_infos in links_j_infos for j_info in link_j_infos if j_info["type"] == gs.JOINT_TYPE.FREE
        ):
            if not all((j_info[name] < gs.EPS).all() for name in non_physical_fieldnames if name in j_info):
                gs.logger.warning(
                    "Some free joint has non-zero frictionloss, damping or armature parameters. Beware it is "
                    "non-physical."
                )

        # Define a flag that determines whether the link at hand is associated with a robot.
        # Note that 0d array is used rather than native type because this algo requires mutable objects.
        for l_info, link_j_infos in zip(l_infos, links_j_infos):
            if not link_j_infos or all(j_info["type"] == gs.JOINT_TYPE.FIXED for j_info in link_j_infos):
                if l_info["parent_idx"] >= 0:
                    l_info["is_robot"] = l_infos[l_info["parent_idx"]]["is_robot"]
                else:
                    l_info["is_robot"] = np.array(False, dtype=np.bool_)
            elif all(j_info["type"] == gs.JOINT_TYPE.FREE for j_info in link_j_infos):
                l_info["is_robot"] = np.array(False, dtype=np.bool_)
            else:
                l_info["is_robot"] = np.array(True, dtype=np.bool_)
                if l_info["parent_idx"] >= 0:
                    l_infos[l_info["parent_idx"]]["is_robot"][()] = True

        # Apply morph pos and quat if specified
        for l_info, link_j_infos in zip(l_infos, links_j_infos):
            if l_info["parent_idx"] < 0:
                if morph.pos is not None or morph.quat is not None:
                    gs.logger.debug("Applying offset to base link's pose with user provided value in morph.")
                    pos = np.asarray(l_info.get("pos", (0.0, 0.0, 0.0)))
                    quat = np.asarray(l_info.get("quat", (1.0, 0.0, 0.0, 0.0)))
                    if morph.pos is None:
                        pos_offset = np.zeros((3,))
                    else:
                        pos_offset = np.asarray(morph.pos)
                    if morph.quat is None:
                        quat_offset = np.array((1.0, 0.0, 0.0, 0.0))
                    else:
                        quat_offset = np.asarray(morph.quat)
                    l_info["pos"], l_info["quat"] = gu.transform_pos_quat_by_trans_quat(
                        pos, quat, pos_offset, quat_offset
                    )

                for j_info in link_j_infos:
                    if j_info["type"] == gs.JOINT_TYPE.FREE:
                        # in this case, l_info['pos'] and l_info['quat'] are actually not used in solver,
                        # but this initial value will be reflected
                        j_info["init_qpos"] = np.concatenate([l_info["pos"], l_info["quat"]])

        # Exclude joints with 0 dofs to align with Mujoco
        links_j_infos = [[j_info for j_info in link_j_infos if j_info["n_dofs"] > 0] for link_j_infos in links_j_infos]

        return l_infos, links_j_infos, links_g_infos, eqs_info

    def _load_scene(self, morph, surface):
        l_infos, links_j_infos, links_g_infos, _eqs_info = self._parse_scene(morph, surface)

        # Add (link, joints, geoms) tuples sequentially
        for l_info, link_j_infos, link_g_infos in zip(l_infos, links_j_infos, links_g_infos):
            self._add_by_info(l_info, link_j_infos, link_g_infos, morph, surface)

    def _build(self):
        for link in self._links:
            link._build()

        self._n_qs = self.n_qs
        self._n_dofs = self.n_dofs
        self._vgeoms = self.vgeoms
        self._is_built = True

        # The per-link inertia (and per heterogeneous variant) is now finalized, so anchor each aligned free root at
        # its fixed subtree center of mass and principal axes. Must run before the solver reads the link poses and
        # inertia. Defined here on the base class so kinematic and rigid entities anchor identically: a kinematic
        # entity is commonly used to visualize the target reference motion a rigid entity tracks, so the two coexist
        # and the same qpos must map to the same world pose for both.
        self._align_free_roots()

    def _align_free_roots(self):
        """Anchor each aligned free root at the center of mass and principal axes of its fixed subtree.

        Runs after build, so it composes the finalized inertia the solver's mass matrix uses, and applies the anchoring
        per heterogeneous variant (each variant's own inertia and geoms). This delivers the 'align' option's promise (a
        COM-centered, principal-axis frame, which also conditions the constraint solve) for mesh and fixed-child
        bodies, not just primitives, while keeping a heterogeneous entity bit-identical to its variants' separate
        entities.
        """
        for root in self._links:
            if root.parent_idx != -1 or not root.aligned:
                continue

            # Gather the fixed subtree (root + transitive n_dofs == 0 descendants) and each link's pose in the root
            # frame; links are in build order so a parent is always visited before its children. A DOF-bearing
            # descendant makes the root an articulated chain rather than a single rigid body: its joint-space mass is
            # not diagonal and the frames of its moving children are not re-expressed here, so such roots are skipped.
            pose_in_root = {root.idx: (gu.zero_pos(), gu.identity_quat())}
            subtree = [root]
            articulated = False
            for link in self._links:
                if link is root or link.parent_idx not in pose_in_root:
                    continue
                if link.n_dofs != 0:
                    articulated = True
                    break
                pose_in_root[link.idx] = gu.transform_pos_quat_by_trans_quat(
                    np.asarray(link.pos, dtype=gs.np_float),
                    np.asarray(link.quat, dtype=gs.np_float),
                    *pose_in_root[link.parent_idx],
                )
                subtree.append(link)
            if articulated:
                continue

            # Anchor each variant on its own finalized inertia (from the load-time stash, available to kinematic and
            # rigid entities alike) and geoms. Composing the fixed subtree's inertia and shifting the body frame to its
            # COM and principal axes re-expresses the variant's geoms (so the world geometry is unchanged), folds the
            # composite into the variant's dynamics inertia (rigid only) and into its offset and init_qpos.
            root_local = root.idx - self._link_start
            is_heterogeneous = root._variant_vgeom_ranges is not None
            for v in range(len(self._links_inertial_info[root_local])):
                inertial_info = []
                mass_explicit = set()
                composite_links = []
                for link in subtree:
                    props, is_mass_explicit, _ = self._links_inertial_info[link.idx - self._link_start][v]
                    if props.mass < gs.EPS:
                        continue
                    composite_links.append(link)
                    mass_explicit.add(is_mass_explicit)
                    rot = gu.quat_to_R(props.quat)
                    inertia_in_link = rot @ props.inertia @ rot.T
                    inertial_info.append(
                        GeomInertialInfo(
                            InertialProperties(props.mass, props.com, inertia_in_link), *pose_in_root[link.idx]
                        )
                    )

                # The composite center of mass and principal axes are density-independent only if every contributing
                # link's mass comes from the same source (see 'LinkInertialInfo.is_mass_explicit'). Mixing explicit
                # and estimated masses would make the anchor density-dependent, and a kinematic entity has no density
                # to fall back on, so alignment could differ from the rigid counterpart. Require all-or-none and raise
                # otherwise.
                if None in mass_explicit:
                    gs.raise_exception(
                        f"Entity '{self.uid}': a link of an aligned free body mixes geoms with and without an "
                        "authored density. Author a density on all of its geoms or none of them."
                    )
                if len(mass_explicit) > 1:
                    gs.raise_exception(
                        f"Entity '{self.uid}': an aligned free body mixes explicit (mass or authored per-geom "
                        "densities) and geometry-estimated link masses. Specify the mass or density of all of its "
                        "links or none of them."
                    )
                if not inertial_info:
                    continue
                mass_total, com_root, inertia_root = compose_inertial_properties(inertial_info)
                if mass_total <= gs.EPS:
                    continue
                principal_R = uu.principal_axes_rot(inertia_root)
                principal_quat = gu.R_to_quat(principal_R)
                inertia_diag = principal_R.T @ inertia_root @ principal_R

                # Re-express the variant's geoms across the subtree so the body frame can move to (com_root, principal
                # axes) while the world geometry stays fixed: bring each geom to the root frame, undo the anchoring
                # there, bring it back to its link frame. The static link poses are not moved (they are shared across
                # heterogeneous variants). A kinematic link has only visual geoms; a rigid link has both.
                for link in subtree:
                    link_pos, link_quat = pose_in_root[link.idx]
                    if is_heterogeneous:
                        vgeom_start, vgeom_end = link._variant_vgeom_ranges[v]
                        geoms = [vg for vg in link.vgeoms if vgeom_start <= vg.idx < vgeom_end]
                        if isinstance(link, RigidLink):
                            geom_start, geom_end = link._variant_geom_ranges[v]
                            geoms += [g for g in link.geoms if geom_start <= g.idx < geom_end]
                    elif isinstance(link, RigidLink):
                        geoms = [*link.geoms, *link.vgeoms]
                    else:
                        geoms = list(link.vgeoms)
                    for geom in geoms:
                        pos, quat = gu.transform_pos_quat_by_trans_quat(
                            geom._init_pos, geom._init_quat, link_pos, link_quat
                        )
                        pos, quat = gu.inv_transform_pos_quat_by_trans_quat(pos, quat, com_root, principal_quat)
                        pos, quat = gu.inv_transform_pos_quat_by_trans_quat(pos, quat, link_pos, link_quat)
                        geom._init_pos, geom._init_quat = pos, quat
                        geom._init_pos_tc = torch.from_numpy(pos).to(device=gs.device, dtype=gs.tc_float)
                        geom._init_quat_tc = torch.from_numpy(quat).to(device=gs.device, dtype=gs.tc_float)

                # Fold the composite (diagonal, COM-centered) into the dynamics inertia.
                # Rescale the unit-density estimate to the link masses.
                if isinstance(root, RigidLink):
                    # Sum the dynamics masses of exactly the links that contributed to 'mass_total'; the massless
                    # links skipped above (a geometry-less link carries only the 'gs.EPS' placeholder) must not
                    # inflate the composite.
                    real_total = sum(
                        float(link._variant_inertial[v][0] if is_heterogeneous else link.inertial_mass)
                        for link in composite_links
                    )
                    scale = real_total / mass_total
                    zero_pos, identity_quat = gu.zero_pos(), gu.identity_quat()
                    for link in subtree:
                        if link is root:
                            props = LinkInertial(real_total, zero_pos, identity_quat, scale * inertia_diag)
                        else:
                            props = LinkInertial(gs.EPS, zero_pos, identity_quat, np.zeros((3, 3), dtype=gs.np_float))
                        if is_heterogeneous:
                            link._variant_inertial[v] = props
                        else:
                            link._inertial_mass, link._inertial_pos, link._inertial_quat, link._inertial_i = props

                # Fold the anchoring into the offset (relative getters keep reporting the user frame) and the root
                # free-joint init_qpos (the body frame moves to old_pose o (com_root, principal), keeping the
                # re-expressed geoms in world).
                if is_heterogeneous:
                    off_pos, off_quat = self._variant_offset_pos[v], self._variant_offset_quat[v]
                    self._variant_offset_pos[v] = gu.transform_by_trans_quat(com_root, off_pos, off_quat)
                    self._variant_offset_quat[v] = gu.transform_quat_by_quat(principal_quat, off_quat)
                    # The solver broadcasts '_links_offset_*' (not '_variant_offset_*') when all variants share an
                    # offset, so keep the base link offset in sync with the primary variant's aligned offset.
                    if v == 0:
                        self._links_offset_pos[root_local] = self._variant_offset_pos[0]
                        self._links_offset_quat[root_local] = self._variant_offset_quat[0]
                        if root_local == 0:
                            self._offset_pos = self._variant_offset_pos[0]
                            self._offset_quat = self._variant_offset_quat[0]
                    qpos = self._variant_init_qpos[v]
                    new_pos = gu.transform_by_trans_quat(com_root, qpos[:3], qpos[3:7])
                    new_quat = gu.transform_quat_by_quat(principal_quat, qpos[3:7])
                    qpos[:3], qpos[3:7] = new_pos, new_quat
                else:
                    off_pos, off_quat = self._links_offset_pos[root_local], self._links_offset_quat[root_local]
                    self._links_offset_pos[root_local] = gu.transform_by_trans_quat(com_root, off_pos, off_quat)
                    self._links_offset_quat[root_local] = gu.transform_quat_by_quat(principal_quat, off_quat)
                    if root_local == 0:
                        self._offset_pos, self._offset_quat = self._links_offset_pos[0], self._links_offset_quat[0]
                    for joint in root.joints:
                        if joint.type == gs.JOINT_TYPE.FREE:
                            new_pos = gu.transform_by_trans_quat(com_root, joint._init_qpos[:3], joint._init_qpos[3:7])
                            new_quat = gu.transform_quat_by_quat(principal_quat, joint._init_qpos[3:7])
                            joint._init_qpos = np.concatenate([new_pos, new_quat])
                    root._pos = gu.transform_by_trans_quat(com_root, np.asarray(root.pos, dtype=gs.np_float), root.quat)
                    root._quat = gu.transform_quat_by_quat(principal_quat, np.asarray(root.quat, dtype=gs.np_float))

        # The load-time inertial info was only needed for the build-time estimates and the anchors; drop it now that
        # the alignment lives in the persistent geometry, offset and init_qpos.
        self._links_inertial_info = []

    def _create_joints(self, j_infos, link_idx, joint_start):
        """Create RigidJoint objects from joint info dicts.

        Shared by KinematicEntity._add_by_info and RigidEntity._add_by_info.
        """
        joints = gs.List()
        self._joints.append(joints)
        for i_j_, j_info in enumerate(j_infos):
            n_dofs = j_info["n_dofs"]

            sol_params = np.array(j_info.get("sol_params", gu.default_solver_params()), copy=True)
            if (
                len(sol_params.shape) == 2
                and sol_params.shape[0] == 1
                and (sol_params[0][3] >= 1.0 or sol_params[0][2] >= sol_params[0][3])
            ):
                gs.logger.warning(
                    f"Joint {j_info['name']}'s sol_params {sol_params[0]} look not right, change to default."
                )
                sol_params = gu.default_solver_params()

            dofs_motion_ang = j_info.get("dofs_motion_ang")
            if dofs_motion_ang is None:
                if n_dofs == 6:
                    dofs_motion_ang = np.eye(6, 3, -3)
                elif n_dofs == 0:
                    dofs_motion_ang = np.zeros((0, 3))
                else:
                    assert False

            dofs_motion_vel = j_info.get("dofs_motion_vel")
            if dofs_motion_vel is None:
                if n_dofs == 6:
                    dofs_motion_vel = np.eye(6, 3)
                elif n_dofs == 0:
                    dofs_motion_vel = np.zeros((0, 3))
                else:
                    assert False

            joint = RigidJoint(
                entity=self,
                name=j_info["name"],
                idx=joint_start + i_j_,
                link_idx=link_idx,
                q_start=self.n_qs + self._q_start,
                dof_start=self.n_dofs + self._dof_start,
                n_qs=j_info["n_qs"],
                n_dofs=n_dofs,
                type=j_info["type"],
                pos=j_info.get("pos", gu.zero_pos()),
                quat=j_info.get("quat", gu.identity_quat()),
                init_qpos=j_info.get("init_qpos", np.zeros(n_dofs)),
                sol_params=sol_params,
                dofs_motion_ang=dofs_motion_ang,
                dofs_motion_vel=dofs_motion_vel,
                dofs_limit=j_info.get("dofs_limit", np.tile([[-np.inf, np.inf]], [n_dofs, 1])),
                dofs_invweight=j_info.get("dofs_invweight", np.zeros(n_dofs)),
                dofs_frictionloss=j_info.get("dofs_frictionloss", np.zeros(n_dofs)),
                dofs_stiffness=j_info.get("dofs_stiffness", np.zeros(n_dofs)),
                dofs_damping=j_info.get("dofs_damping", np.zeros(n_dofs)),
                dofs_armature=j_info.get("dofs_armature", np.zeros(n_dofs)),
                dofs_act_gain=j_info.get("dofs_act_gain", np.zeros(n_dofs)),
                dofs_act_bias=j_info.get("dofs_act_bias", np.zeros((n_dofs, 3))),
                dofs_force_range=j_info.get("dofs_force_range", np.tile([[-np.inf, np.inf]], [n_dofs, 1])),
            )
            joints.append(joint)

        return joints

    def _add_by_info(self, l_info, j_infos, g_infos, morph, surface):
        if len(j_infos) > 1 and any(j_info["type"] in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.FIXED) for j_info in j_infos):
            raise ValueError(
                "Compounding joints of types 'FREE' or 'FIXED' with any other joint on the same body not supported"
            )

        parent_idx = l_info["parent_idx"]
        if parent_idx >= 0:
            parent_idx += self._link_start
        root_idx = l_info.get("root_idx")
        if root_idx is not None and root_idx >= 0:
            root_idx += self._link_start
        link_idx = self.n_links + self._link_start
        joint_start = self.n_joints + self._joint_start

        cg_infos, vg_infos = self._postprocess_geoms_info(morph, g_infos, l_info.get("is_robot", False))
        self._align_link(l_info, j_infos, cg_infos, vg_infos, morph, link_idx)

        joints = self._create_joints(j_infos, link_idx, joint_start)

        # Add child link
        link = KinematicLink(
            entity=self,
            name=l_info["name"],
            idx=link_idx,
            joint_start=joint_start,
            n_joints=len(j_infos),
            vgeom_start=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            pos=l_info["pos"],
            quat=l_info["quat"],
            parent_idx=parent_idx,
            root_idx=root_idx,
            aligned=l_info.get("aligned", False),
        )
        self._links.append(link)

        # Add visual geometries
        for g_info in vg_infos:
            link._add_vgeom(
                vmesh=g_info["vmesh"],
                init_pos=g_info.get("pos", gu.zero_pos()),
                init_quat=g_info.get("quat", gu.identity_quat()),
            )

        return link, joints

    def _postprocess_geoms_info(self, morph, g_infos, is_robot):
        """
        Split g_infos into (cg_infos, vg_infos) collision and visual lists, then
        post-process collision meshes (convexification / decomposition).
        Used for both normal loading and heterogeneous simulation.
        """
        # An explicitly set material density overrides any asset-authored per-geom density, as material friction does
        # for authored frictions. Dropping the keys up front keeps the align anchor, its all-or-none source check,
        # and the build-time inertial estimate consistently uniform-density.
        if isinstance(self.material, gs.materials.Rigid) and self.material.rho is not None:
            for g_info in g_infos:
                g_info.pop("density", None)

        cg_infos, vg_infos = [], []
        for g_info in g_infos:
            is_col = g_info["contype"] or g_info["conaffinity"]
            if morph.collision and is_col:
                cg_infos.append(g_info)
            if morph.visualization and not is_col:
                vg_infos.append(g_info)

        # Post-process all collision meshes at once.
        # Destroying the original geometries should be avoided if possible as it will change the way objects
        # interact with the world due to only computing one contact point per convex geometry. The idea is to
        # check if each geometry can be convexified independently without resorting on convex decomposition.
        # If so, the original geometries are preserve. If not, then they are all merged as one. Following the
        # same approach as before, the resulting geometry is convexify without resorting on convex decomposition
        # if possible. Mergeing before falling back directly to convex decompositio is important as it gives one
        # last chance to avoid it. Moreover, it tends to reduce the final number of collision geometries. In
        # both cases, this improves runtime performance, numerical stability and compilation time.
        if isinstance(morph, gs.options.morphs.FileMorph):
            # Choose the appropriate convex decomposition error threshold depending on whether the link at hand
            # is associated with a robot.
            # The rational behind it is that performing convex decomposition for robots is mostly useless because
            # the non-physical part that is added to the original geometries to convexify them are generally inside
            # the mechanical structure and not interacting directly with the outer world. On top of that, not only
            # iy increases the memory footprint and compilation time, but also the simulation speed (marginally).
            if is_robot:
                decompose_error_threshold = morph.decompose_robot_error_threshold
            else:
                decompose_error_threshold = morph.decompose_object_error_threshold

            # A collision geom may carry per-geom post-processing overrides (a USD MeshCollisionAPI approximation
            # hint sets "convexify"/"decimate"/"decompose_error_threshold"), with the morph options as defaults for
            # whatever is left unset. Post-processing merges geoms within a call, so each set of effective options
            # gets its own call; geoms without overrides share one, preserving the whole-entity merge behavior.
            # Thresholds match under gs.EPS tolerance so float noise cannot split a group.
            cg_infos_by_options: list[tuple[tuple, list]] = []
            for g_info in cg_infos:
                convexify = g_info.pop("convexify", morph.convexify)
                decimate = g_info.pop("decimate", morph.decimate)
                threshold = g_info.pop("decompose_error_threshold", decompose_error_threshold)
                for options, options_cg_infos in cg_infos_by_options:
                    if options[:2] == (convexify, decimate) and math.isclose(options[2], threshold, abs_tol=gs.EPS):
                        options_cg_infos.append(g_info)
                        break
                else:
                    cg_infos_by_options.append(((convexify, decimate, threshold), [g_info]))

            cg_infos = []
            for (convexify, decimate, threshold), options_cg_infos in cg_infos_by_options:
                cg_infos += mu.postprocess_collision_geoms(
                    options_cg_infos,
                    decimate,
                    morph.decimate_face_num,
                    morph.decimate_aggressiveness,
                    convexify,
                    threshold,
                    morph.coacd_options,
                    morph.watertighten,
                )

        # Randomize collision mesh colors. This is especially useful to check convex decomposition.
        for g_info in cg_infos:
            mesh = g_info["mesh"]
            mesh.set_color((*np.random.rand(3), 0.7))

        return cg_infos, vg_infos

    def _finalize_inertial(
        self, explicit_mass, explicit_com, explicit_quat, explicit_inertia, cg_infos, vg_infos, is_robot
    ):
        """Compute a link's load-time inertial data (see 'LinkInertialInfo').

        The align-anchor inertial weighs each collision geom by its authored density, falling back to unit density,
        so it never needs the material density - which a kinematic entity does not have. The geometry hint consumed
        by 'RigidLink._build' uses the resolved material density as fallback instead, and falls back to the visual
        geoms for a link without collision geometry.
        """
        if cg_infos:
            hint = compose_inertial_from_g_infos(cg_infos, rho=1.0)
        else:
            hint = InertialProperties(0.0, np.zeros(3, dtype=gs.np_float), np.zeros((3, 3), dtype=gs.np_float))
        props = finalize_inertial(
            explicit_mass, explicit_com, explicit_quat, explicit_inertia, *hint, clamp_min_mass=False
        )
        if explicit_mass is not None and explicit_mass > 0.0:
            is_mass_explicit = True
        else:
            geoms_with_density = sum(g_info.get("density") is not None for g_info in cg_infos)
            if geoms_with_density == 0:
                is_mass_explicit = False
            elif geoms_with_density == len(cg_infos):
                is_mass_explicit = True
            else:
                is_mass_explicit = None

        dynamics_hint = None
        if isinstance(self.material, gs.materials.Rigid):
            rho = self.material.rho
            if rho is None:
                if self._solver._enable_mujoco_compatibility:
                    rho = RHO_MUJOCO
                else:
                    rho = RHO_ROBOT if is_robot else RHO_OBJECT
            hint_g_infos = cg_infos if cg_infos else vg_infos
            if hint_g_infos:
                dynamics_hint = compose_inertial_from_g_infos(hint_g_infos, rho)
            else:
                dynamics_hint = InertialProperties(
                    0.0, np.zeros(3, dtype=gs.np_float), np.zeros((3, 3), dtype=gs.np_float)
                )
        return LinkInertialInfo(props, is_mass_explicit, dynamics_hint)

    def _align_link(self, l_info, j_infos, cg_infos, vg_infos, morph, link_idx):
        """Carry the morph pose offset into a root link and record its offset, and stash each link's finalized inertia.

        Only root (floating-base) links carry the morph 'offset_pos'/'offset_quat' and the 'aligned' flag; the resulting
        body-frame offset is stored in '_links_offset_*' so the relative getters report the user's original pose.
        Separately, every link's finalized inertia (explicit values, else from its collision geometry) is computed here
        - the one point where the collision geometry is available to both kinematic and rigid entities - and stashed
        transiently so '_align_free_roots' can derive the COM/principal anchor identically for both.
        """
        # Stash the finalized local inertia of this link (primary variant). recompute_inertia discards explicit values
        # for non-world-fixed links, exactly as RigidLink._build does; an aligned free body's subtree is never
        # world-fixed, so it suffices to honor the morph flag here.
        recompute = isinstance(morph, gs.options.morphs.FileMorph) and morph.recompute_inertia
        self._links_inertial_info.append(
            [
                self._finalize_inertial(
                    None if recompute else l_info.get("inertial_mass"),
                    None if recompute else l_info.get("inertial_pos"),
                    None if recompute else l_info.get("inertial_quat"),
                    None if recompute else l_info.get("inertial_i"),
                    cg_infos,
                    vg_infos,
                    l_info.get("is_robot", False),
                )
            ]
        )

        if l_info["parent_idx"] != -1:
            return

        # Compose the morph pose offset into the root link's world pose. The solver strips the matching offset in
        # relative getters, so the user frame is unchanged.
        offset_pos = np.array(morph.offset_pos, dtype=gs.np_float)
        offset_quat = np.array(morph.offset_quat, dtype=gs.np_float)
        l_info["pos"], l_info["quat"] = gu.transform_pos_quat_by_trans_quat(
            offset_pos, offset_quat, l_info["pos"], l_info["quat"]
        )

        align = morph.align if isinstance(morph, gs.options.morphs.FileMorph) else False
        if align is None:
            # Auto: True for basic rigid objects (root with free joint only, no articulated descendants). A link
            # mixing geoms with and without an authored density (see 'LinkInertialInfo.is_mass_explicit') quietly
            # declines auto-alignment; asking for it with an explicit align=True raises at build instead.
            geoms_with_density = sum(g_info.get("density") is not None for g_info in cg_infos)
            align = (
                not bool(l_info.get("is_robot", False))
                and all(j_info["type"] == gs.JOINT_TYPE.FREE for j_info in j_infos)
                and geoms_with_density in (0, len(cg_infos))
            )

        # A free body opting into alignment (or any primitive, which is inherently principal-axis and COM-centered) has
        # an exactly-diagonal joint-space mass matrix once anchored. The COM/principal-axis anchoring itself is deferred
        # to '_align_free_roots' after build, where the finalized composite inertia of the fixed subtree is known and
        # can be applied per heterogeneous variant. Here only the morph pose offset is composed (child link poses are
        # defined relative to it); the anchoring transform is folded in later.
        l_info["aligned"] = any(j_info["type"] == gs.JOINT_TYPE.FREE for j_info in j_infos) and (
            align or isinstance(morph, gs.options.morphs.Primitive)
        )

        # Refresh the free joint init_qpos to reflect the composed world pose.
        for j_info in j_infos:
            if j_info["type"] == gs.JOINT_TYPE.FREE:
                j_info["init_qpos"] = np.concatenate([l_info["pos"], l_info["quat"]])

        # Record the body-frame offset; the base link's also seeds the heterogeneous-variant offset.
        if not self._links_offset_quat:
            self._offset_pos, self._offset_quat = offset_pos, offset_quat
        self._links_offset_pos[link_idx - self._link_start] = offset_pos
        self._links_offset_quat[link_idx - self._link_start] = offset_quat

    @gs.assert_unbuilt
    def attach(
        self,
        parent_entity,
        parent_link_name: str | None = None,
        pos: Vec3FType | None = None,
        quat: UnitVec4FType | None = None,
    ):
        """
        Merge two entities to act as single one, by attaching the base link of this entity as a child of a given link of
        another entity.

        The merged pair is simulated as one kinematic tree, whose degrees of freedom must form one contiguous range.
        This method enforces it: instantiate attached entities consecutively, attach onto the last kinematic tree of a
        multi-tree parent, and attach all children of an entity onto the same tree.

        Parameters
        ----------
        parent_entity : genesis.Entity
            The entity in the scene that will be a parent of kinematic tree.
        parent_link_name : str
            The name of the link in the parent entity to be linked. Default to the latest link the parent kinematic
            tree.
        pos : array_like, shape (3,), optional
            Mounting position of this entity's base link relative to the parent link frame. If neither `pos` nor
            `quat` is provided, the base link keeps its current local pose, i.e. the morph `pos` / `quat` this entity
            was created with acts as the mounting transform. If only `quat` is provided, defaults to (0, 0, 0).
        quat : array_like, shape (4,), optional
            Mounting orientation (w, x, y, z) of this entity's base link relative to the parent link frame. If only
            `pos` is provided, defaults to the identity quaternion. Providing `pos` and/or `quat` overrides the pose
            inherited from the morph.
        """
        if self._is_attached:
            gs.raise_exception("Entity already attached.")
        if self._solver._requires_grad:
            gs.raise_exception("Attach is not supported yet when requires_grad is True.")

        is_mounting = pos is not None or quat is not None
        if is_mounting:
            mount_pos = gu.zero_pos() if pos is None else np.asarray(pos, dtype=gs.np_float)
            mount_quat = gu.identity_quat() if quat is None else np.asarray(quat, dtype=gs.np_float)
            if mount_pos.shape != (3,):
                gs.raise_exception(f"Mounting 'pos' must have shape (3,), got {mount_pos.shape}.")
            if mount_quat.shape != (4,):
                gs.raise_exception(f"Mounting 'quat' must have shape (4,) (w, x, y, z), got {mount_quat.shape}.")
            if np.linalg.norm(mount_quat) < gs.EPS:
                gs.raise_exception("Mounting 'quat' cannot be a zero-length quaternion.")
            mount_quat = gu.normalize(mount_quat)

        if not isinstance(parent_entity, KinematicEntity):
            gs.raise_exception("Parent entity must derive from 'KinematicEntity'.")

        if parent_entity is self:
            gs.raise_exception("Cannot attach entity to itself.")

        if parent_entity.idx > self.idx:
            gs.raise_exception("Parent entity must be instantiated before child entity.")

        # Attaching reshapes the kinematic tree, but the post-build pass that anchors each heterogeneous variant's
        # inertia and frame runs per free root and cannot follow a reparented (or absorbed) base link.
        if self._enable_heterogeneous or parent_entity._enable_heterogeneous:
            gs.raise_exception("Attaching heterogeneous entities is not supported.")

        # Check if base link was fixed but no longer is
        base_link = self.links[0]
        parent_link = parent_entity.get_link(parent_link_name)
        if base_link.is_fixed and not parent_link.is_fixed:
            if not self._batch_fixed_verts:
                gs.raise_exception(
                    "Attaching fixed-based entity to parent link requires setting Morph option 'batch_fixed_verts=True'."
                )

        # The merged kinematic tree must keep contiguous DOFs (each mass block is processed as one interval), so every
        # DOF-carrying link numbered between the target tree's root and this entity must already belong to that tree.
        # Each violation cause gets its own actionable error.
        root_link = self._solver.links[parent_link.root_idx]
        for link in self._solver.links[root_link.idx + 1 : self.link_start]:
            if link.n_dofs == 0 or link.root_idx == root_link.idx:
                continue
            foreign_root_link = self._solver.links[link.root_idx]
            if foreign_root_link.entity is root_link.entity:
                is_foreign_tree_merged = any(
                    other_link.root_idx == link.root_idx and other_link.entity is not foreign_root_link.entity
                    for other_link in self._solver.links[link.root_idx :]
                )
                if is_foreign_tree_merged:
                    gs.raise_exception(
                        "Attaching entities onto different kinematic trees of the same parent entity is not "
                        "supported. Load the parent's trees as separate entities."
                    )
                gs.raise_exception(
                    "Attaching an entity onto a kinematic tree that is not the last one declared in its file is not "
                    "supported. Declare the attached-onto tree last, or load the file's other trees as separate "
                    "entities."
                )
            gs.raise_exception(
                "Creating entities between attached entities is not supported. Instantiate attached entities "
                "consecutively."
            )

        # Remove all root joints if necessary.
        # The requires shifting joint and dof indices of all subsequent entities.
        # Note that we do not remove world link if any, but rather remove all base joints. This is to avoid altering
        # the parent entity by moving all fixed geometries to the new parent link.
        if not base_link.is_fixed:
            n_base_joints = base_link.n_joints
            n_base_dofs = base_link.n_dofs
            n_base_qs = base_link.n_qs

            base_link._n_joints = 0
            self._joints[0].clear()
            for entity in self._solver.entities[(self.idx + 1) :]:
                entity._joint_start -= n_base_joints
                entity._dof_start -= n_base_dofs
                entity._q_start -= n_base_qs
            for joint in self._solver.joints[self.joint_start :]:
                joint._idx -= n_base_joints
                joint._dof_start -= n_base_dofs
                joint._q_start -= n_base_qs
            for link in self._solver.links[(self.link_start + 1) :]:
                link._joint_start -= n_base_joints

            # Joint-equality constraints (e.g. mimic joints) reference joints by global index, which must stay
            # aligned with the dense joint ordering. Shift those references in lockstep with the joint re-indexing
            # above. Link-based equalities (connect/weld) are unaffected since no link is removed here.
            removed_joints_end = self.joint_start + n_base_joints
            for equality in self._solver.equalities:
                if equality.type == gs.EQUALITY_TYPE.JOINT:
                    if equality._eq_obj1id >= removed_joints_end:
                        equality._eq_obj1id -= n_base_joints
                    if equality._eq_obj2id >= removed_joints_end:
                        equality._eq_obj2id -= n_base_joints

        # Overwrite parent link
        base_link._parent_idx = parent_link.idx

        # Re-root the whole tree hanging from this entity's base link - scene-wide, because entities previously
        # attached into this one already share its root and must follow it (chained attaches may run in any order);
        # links of other trees declared in the same file keep their own root, as do the fixed flag and invweight.
        for link in self._solver.links:
            if link.root_idx == base_link.idx:
                link._root_idx = parent_link.root_idx
                link._is_fixed &= parent_link.is_fixed
                link._invweight = None

        # Apply the explicit mounting transform. Forward kinematics interprets the base link's local pose relative to
        # its (new) parent link, so overwriting it here mounts the entity at (pos, quat) in the parent link frame.
        # Compose with the morph frame offset exactly as '_align_link' does for the morph pose at load time, so the
        # relative pose getters keep stripping the offset correctly.
        if is_mounting:
            base_link._pos, base_link._quat = gu.transform_pos_quat_by_trans_quat(
                self._offset_pos, self._offset_quat, mount_pos, mount_quat
            )

        self._is_attached = True

    # ------------------------------------------------------------------------------------
    # ---------------------------------- control & io ------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        if in_backward:
            # use negative index because buffer length might not be full
            index = self._sim.cur_step_local - self._sim._steps_local
            self._tgt = self._tgt_buffer[index].copy()
        else:
            self._tgt_buffer.append(self._tgt.copy())

        update_tgt_while_set = self._update_tgt_while_set
        # Apply targets in the order of insertion
        for key in self._tgt.keys():
            data_kwargs = self._tgt[key]

            # We do not need zero velocity here because if it was true, [set_dofs_velocity] from zero_velocity would
            # be in [tgt]
            if "zero_velocity" in data_kwargs:
                data_kwargs["zero_velocity"] = False
            # Do not update [tgt], as input information is finalized at this point
            self._update_tgt_while_set = False

            # Every tracked setter replays uniformly from its taped kwargs, so dispatch by name (a rare legitimate
            # use of getattr, on our own vetted key set).
            if key[0] not in self._tgt_keys:
                gs.raise_exception(f"Invalid target key: {key[0]} not in {self._tgt_keys}")
            getattr(self, key[0])(**data_kwargs)

        self._tgt = dict()
        self._update_tgt_while_set = update_tgt_while_set

    def process_input_grad(self):
        index = self._sim.cur_step_local - self._sim._steps_local
        for key in reversed(self._tgt_buffer[index].keys()):
            data_kwargs = self._tgt_buffer[index][key]

            match key[0]:
                # We need to unpack the data_kwargs because [_backward_from_qd] only supports positional arguments.
                # Inputs are stored on the tape as passed by the user, so scalars and array-likes (valid setter
                # inputs that cannot carry a gradient) are filtered out with the tensor check.
                case "set_pos":
                    pos = data_kwargs.pop("pos")
                    if isinstance(pos, torch.Tensor) and pos.requires_grad:
                        pos._backward_from_qd(self.set_pos_grad, data_kwargs["envs_idx"], data_kwargs["relative"])

                case "set_quat":
                    quat = data_kwargs.pop("quat")
                    if isinstance(quat, torch.Tensor) and quat.requires_grad:
                        quat._backward_from_qd(self.set_quat_grad, data_kwargs["envs_idx"], data_kwargs["relative"])

                case "set_dofs_velocity":
                    velocity = data_kwargs.pop("velocity")
                    # [velocity] could be None when we want to zero the velocity (see set_dofs_velocity of RigidSolver)
                    if isinstance(velocity, torch.Tensor) and velocity.requires_grad:
                        velocity._backward_from_qd(
                            self.set_dofs_velocity_grad, data_kwargs["dofs_idx_local"], data_kwargs["envs_idx"]
                        )

                case "control_dofs_force":
                    force = data_kwargs.pop("force")
                    if isinstance(force, torch.Tensor) and force.requires_grad:
                        force._backward_from_qd(
                            self.set_dofs_force_grad, data_kwargs["dofs_idx_local"], data_kwargs["envs_idx"]
                        )

                case "control_dofs_velocity" | "control_dofs_position" | "control_dofs_position_velocity":
                    # PD control targets are replayed for primal correctness but have no input-gradient path.
                    for target in (data_kwargs.get("position"), data_kwargs.get("velocity")):
                        if isinstance(target, torch.Tensor) and target.requires_grad:
                            gs.raise_exception(
                                "Gradients with respect to PD control targets are not supported yet. Use "
                                "'control_dofs_force' for differentiable control inputs."
                            )

                case _:
                    gs.raise_exception(f"Invalid target key: {key[0]} not in {self._tgt_keys}")

    def save_ckpt(self, ckpt_name):
        if ckpt_name not in self._ckpt:
            self._ckpt[ckpt_name] = {}
        self._ckpt[ckpt_name]["_tgt_buffer"] = self._tgt_buffer.copy()
        self._tgt_buffer.clear()

    def load_ckpt(self, ckpt_name):
        self._tgt_buffer = self._ckpt[ckpt_name]["_tgt_buffer"].copy()

    def reset_grad(self):
        self._tgt_buffer.clear()

    @gs.assert_built
    def get_state(self):
        state = RigidEntityState(self, self._sim.cur_step_global)

        solver_state = self._solver.get_state()
        pos = solver_state.links_pos[:, self.base_link_idx]
        quat = solver_state.links_quat[:, self.base_link_idx]

        state._pos = pos
        state._quat = quat

        return state

    def _get_global_idx(self, idx_local, idx_local_max, idx_global_start=0, *, unsafe=False):
        # Handling default argument and special cases
        if idx_local is None:
            idx_global = range(idx_global_start, idx_local_max + idx_global_start)
        elif isinstance(idx_local, (slice, range)):
            idx_global = range(
                (idx_local.start or 0) + idx_global_start,
                (idx_local.stop if idx_local.stop is not None else idx_local_max) + idx_global_start,
                idx_local.step or 1,
            )
        elif isinstance(idx_local, (int, np.integer)):
            if idx_local < 0:
                idx_local = idx_local_max + idx_local
            idx_global = (idx_local + idx_global_start,)
        elif isinstance(idx_local, (list, tuple)):
            try:
                idx_global = [i + idx_global_start for i in idx_local]
            except TypeError:
                gs.raise_exception("Expecting a sequence of integers for `idx_local`.")
        else:
            # Increment may be slow when dealing with heterogenuous data, so it must be avoided if possible
            if idx_global_start > 0:
                idx_global = idx_local + idx_global_start
            else:
                idx_global = idx_local

        # Early return if unsafe
        if unsafe:
            return idx_global

        # Perform a bunch of sanity checks
        if isinstance(idx_global, torch.Tensor) and idx_global.dtype == torch.bool:
            if idx_global.shape != (idx_local_max - idx_global_start,):
                gs.raise_exception("Boolean masks must be 1D tensors of fixed size.")
            idx_global = idx_global_start + idx_global.nonzero()[:, 0]
        else:
            idx_global = torch.as_tensor(idx_global, dtype=gs.tc_int, device=gs.device).contiguous()
            ndim = idx_global.ndim
            if ndim == 0:
                idx_global = idx_global[None]
            elif ndim > 1:
                gs.raise_exception("Expecting a 1D tensor for local index.")

            # FIXME: This check is too expensive
            # if (idx_global < 0).any() or (idx_global >= idx_global_start + idx_local_max).any():
            #     gs.raise_exception("`idx_local` exceeds valid range.")

        return idx_global

    def get_joint(self, name=None, uid=None):
        """
        Get a RigidJoint object by name or uid.

        Parameters
        ----------
        name : str, optional
            The name of the joint. Defaults to None.
        uid : str, optional
            The uid of the joint. This can be a substring of the joint's uid. Defaults to None.

        Returns
        -------
        joint : RigidJoint
            The joint object.
        """

        if name is not None:
            for joint in self.joints:
                if joint.name == name:
                    return joint
            gs.raise_exception(
                f"Joint not found for name: {name}. Available joint names: {[joint.name for joint in self.joints]}."
            )

        elif uid is not None:
            for joint in self.joints:
                if uid in str(joint.uid):
                    return joint
            gs.raise_exception(f"Joint not found for uid: {uid}.")

        else:
            gs.raise_exception("Neither `name` nor `uid` is provided.")

    def get_link(self, name=None, uid=None):
        """
        Get a RigidLink object by name or uid.

        Parameters
        ----------
        name : str, optional
            The name of the link. Defaults to None.
        uid : str, optional
            The uid of the link. This can be a substring of the link's uid. Defaults to None.

        Returns
        -------
        link : RigidLink
            The link object.
        """

        if name is not None:
            for link in self._links:
                if link.name == name:
                    return link
            gs.raise_exception(
                f"Link not found for name: {name}. Available link names: {[link.name for link in self._links]}."
            )

        elif uid is not None:
            for link in self._links:
                if uid in str(link.uid):
                    return link
            gs.raise_exception(f"Link not found for uid: {uid}.")

        else:
            gs.raise_exception("Neither `name` nor `uid` is provided.")

    @gs.assert_built
    def get_pos(self, envs_idx=None, *, relative=True):
        """
        Returns position of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        relative : bool, optional
            Whether to report the position in the user frame, with the morph pose offset and inertial alignment
            stripped, rather than the world frame used by the solver. Defaults to True.

        Returns
        -------
        pos : torch.Tensor, shape (3,) or (n_envs, 3)
            The position of the entity's base link.
        """
        return self._solver.get_links_pos(self.base_link_idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_quat(self, envs_idx=None, *, relative=True):
        """
        Returns quaternion of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        relative : bool, optional
            Whether to report the orientation in the user frame, with the morph pose offset and inertial alignment
            stripped, rather than the world frame used by the solver. Defaults to True.

        Returns
        -------
        quat : torch.Tensor, shape (4,) or (n_envs, 4)
            The quaternion of the entity's base link.
        """
        return self._solver.get_links_quat(self.base_link_idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_vel(self, envs_idx=None):
        """
        Returns linear velocity of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        vel : torch.Tensor, shape (3,) or (n_envs, 3)
            The linear velocity of the entity's base link.
        """
        return self._solver.get_links_vel(self.base_link_idx, envs_idx)[..., 0, :]

    @gs.assert_built
    def get_ang(self, envs_idx=None):
        """
        Returns angular velocity of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        ang : torch.Tensor, shape (3,) or (n_envs, 3)
            The angular velocity of the entity's base link.
        """
        return self._solver.get_links_ang(self.base_link_idx, envs_idx)[..., 0, :]

    @gs.assert_built
    def get_links_pos(self, links_idx_local=None, envs_idx=None, *, relative=True):
        """
        Returns the position of a given reference point for all the entity's links.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        relative : bool, optional
            If True, return the user-frame position with the morph pose offset stripped (matching the morph 'pos'); if
            False, return the world-frame position used by the solver. Defaults to True.

        Returns
        -------
        pos : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The position of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_pos(links_idx, envs_idx, relative=relative)

    @gs.assert_built
    def get_links_quat(self, links_idx_local=None, envs_idx=None, *, relative=True):
        """
        Returns quaternion of all the entity's links.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        relative : bool, optional
            If True, return the user-frame orientation with the morph pose offset stripped (matching the morph
            'quat'/'euler'); if False, return the world-frame orientation used by the solver. Defaults to True.

        Returns
        -------
        quat : torch.Tensor, shape (n_links, 4) or (n_envs, n_links, 4)
            The quaternion of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_quat(links_idx, envs_idx, relative=relative)

    @gs.assert_built
    def get_vAABB(self, envs_idx=None):
        """
        Get the axis-aligned bounding box (AABB) of the entity in world frame by aggregating all the visual
        geometries associated with this entity.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        aabb : torch.Tensor, shape (2, 3) or (n_envs, 2, 3)
            The AABB of the entity, where `[:, 0] = min_corner (x_min, y_min, z_min)` and
            `[:, 1] = max_corner (x_max, y_max, z_max)`.
        """
        if self.n_vgeoms == 0:
            gs.raise_exception("Entity has no visual geometries.")

        # For heterogeneous entities, compute AABB per-environment respecting active_envs_idx
        if self._enable_heterogeneous:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            n_envs = len(envs_idx)
            aabb_min = torch.full((n_envs, 3), float("inf"), dtype=gs.tc_float, device=gs.device)
            aabb_max = torch.full((n_envs, 3), float("-inf"), dtype=gs.tc_float, device=gs.device)
            for vgeom in self.vgeoms:
                vgeom_aabb = vgeom.get_vAABB(envs_idx)
                active_mask = vgeom.active_envs_mask[envs_idx] if vgeom.active_envs_mask is not None else ()
                aabb_min[active_mask] = torch.minimum(aabb_min[active_mask], vgeom_aabb[active_mask, 0])
                aabb_max[active_mask] = torch.maximum(aabb_max[active_mask], vgeom_aabb[active_mask, 1])
            return torch.stack((aabb_min, aabb_max), dim=-2)

        aabbs = torch.stack([vgeom.get_vAABB(envs_idx) for vgeom in self._vgeoms], dim=-3)
        return torch.stack((aabbs[..., 0, :].min(dim=-2).values, aabbs[..., 1, :].max(dim=-2).values), dim=-2)

    @gs.assert_built
    def get_links_vel(self, links_idx_local=None, envs_idx=None):
        """
        Returns linear velocity of all the entity's links expressed at a given reference position in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        vel : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear velocity of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_vel(links_idx, envs_idx)

    @gs.assert_built
    def get_links_ang(self, links_idx_local=None, envs_idx=None):
        """
        Returns angular velocity of all the entity's links in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        ang : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The angular velocity of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_ang(links_idx, envs_idx)

    @gs.assert_built
    @tracked
    def set_pos(self, pos, envs_idx=None, *, zero_velocity=False, relative=True, skip_forward=False):
        """
        Set position of the entity's base link.

        Parameters
        ----------
        pos : array_like
            The position to set.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to False.
        relative : bool, optional
            Whether 'pos' is expressed in the user frame, with the morph pose offset and inertial alignment applied on
            top to reach the world frame used by the solver, rather than directly in the world frame. Defaults to True.
        skip_forward : bool, optional
            Whether to skip forward kinematics after setting position. Defaults to False.
        """
        # Throw exception in entity no longer has a "true" base link becaused it has attached
        if self._is_attached:
            gs.raise_exception("Impossible to set position of an entity that has been attached.")
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_base_links_pos(pos, self.base_link_idx, envs_idx, relative=relative, skip_forward=skip_forward)

    @gs.assert_built
    def set_pos_grad(self, envs_idx, relative, pos_grad):
        self._solver.set_base_links_pos_grad(self.base_link_idx, envs_idx, relative, pos_grad.data)

    @gs.assert_built
    @tracked
    def set_quat(self, quat, envs_idx=None, *, zero_velocity=False, relative=True, skip_forward=False):
        """
        Set quaternion of the entity's base link.

        Parameters
        ----------
        quat : array_like
            The quaternion to set.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to False.
        relative : bool, optional
            Whether 'quat' is expressed in the user frame, with the morph pose offset and inertial alignment applied on
            top to reach the world frame used by the solver, rather than directly in the world frame. Defaults to True.
        skip_forward : bool, optional
            Whether to skip forward kinematics after setting quaternion. Defaults to False.
        """
        if self._is_attached:
            gs.raise_exception("Impossible to set position of an entity that has been attached.")
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_base_links_quat(
            quat, self.base_link_idx, envs_idx, relative=relative, skip_forward=skip_forward
        )

    @gs.assert_built
    def set_quat_grad(self, envs_idx, relative, quat_grad):
        self._solver.set_base_links_quat_grad(self.base_link_idx, envs_idx, relative, quat_grad.data)

    @gs.assert_built
    def set_qpos(self, qpos, qs_idx_local=None, envs_idx=None, *, zero_velocity=False, skip_forward=False):
        """
        Set the entity's qpos.

        Parameters
        ----------
        qpos : array_like
            The qpos to set.
        qs_idx_local : None | array_like, optional
            The indices of the qpos to set. If None, all qpos will be set. Note that here this uses the local `q_idx`,
            not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to False.
        """
        qs_idx = self._get_global_idx(qs_idx_local, self.n_qs, self._q_start, unsafe=True)
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_qpos(qpos, qs_idx, envs_idx, skip_forward=skip_forward)

    @gs.assert_built
    @tracked
    def set_dofs_velocity(self, velocity=None, dofs_idx_local=None, envs_idx=None, *, skip_forward=False):
        """
        Set the entity's dofs' velocity.

        Parameters
        ----------
        velocity : array_like | None
            The velocity to set. Zero if not specified.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_velocity(velocity, dofs_idx, envs_idx, skip_forward=skip_forward)

    @gs.assert_built
    def set_dofs_position(self, position, dofs_idx_local=None, envs_idx=None, *, zero_velocity=False):
        """
        Set the entity's dofs' position.

        Parameters
        ----------
        position : array_like
            The position to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`,
            not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to False.
        """
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_position(position, dofs_idx, envs_idx)

    @gs.assert_built
    def get_qpos(self, qs_idx_local=None, envs_idx=None):
        """
        Get the entity's qpos.

        For a free joint, the qpos holds the world-frame pose of the (solver) link origin: it is the raw generalized
        coordinate and is never expressed in the user/offset frame, unlike the relative `get_pos`/`get_quat`.

        Parameters
        ----------
        qs_idx_local : None | array_like, optional
            The indices of the qpos to get. If None, all qpos will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        qpos : torch.Tensor, shape (n_qs,) or (n_envs, n_qs)
            The entity's qpos.
        """
        qs_idx = self._get_global_idx(qs_idx_local, self.n_qs, self._q_start, unsafe=True)
        return self._solver.get_qpos(qs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_velocity(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' velocity.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        velocity : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' velocity.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_velocity(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_position(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' position.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        position : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' position.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_position(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_limit(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the positional limits (min and max) for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The lower limit of the positional limits for the entity's dofs.
        upper_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The upper limit of the positional limits for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_limit(dofs_idx, envs_idx)

    @gs.assert_built
    def zero_all_dofs_velocity(self, envs_idx=None, *, skip_forward=False):
        """
        Zero the velocity of all the entity's dofs.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        self.set_dofs_velocity(None, slice(0, self._n_dofs), envs_idx, skip_forward=skip_forward)

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        if self._enable_heterogeneous:
            return "heterogeneous"
        return self._morph._identifier()

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_built(self):
        """
        Whether this rigid entity is built.
        """
        return self._is_built

    @property
    def is_attached(self):
        """
        Whether this rigid entity has already been attached to another one.
        """
        return self._is_attached

    @property
    def init_qpos(self):
        """The initial qpos of the entity."""
        if self.joints:
            return np.concatenate([joint.init_qpos for joint in self.joints])
        return np.array([])

    @property
    def n_qs(self):
        """The number of `q` (generalized coordinates) of the entity."""
        if self._is_built:
            return self._n_qs
        return sum(joint.n_qs for joint in self.joints)

    @property
    def n_links(self):
        """The number of `RigidLink` in the entity."""
        return len(self._links)

    @property
    def morph(self):
        """The morph of the entity.

        Raises an exception for heterogeneous entities, which have multiple morph variants: use morphs for all
        variants, or main_morph for the first one.
        """
        if self._enable_heterogeneous:
            gs.raise_exception(
                "Heterogeneous entities have multiple morph variants. Use `.morphs` for all variants, "
                "or `.main_morph` only when explicitly using the first variant."
            )
        return self._morph

    @property
    def main_morph(self):
        """The main morph of the entity (first morph for heterogeneous entities)."""
        return self._morph

    @property
    def morphs(self):
        """All morphs of the entity (main morph + heterogeneous variants if any)."""
        return gs.List((self._morph, *self._morph_heterogeneous))

    def _repr_morph(self):
        if self._enable_heterogeneous:
            return f"{len(self.morphs)} morph variants"
        return f"{self.main_morph}"

    @property
    def n_joints(self):
        """The number of `RigidJoint` in the entity."""
        return sum(map(len, self._joints))

    @property
    def n_dofs(self):
        """The number of degrees of freedom (DOFs) of the entity."""
        if self._is_built:
            return self._n_dofs
        return sum(joint.n_dofs for joint in self.joints)

    @property
    def n_vgeoms(self):
        """The number of vgeoms (visual geoms - `RigidVisGeom`) in the entity."""
        return sum(link.n_vgeoms for link in self._links)

    @property
    def n_vverts(self):
        """The number of vverts (visual vertices, from vgeoms) in the entity."""
        return sum([link.n_vverts for link in self._links])

    @property
    def n_vfaces(self):
        """The number of vfaces (visual faces, from vgeoms) in the entity."""
        return sum([link.n_vfaces for link in self._links])

    @property
    def base_link_idx(self):
        """The index of the entity's base link in the scene."""
        return self._link_start

    @property
    def link_start(self):
        """The index of the entity's first RigidLink in the scene."""
        return self._link_start

    @property
    def link_end(self):
        """The index of the entity's last RigidLink in the scene *plus one*."""
        return self._link_start + self.n_links

    @property
    def joint_start(self):
        """The index of the entity's first RigidJoint in the scene."""
        return self._joint_start

    @property
    def joint_end(self):
        """The index of the entity's last RigidJoint in the scene *plus one*."""
        return self._joint_start + self.n_joints

    @property
    def dof_start(self):
        """The index of the entity's first degree of freedom (DOF) in the scene."""
        return self._dof_start

    @property
    def dof_end(self):
        """The index of the entity's last degree of freedom (DOF) in the scene *plus one*."""
        return self._dof_start + self.n_dofs

    @property
    def vvert_start(self):
        """The index of the entity's first `vvert` (visual vertex) in the scene."""
        return self._vvert_start

    @property
    def vface_start(self):
        """The index of the entity's first `vface` (visual face) in the scene."""
        return self._vface_start

    @property
    def q_start(self):
        """The index of the entity's first `q` (generalized coordinates) in the scene."""
        return self._q_start

    @property
    def q_end(self):
        """The index of the entity's last `q` (generalized coordinates) in the scene *plus one*."""
        return self._q_start + self.n_qs

    @property
    def vgeoms(self):
        """The list of visual geoms (`RigidVisGeom`) in the entity."""
        if self.is_built:
            return self._vgeoms
        return gs.List(vgeom for link in self._links for vgeom in link.vgeoms)

    @gs.assert_built
    def set_vverts(self, vverts, envs_idx=None):
        """Override this entity's visual vertex positions for rendering and sensors.

        vverts is broadcast to (len(envs_idx), n_vverts, 3); scalars, (3,) and (n_vverts, 3) are accepted. vverts=None
        re-runs FK over the entity's vgeoms and writes the result back into the custom buffer. Requires the entity's
        morph to be created with enable_custom_vverts=True.
        """
        if self._enable_heterogeneous:
            gs.raise_exception("This method is not supported by heterogeneous entities.")
        if not self._morph.enable_custom_vverts:
            gs.raise_exception(
                "'set_vverts' requires the entity's morph to be created with 'enable_custom_vverts=True'."
            )
        self._solver.set_vverts(
            self._custom_vvert_start,
            self._custom_vvert_start + self.n_vverts,
            np.array([vg.idx for vg in self.vgeoms], dtype=gs.np_int),
            vverts,
            envs_idx,
        )

    @gs.assert_built
    def get_vverts(self, envs_idx=None):
        """Return a copy of this entity's visual vertex positions in world space.

        For entities created with enable_custom_vverts=True the positions are read from the engine custom buffer; for
        other entities they are computed on the fly from each vgeom's current pose applied to its rest-pose init_vverts.
        """
        if self._enable_heterogeneous:
            gs.raise_exception("This method is not supported by heterogeneous entities.")
        if self._morph.enable_custom_vverts:
            return self._solver.get_vverts(self._custom_vvert_start, self._custom_vvert_start + self.n_vverts, envs_idx)

        self._solver.update_vgeoms()
        vgeoms_pos = qd_to_torch(self._solver.dyn_state.vgeoms.pos, envs_idx, transpose=True, copy=None)
        vgeoms_quat = qd_to_torch(self._solver.dyn_state.vgeoms.quat, envs_idx, transpose=True, copy=None)
        parts = []
        for vgeom in self.vgeoms:
            init = torch.as_tensor(vgeom.init_vverts, dtype=gs.tc_float, device=gs.device)
            pos = vgeoms_pos[..., vgeom.idx, :].unsqueeze(-2)
            quat = vgeoms_quat[..., vgeom.idx, :].unsqueeze(-2)
            parts.append(gu.transform_by_trans_quat(init, pos, quat))
        tensor = torch.cat(parts, dim=-2)
        return tensor[0] if self._solver.n_envs == 0 else tensor

    @property
    def links(self) -> list[RigidLink]:
        """The list of links (`RigidLink`) in the entity."""
        return self._links

    @property
    def joints(self) -> list[RigidJoint]:
        """The list of joints (`RigidJoint`) in the entity."""
        return gs.List(chain.from_iterable(self._joints))

    @property
    def joints_by_links(self):
        """The list of joints (`RigidJoint`) in the entity grouped by parent links."""
        return self._joints

    @property
    def base_link(self) -> RigidLink:
        """The base link of the entity"""
        return self._links[0]

    @property
    def base_joint(self) -> RigidJoint:
        """The base joint of the entity"""
        return self._joints[0][0]


class RigidEntity(KinematicEntity):
    """
    Physics-enabled rigid entity (collision, constraints, dynamics).

    Inherits morphology, FK, IK, and DOF position get/set from KinematicEntity.
    Adds physics simulation methods: forces, velocities, contacts, etc.
    """

    if TYPE_CHECKING:
        material: gs.materials.Rigid
        _solver: "RigidSolver"

    def __init__(
        self,
        scene: "Scene",
        solver: "RigidSolver",
        material: Material,
        morph: Morph,
        surface: Surface,
        idx: int,
        idx_in_solver,
        link_start: int = 0,
        joint_start: int = 0,
        q_start=0,
        dof_start=0,
        geom_start=0,
        cell_start=0,
        vert_start=0,
        free_verts_state_start=0,
        fixed_verts_state_start=0,
        face_start=0,
        edge_start=0,
        vgeom_start=0,
        vvert_start=0,
        vface_start=0,
        custom_vvert_start=0,
        custom_vface_start=0,
        equality_start=0,
        visualize_contact: bool = False,
        morph_heterogeneous: list[Morph] | None = None,
        name: str | None = None,
    ):
        self._geom_start = geom_start
        self._cell_start = cell_start
        self._vert_start = vert_start
        self._face_start = face_start
        self._edge_start = edge_start
        self._free_verts_state_start = free_verts_state_start
        self._fixed_verts_state_start = fixed_verts_state_start
        self._equality_start = equality_start
        self._free_verts_idx_local = torch.tensor([], dtype=gs.tc_int, device=gs.device)
        self._fixed_verts_idx_local = torch.tensor([], dtype=gs.tc_int, device=gs.device)
        self._visualize_contact: bool = visualize_contact

        self._batch_fixed_verts: bool = morph.batch_fixed_verts

        super().__init__(
            scene,
            solver,
            material,
            morph,
            surface,
            idx,
            idx_in_solver,
            link_start,
            joint_start,
            q_start,
            dof_start,
            vgeom_start,
            vvert_start,
            vface_start,
            custom_vvert_start,
            custom_vface_start,
            morph_heterogeneous,
            name,
        )

    def _add_heterogeneous_variant(self, link, cg_infos, vg_infos):
        # Add collision geometries
        coup_links = self.material.coup_links
        for g_info in cg_infos:
            friction = self.material.friction
            if friction is None:
                friction = g_info.get("friction", gu.default_friction())
            friction_torsional = self.material.friction_torsional
            if friction_torsional is None:
                friction_torsional = g_info.get("friction_torsional", gu.default_friction_torsional())
            friction_rolling = self.material.friction_rolling
            if friction_rolling is None:
                friction_rolling = g_info.get("friction_rolling", gu.default_friction_rolling())
            needs_coup = self.material.needs_coup and (coup_links is None or link.name in coup_links)
            link._add_geom(
                mesh=g_info["mesh"],
                init_pos=g_info.get("pos", gu.zero_pos()),
                init_quat=g_info.get("quat", gu.identity_quat()),
                type=g_info["type"],
                friction=friction,
                friction_torsional=friction_torsional,
                friction_rolling=friction_rolling,
                sol_params=g_info["sol_params"],
                data=g_info.get("data"),
                needs_coup=needs_coup,
                contype=g_info["contype"],
                conaffinity=g_info["conaffinity"],
            )

        # Add visual geoms and record vgeom range via parent
        super()._add_heterogeneous_variant(link, cg_infos, vg_infos)

        # Record geom range on the link (vgeom range already recorded by parent)
        link._record_variant_geom_range(len(cg_infos))

    def _reassign_heterogeneous_indices(self):
        """Reassign collision and visual geom indices for multi-link heterogeneous entities."""
        # Reassign collision geom indices sequentially
        running_idx = self._geom_start
        running_cell = self._cell_start
        running_vert = self._vert_start
        running_face = self._face_start
        running_edge = self._edge_start
        running_free_vs = self._free_verts_state_start
        running_fixed_vs = self._fixed_verts_state_start

        for link in self._links:
            for geom in link.geoms:
                geom._idx = running_idx
                geom._cell_start = running_cell
                geom._vert_start = running_vert
                geom._face_start = running_face
                geom._edge_start = running_edge
                if link.is_fixed and not self._batch_fixed_verts:
                    geom._verts_state_start = running_fixed_vs
                    running_fixed_vs += geom.n_verts
                else:
                    geom._verts_state_start = running_free_vs
                    running_free_vs += geom.n_verts
                running_idx += 1
                running_cell += geom.n_cells
                running_vert += geom.n_verts
                running_face += geom.n_faces
                running_edge += geom.n_edges

        # Reassign visual geom indices and recompute variant ranges via parent
        super()._reassign_heterogeneous_indices()

        # Recompute collision geom variant ranges from counts and reassigned indices
        for link in self._links:
            if link._variant_geom_ranges is None:
                continue
            geom_counts = [end - start for start, end in link._variant_geom_ranges]
            geom_cursor = link.geoms[0].idx if link.geoms else 0
            link._variant_geom_ranges = []
            for count in geom_counts:
                link._variant_geom_ranges.append((geom_cursor, geom_cursor + count))
                geom_cursor += count

    def _on_heterogeneous_scene_variant_loaded(self, link, morph, v_l_info):
        """Store parsed inertial from the variant file for use during link._build()."""
        if link._variant_scene_inertial is None:
            link._variant_scene_inertial = []
        link._variant_scene_inertial.append(
            (
                morph,
                v_l_info.get("inertial_mass"),
                v_l_info.get("inertial_pos"),
                v_l_info.get("inertial_quat"),
                v_l_info.get("inertial_i"),
            )
        )

    def _load_model(self):
        self._equalities = gs.List()
        self._requires_jac_and_IK = self._morph.requires_jac_and_IK
        # MJCF and USD express in-model collision filtering (MJCF '<contact><exclude>', USD CollisionGroup /
        # FilteredPairsAPI) through synthesized contype/conaffinity bitmasks. Those masks are only consistent within
        # the entity: applied across entities, they would spuriously disable collision against geoms whose default
        # masks happen not to overlap (e.g. the ground plane).
        self._is_local_collision_mask = isinstance(self._morph, (gs.morphs.MJCF, gs.morphs.USD))

        super()._load_model()

    def _load_scene(self, morph, surface):
        from genesis.engine.couplers import IPCCoupler

        l_infos, links_j_infos, links_g_infos, eqs_info = self._parse_scene(morph, surface)

        # Make sure that the entity is not object
        if (
            isinstance(self.sim.coupler, IPCCoupler)
            and self.material.coup_type == "ipc_only"
            and any(l_info["is_robot"] for l_info in l_infos)
        ):
            gs.raise_exception("`RigidMaterial.coup_type='ipc_only'` only supported by rigid non-articulated objects.")

        # Add (link, joints, geoms) tuples sequentially
        for l_info, link_j_infos, link_g_infos in zip(l_infos, links_j_infos, links_g_infos):
            self._add_by_info(l_info, link_j_infos, link_g_infos, morph, surface)

        # Add equality constraints sequentially
        for eq_info in eqs_info:
            self._add_equality(
                name=eq_info["name"],
                type=eq_info["type"],
                objs_name=eq_info["objs_name"],
                data=eq_info["data"],
                sol_params=eq_info["sol_params"],
            )

    def _build(self):
        self._n_geoms = self.n_geoms
        self._geoms = self.geoms

        super()._build()

        verts_start = 0
        free_verts_idx_local, fixed_verts_idx_local = [], []
        for link in self.links:
            verts_idx = torch.arange(verts_start, verts_start + link.n_verts, dtype=gs.tc_int, device=gs.device)
            if link.is_fixed and not self._batch_fixed_verts:
                fixed_verts_idx_local.append(verts_idx)
            else:
                free_verts_idx_local.append(verts_idx)
            verts_start += link.n_verts
        if free_verts_idx_local:
            self._free_verts_idx_local = torch.cat(free_verts_idx_local)
        if fixed_verts_idx_local:
            self._fixed_verts_idx_local = torch.cat(fixed_verts_idx_local)
        self._n_free_verts = len(self._free_verts_idx_local)
        self._n_fixed_verts = len(self._fixed_verts_idx_local)

        self._init_jac_and_IK()

    def _init_jac_and_IK(self):
        # Build-time IK setup. Only the cheap, SNode-free metadata is computed here: the per-DOF joint limits in q-space
        # (a small NumPy array, also read by path planning) and the IK error dimensions. The Jacobian / IK scratch
        # `qd.field`s are allocated lazily on first use (in `get_jacobian` / `inverse_kinematics_multilink`) so that
        # entities which never query the Jacobian or run IK do not each add a per-entity field bundle to the SNode tree.
        if not self._requires_jac_and_IK:
            return

        if self.n_dofs == 0:
            return

        # compute joint limit in q space
        q_limit_lower = []
        q_limit_upper = []
        for joint in self.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                q_limit_lower.append(joint.dofs_limit[:3, 0])
                q_limit_lower.append(-np.ones(4))  # quaternion lower bound
                q_limit_upper.append(joint.dofs_limit[:3, 1])
                q_limit_upper.append(np.ones(4))  # quaternion upper bound
            elif joint.type == gs.JOINT_TYPE.FIXED:
                pass
            else:
                q_limit_lower.append(joint.dofs_limit[:, 0])
                q_limit_upper.append(joint.dofs_limit[:, 1])
        self.q_limit = np.stack(
            (np.concatenate(q_limit_lower), np.concatenate(q_limit_upper)), axis=0, dtype=gs.np_float
        )

        self._IK_n_tgts = self._solver._options.IK_max_targets
        self._IK_error_dim = self._IK_n_tgts * 6

        # The Jacobian and IK scratch fields are allocated lazily on first use; None marks them as not yet created.
        self._jacobian = None
        self._IK_mat = None

    def _add_by_info(self, l_info, j_infos, g_infos, morph, surface):
        if len(j_infos) > 1 and any(j_info["type"] in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.FIXED) for j_info in j_infos):
            raise ValueError(
                "Compounding joints of types 'FREE' or 'FIXED' with any other joint on the same body not supported"
            )

        parent_idx = l_info["parent_idx"]
        if parent_idx >= 0:
            parent_idx += self._link_start
        root_idx = l_info.get("root_idx")
        if root_idx is not None and root_idx >= 0:
            root_idx += self._link_start
        link_idx = self.n_links + self._link_start
        joint_start = self.n_joints + self._joint_start
        free_verts_start, fixed_verts_start = self._free_verts_state_start, self._fixed_verts_state_start
        for link in self.links:
            if link.is_fixed and not self._batch_fixed_verts:
                fixed_verts_start += link.n_verts
            else:
                free_verts_start += link.n_verts

        # Split and convexify collision geometry. Must be done before alignment so that
        # convexified geoms are used to compute the inertia frame.
        cg_infos, vg_infos = self._postprocess_geoms_info(morph, g_infos, l_info.get("is_robot", False))

        # Carry the morph pose offset into root links and align them to their collision-geometry COM/principal axes.
        self._align_link(l_info, j_infos, cg_infos, vg_infos, morph, link_idx)

        joints = self._create_joints(j_infos, link_idx, joint_start)

        # Add child link
        link = RigidLink(
            entity=self,
            name=l_info["name"],
            idx=link_idx,
            joint_start=joint_start,
            n_joints=len(j_infos),
            geom_start=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            free_verts_state_start=free_verts_start,
            fixed_verts_state_start=fixed_verts_start,
            vgeom_start=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            pos=l_info["pos"],
            quat=l_info["quat"],
            inertial_pos=l_info.get("inertial_pos"),
            inertial_quat=l_info.get("inertial_quat"),
            inertial_i=l_info.get("inertial_i"),
            inertial_mass=l_info.get("inertial_mass"),
            parent_idx=parent_idx,
            root_idx=root_idx,
            invweight=l_info.get("invweight"),
            visualize_contact=self.visualize_contact,
            is_robot=l_info.get("is_robot", root_idx != -1),
            aligned=l_info.get("aligned", False),
        )
        self._links.append(link)

        if not link.is_fixed and isinstance(morph, gs.options.morphs.FileMorph) and morph.recompute_inertia:
            link._inertial_pos = None
            link._inertial_quat = None
            link._inertial_i = None
            link._inertial_mass = None

        # Add visual geometries
        for g_info in vg_infos:
            link._add_vgeom(
                vmesh=g_info["vmesh"],
                init_pos=g_info.get("pos", gu.zero_pos()),
                init_quat=g_info.get("quat", gu.identity_quat()),
            )

        # Add collision geometries
        coup_links = self.material.coup_links
        for g_info in cg_infos:
            friction = self.material.friction
            if friction is None:
                friction = g_info.get("friction", gu.default_friction())
            friction_torsional = self.material.friction_torsional
            if friction_torsional is None:
                friction_torsional = g_info.get("friction_torsional", gu.default_friction_torsional())
            friction_rolling = self.material.friction_rolling
            if friction_rolling is None:
                friction_rolling = g_info.get("friction_rolling", gu.default_friction_rolling())
            needs_coup = self.material.needs_coup and (coup_links is None or link.name in coup_links)
            link._add_geom(
                mesh=g_info["mesh"],
                init_pos=g_info.get("pos", gu.zero_pos()),
                init_quat=g_info.get("quat", gu.identity_quat()),
                type=g_info["type"],
                friction=friction,
                friction_torsional=friction_torsional,
                friction_rolling=friction_rolling,
                sol_params=g_info["sol_params"],
                data=g_info.get("data"),
                needs_coup=needs_coup,
                contype=g_info["contype"],
                conaffinity=g_info["conaffinity"],
            )

        return link, joints

    def _add_equality(self, name, type, objs_name, data, sol_params):
        objs_id = []
        for obj_name in objs_name:
            if type == gs.EQUALITY_TYPE.CONNECT:
                obj_id = self.get_link(obj_name).idx
            elif type == gs.EQUALITY_TYPE.JOINT:
                obj_id = self.get_joint(obj_name).idx
            elif type == gs.EQUALITY_TYPE.WELD:
                obj_id = self.get_link(obj_name).idx
            else:
                gs.raise_exception(f"Equality type {type} not supported. Only CONNECT, JOINT, and WELD are supported.")
            objs_id.append(obj_id)

        equality = RigidEquality(
            entity=self,
            name=name,
            idx=self.n_equalities + self._equality_start,
            type=type,
            eq_obj1id=objs_id[0],
            eq_obj2id=objs_id[1],
            eq_data=data,
            sol_params=sol_params,
        )
        self._equalities.append(equality)
        return equality

    # ------------------------------------------------------------------------------------
    # --------------------------------- Jacobian & IK ------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_jacobian(self, link, local_point=None):
        """
        Get the spatial Jacobian for a point on a target link.

        Parameters
        ----------
        link : RigidLink
            The target link.
        local_point : torch.Tensor or None, shape (3,)
            Coordinates of the point in the link's *local* frame.
            If None, the link origin is used (back-compat).

        Returns
        -------
        jacobian : torch.Tensor
            The Jacobian matrix of shape (n_envs, 6, entity.n_dofs) or (6, entity.n_dofs) if n_envs == 0.
        """
        if not self._requires_jac_and_IK:
            gs.raise_exception(
                "Inverse kinematics and jacobian are disabled for this entity. Set `morph.requires_jac_and_IK` to True if you need them."
            )

        if self.n_dofs == 0:
            gs.raise_exception("Entity has zero dofs.")

        # Lazily allocate the Jacobian field on first use.
        if self._jacobian is None:
            self._jacobian = qd.field(dtype=gs.qd_float, shape=(6, self.n_dofs, self._solver._B))

        if local_point is None:
            sol = self._solver
            self._kernel_get_jacobian_zero(link.idx, sol.dyn_state, sol.dyn_info)
        else:
            p_local = torch.as_tensor(local_point, dtype=gs.tc_float, device=gs.device)
            if p_local.shape != (3,):
                gs.raise_exception("Must be a vector of length 3")
            sol = self._solver
            self._kernel_get_jacobian(link.idx, p_local, sol.dyn_state, sol.dyn_info)

        jacobian = qd_to_torch(self._jacobian, transpose=True, copy=True)
        if self._solver.n_envs == 0:
            jacobian = jacobian[0]

        return jacobian

    @qd.func
    def _impl_get_jacobian(
        self, tgt_link_idx, i_b, p_vec, dyn_state: array_class.DynState, dyn_info: array_class.DynInfo
    ):
        self._func_get_jacobian(
            tgt_link_idx, i_b, p_vec, qd.Vector.one(gs.qd_int, 3), qd.Vector.one(gs.qd_int, 3), dyn_state, dyn_info
        )

    @qd.kernel
    def _kernel_get_jacobian(
        self,
        tgt_link_idx: qd.i32,
        p_local: qd.types.ndarray(),
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
    ):
        p_vec = qd.Vector([p_local[0], p_local[1], p_local[2]], dt=gs.qd_float)
        for i_b in range(self._solver._B):
            self._impl_get_jacobian(tgt_link_idx, i_b, p_vec, dyn_state, dyn_info)

    @qd.kernel
    def _kernel_get_jacobian_zero(
        self, tgt_link_idx: qd.i32, dyn_state: array_class.DynState, dyn_info: array_class.DynInfo
    ):
        for i_b in range(self._solver._B):
            self._impl_get_jacobian(tgt_link_idx, i_b, qd.Vector.zero(gs.qd_float, 3), dyn_state, dyn_info)

    @qd.func
    def _func_get_jacobian(
        self,
        tgt_link_idx,
        i_b,
        p_local,
        pos_mask,
        rot_mask,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
    ):
        for i_row, i_d in qd.ndrange(6, self.n_dofs):
            self._jacobian[i_row, i_d, i_b] = 0.0

        tgt_link_pos = dyn_state.links.pos[tgt_link_idx, i_b] + gu.qd_transform_by_quat(
            p_local, dyn_state.links.quat[tgt_link_idx, i_b]
        )
        i_l = tgt_link_idx
        while i_l > -1:
            I_l = [i_l, i_b] if qd.static(self.solver._options.batch_links_info) else i_l

            for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                I_j = [i_j, i_b] if qd.static(self.solver._options.batch_joints_info) else i_j

                if dyn_info.joints.type[I_j] == gs.JOINT_TYPE.FIXED:
                    pass

                elif dyn_info.joints.type[I_j] == gs.JOINT_TYPE.REVOLUTE:
                    i_d_jac = dyn_info.joints.dof_start[I_j] - self._dof_start
                    rotation = dyn_state.joints.xaxis[i_j, i_b]
                    translation = rotation.cross(tgt_link_pos - dyn_state.joints.xanchor[i_j, i_b])

                    self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                    self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                    self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]
                    self._jacobian[3, i_d_jac, i_b] = rotation[0] * rot_mask[0]
                    self._jacobian[4, i_d_jac, i_b] = rotation[1] * rot_mask[1]
                    self._jacobian[5, i_d_jac, i_b] = rotation[2] * rot_mask[2]

                elif dyn_info.joints.type[I_j] == gs.JOINT_TYPE.PRISMATIC:
                    i_d_jac = dyn_info.joints.dof_start[I_j] - self._dof_start
                    translation = dyn_state.joints.xaxis[i_j, i_b]

                    self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                    self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                    self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]

                elif dyn_info.joints.type[I_j] == gs.JOINT_TYPE.FREE:
                    # translation
                    for i_d_ in qd.static(range(3)):
                        i_d_jac = dyn_info.joints.dof_start[I_j] + i_d_ - self._dof_start

                        self._jacobian[i_d_, i_d_jac, i_b] = 1.0 * pos_mask[i_d_]

                    # rotation
                    for i_d_ in qd.static(range(3)):
                        i_d = dyn_info.joints.dof_start[I_j] + i_d_ + 3
                        i_d_jac = i_d - self._dof_start
                        I_d = [i_d, i_b] if qd.static(self.solver._options.batch_dofs_info) else i_d
                        rotation = dyn_info.dofs.motion_ang[I_d]
                        translation = rotation.cross(tgt_link_pos - dyn_state.links.pos[i_l, i_b])

                        self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                        self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                        self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]
                        self._jacobian[3, i_d_jac, i_b] = rotation[0] * rot_mask[0]
                        self._jacobian[4, i_d_jac, i_b] = rotation[1] * rot_mask[1]
                        self._jacobian[5, i_d_jac, i_b] = rotation[2] * rot_mask[2]

            i_l = dyn_info.links.parent_idx[I_l]

    @gs.assert_built
    def inverse_kinematics(
        self,
        link,
        pos=None,
        quat=None,
        local_point=None,
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,  # 0.5 mm
        rot_tol=5e-3,  # 0.28 degree
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        dofs_idx_local=None,
        return_error=False,
        envs_idx=None,
    ):
        """
        Compute inverse kinematics for a single target link.

        The target `pos`/`quat` are interpreted in the world frame (the morph pose offset is not applied), matching
        the world-frame link poses returned by `forward_kinematics`.

        Parameters
        ----------
        link : RigidLink
            The link to be used as the end-effector.
        pos : None | array_like, shape (3,), optional
            The target position. If None, position error will not be considered. Defaults to None.
        quat : None | array_like, shape (4,), optional
            The target orientation. If None, orientation error will not be considered. Defaults to None.
        local_point : None | array_like, shape (3,), optional
            A point in the link's local frame to be positioned at `pos`. If None, the link origin is used.
            This is useful for positioning a tool center point (TCP) or fingertip that is offset from the link origin.
            Defaults to None (equivalent to [0, 0, 0]).
        init_qpos : None | array_like, shape (n_dofs,), optional
            Initial qpos used for solving IK. If None, the current qpos will be used. Defaults to None.
        respect_joint_limit : bool, optional
            Whether to respect joint limits. Defaults to True.
        max_samples : int, optional
            Number of resample attempts. Defaults to 50.
        max_solver_iters : int, optional
            Maximum number of solver iterations per sample. Defaults to 20.
        damping : float, optional
            Damping for damped least squares. Defaults to 0.01.
        pos_tol : float, optional
            Position tolerance for normalized position error (in meter). Defaults to 1e-4.
        rot_tol : float, optional
            Rotation tolerance for normalized rotation vector error (in radian). Defaults to 1e-4.
        pos_mask : list, shape (3,), optional
            Mask for position error. Defaults to [True, True, True]. E.g.: If you only care about position along x and y, you can set it to [True, True, False].
        rot_mask : list, shape (3,), optional
            Mask for rotation axis alignment. Defaults to [True, True, True]. E.g.: If you only want the link's Z-axis to be aligned with the Z-axis in the given quat, you can set it to [False, False, True].
        max_step_size : float, optional
            Maximum step size in q space for each IK solver step. Defaults to 0.5.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None. This is used to specify which dofs the IK is applied to.
        return_error : bool, optional
            Whether to return the final errorqpos. Defaults to False.
        envs_idx: None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.

        Returns
        -------
        qpos : array_like, shape (n_dofs,) or (n_envs, n_dofs) or (len(envs_idx), n_dofs)
            Solver qpos (joint positions).
        (optional) error_pose : array_like, shape (6,) or (n_envs, 6) or (len(envs_idx), 6)
            Pose error for each target. The 6-vector is [err_pos_x, err_pos_y, err_pos_z, err_rot_x, err_rot_y, err_rot_z]. Only returned if `return_error` is True.
        """
        if self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)

            if pos is not None:
                if pos.shape[0] != len(envs_idx):
                    gs.raise_exception("First dimension of `pos` must be equal to `scene.n_envs`.")
            if quat is not None:
                if quat.shape[0] != len(envs_idx):
                    gs.raise_exception("First dimension of `quat` must be equal to `scene.n_envs`.")

        ret = self.inverse_kinematics_multilink(
            links=[link],
            poss=[pos] if pos is not None else [],
            quats=[quat] if quat is not None else [],
            local_points=[local_point] if local_point is not None else [],
            init_qpos=init_qpos,
            respect_joint_limit=respect_joint_limit,
            max_samples=max_samples,
            max_solver_iters=max_solver_iters,
            damping=damping,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            pos_mask=pos_mask,
            rot_mask=rot_mask,
            max_step_size=max_step_size,
            dofs_idx_local=dofs_idx_local,
            return_error=return_error,
            envs_idx=envs_idx,
        )

        if return_error:
            qpos, error_pose = ret
            return qpos, error_pose[..., 0, :]
        return ret

    @gs.assert_built
    def inverse_kinematics_multilink(
        self,
        links,
        poss=None,
        quats=None,
        local_points=None,
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,  # 0.5 mm
        rot_tol=5e-3,  # 0.28 degree
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        dofs_idx_local=None,
        return_error=False,
        envs_idx=None,
    ):
        """
        Compute inverse kinematics for multiple target links.

        Parameters
        ----------
        links : list of RigidLink
            List of links to be used as the end-effectors.
        poss : list, optional
            List of target positions. If empty, position error will not be considered. Defaults to None.
        quats : list, optional
            List of target orientations. If empty, orientation error will not be considered. Defaults to None.
        local_points : list, optional
            List of local points (one per link) in each link's local frame to be positioned at the corresponding target position.
            If empty or None, link origins are used. Each element should be array_like of shape (3,) or None.
            This is useful for positioning tool center points (TCP) or fingertips that are offset from the link origin.
            Defaults to None.
        init_qpos : array_like, shape (n_dofs,), optional
            Initial qpos used for solving IK. If None, the current qpos will be used. Defaults to None.
        respect_joint_limit : bool, optional
            Whether to respect joint limits. Defaults to True.
        max_samples : int, optional
            Number of resample attempts. Defaults to 50.
        max_solver_iters : int, optional
            Maximum number of solver iterations per sample. Defaults to 20.
        damping : float, optional
            Damping for damped least squares. Defaults to 0.01.
        pos_tol : float, optional
            Position tolerance for normalized position error (in meter). Defaults to 1e-4.
        rot_tol : float, optional
            Rotation tolerance for normalized rotation vector error (in radian). Defaults to 1e-4.
        pos_mask : list, shape (3,), optional
            Mask for position error. Defaults to [True, True, True]. E.g.: If you only care about position along x and y, you can set it to [True, True, False].
        rot_mask : list, shape (3,), optional
            Mask for rotation axis alignment. Defaults to [True, True, True]. E.g.: If you only want the link's Z-axis to be aligned with the Z-axis in the given quat, you can set it to [False, False, True].
        max_step_size : float, optional
            Maximum step size in q space for each IK solver step. Defaults to 0.5.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None. This is used to specify which dofs the IK is applied to.
        return_error : bool, optional
            Whether to return the final errorqpos. Defaults to False.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.

        Returns
        -------
        qpos : array_like, shape (n_dofs,) or (n_envs, n_dofs) or (len(envs_idx), n_dofs)
            Solver qpos (joint positions).
        (optional) error_pose : array_like, shape (6,) or (n_envs, 6) or (len(envs_idx), 6)
            Pose error for each target. The 6-vector is [err_pos_x, err_pos_y, err_pos_z, err_rot_x, err_rot_y, err_rot_z]. Only returned if `return_error` is True.
        """
        from genesis.engine.solvers.rigid.abd.inverse_kinematics import kernel_rigid_entity_inverse_kinematics

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        if not self._requires_jac_and_IK:
            gs.raise_exception(
                "Inverse kinematics and jacobian are disabled for this entity. Set `morph.requires_jac_and_IK` to True if you need them."
            )

        if self.n_dofs == 0:
            gs.raise_exception("Entity has zero dofs.")

        # Lazily allocate the Jacobian and IK scratch fields on first use.
        if self._jacobian is None:
            self._jacobian = qd.field(dtype=gs.qd_float, shape=(6, self.n_dofs, self._solver._B))
        if self._IK_mat is None:
            # for storing intermediate results
            self._IK_mat = qd.field(dtype=gs.qd_float, shape=(self._IK_error_dim, self._IK_error_dim, self._solver._B))
            self._IK_inv = qd.field(dtype=gs.qd_float, shape=(self._IK_error_dim, self._IK_error_dim, self._solver._B))
            self._IK_L = qd.field(dtype=gs.qd_float, shape=(self._IK_error_dim, self._IK_error_dim, self._solver._B))
            self._IK_U = qd.field(dtype=gs.qd_float, shape=(self._IK_error_dim, self._IK_error_dim, self._solver._B))
            self._IK_y = qd.field(dtype=gs.qd_float, shape=(self._IK_error_dim, self._IK_error_dim, self._solver._B))
            self._IK_qpos_orig = qd.field(dtype=gs.qd_float, shape=(self.n_qs, self._solver._B))
            self._IK_qpos_best = qd.field(dtype=gs.qd_float, shape=(self.n_qs, self._solver._B))
            self._IK_delta_qpos = qd.field(dtype=gs.qd_float, shape=(self.n_dofs, self._solver._B))
            self._IK_vec = qd.field(dtype=gs.qd_float, shape=(self._IK_error_dim, self._solver._B))
            self._IK_err_pose = qd.field(dtype=gs.qd_float, shape=(self._IK_error_dim, self._solver._B))
            self._IK_err_pose_best = qd.field(dtype=gs.qd_float, shape=(self._IK_error_dim, self._solver._B))
            self._IK_jacobian = qd.field(dtype=gs.qd_float, shape=(self._IK_error_dim, self.n_dofs, self._solver._B))
            self._IK_jacobian_T = qd.field(dtype=gs.qd_float, shape=(self.n_dofs, self._IK_error_dim, self._solver._B))

        n_links = len(links)
        if n_links == 0:
            gs.raise_exception("Target link not provided.")

        poss = list(poss) if poss is not None else []
        if not poss:
            poss = [None for _ in range(n_links)]
            pos_mask = [False, False, False]
        elif len(poss) != n_links:
            gs.raise_exception("Accepting only `poss` with length equal to `links` or empty list.")

        quats = list(quats) if quats is not None else []
        if not quats:
            quats = [None for _ in range(n_links)]
            rot_mask = [False, False, False]
        elif len(quats) != n_links:
            gs.raise_exception("Accepting only `quats` with length equal to `links` or empty list.")

        # Process local_points - default to origin [0, 0, 0] for each link
        local_points = list(local_points) if local_points is not None else []
        if not local_points:
            local_points = [None for _ in range(n_links)]
        elif len(local_points) != n_links:
            gs.raise_exception("Accepting only `local_points` with length equal to `links` or empty list.")
        for i, lp in enumerate(local_points):
            if lp is None:
                lp = [0.0, 0.0, 0.0]
            local_points[i] = torch.as_tensor(lp, dtype=gs.tc_float, device=gs.device)
        local_points = torch.stack(local_points, dim=0)  # (n_links, 3)

        link_pos_mask, link_rot_mask = [], []
        for i, (pos, quat) in enumerate(zip(poss, quats)):
            if pos is None and quat is None:
                gs.raise_exception("At least one of `poss` or `quats` must be provided.")
            link_pos_mask.append(pos is not None)
            poss[i] = broadcast_tensor(pos, gs.tc_float, (len(envs_idx), 3), ("envs_idx", "")).contiguous()
            link_rot_mask.append(quat is not None)
            if quat is None:
                quat = gu.identity_quat()
            quats[i] = broadcast_tensor(quat, gs.tc_float, (len(envs_idx), 4), ("envs_idx", "")).contiguous()
        link_pos_mask = torch.tensor(link_pos_mask, dtype=gs.tc_int, device=gs.device)
        link_rot_mask = torch.tensor(link_rot_mask, dtype=gs.tc_int, device=gs.device)
        poss = torch.stack(poss, dim=0)
        quats = torch.stack(quats, dim=0)

        custom_init_qpos = init_qpos is not None
        init_qpos = broadcast_tensor(
            init_qpos, gs.tc_float, (len(envs_idx), self.n_qs), ("envs_idx", "qs_idx")
        ).contiguous()

        # pos and rot mask
        pos_mask = broadcast_tensor(pos_mask, gs.tc_bool, (3,)).contiguous()
        rot_mask = broadcast_tensor(rot_mask, gs.tc_bool, (3,)).contiguous()
        if (num_axis := rot_mask.sum()) == 1:
            rot_mask = ~rot_mask if gs.tc_bool == torch.bool else 1 - rot_mask
        elif num_axis == 2:
            gs.raise_exception("You can only align 0, 1 axis or all 3 axes.")

        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs)
        n_dofs = len(dofs_idx)
        if n_dofs == 0:
            gs.raise_exception("Target dofs not provided.")

        links_idx = torch.tensor([link.idx for link in links], dtype=gs.tc_int, device=gs.device)

        kernel_rigid_entity_inverse_kinematics(
            links_idx,
            dofs_idx,
            envs_idx,
            self,
            poss,
            quats,
            local_points,
            init_qpos,
            pos_mask,
            rot_mask,
            link_pos_mask,
            link_rot_mask,
            self._solver.dyn_state,
            self._solver.dyn_info,
            self._solver.rigid_info,
            self._solver.rigid_config,
            custom_init_qpos,
            max_samples,
            max_solver_iters,
            damping,
            pos_tol,
            rot_tol,
            max_step_size,
            respect_joint_limit,
        )

        qpos = qd_to_torch(self._IK_qpos_best, transpose=True, copy=True)
        qpos = qpos[0] if self._solver.n_envs == 0 else qpos[envs_idx]

        if return_error:
            error_pose = qd_to_torch(self._IK_err_pose_best, transpose=True, copy=True).reshape(
                (-1, self._IK_n_tgts, 6)
            )[:, :n_links]
            error_pose = error_pose[0] if self._solver.n_envs == 0 else error_pose[envs_idx]
            return qpos, error_pose
        return qpos

    @gs.assert_built
    def forward_kinematics(self, qpos, qs_idx_local=None, links_idx_local=None, envs_idx=None):
        """
        Compute forward kinematics for a single target link.

        The returned link poses are in the world frame (the morph pose offset is not stripped), consistent with the
        world-frame `qpos` input.

        Parameters
        ----------
        qpos : array_like, shape (n_qs,) or (n_envs, n_qs) or (len(envs_idx), n_qs)
            The joint positions.
        qs_idx_local : None | array_like, optional
            The indices of the qpos to set. If None, all qpos will be set. Defaults to None.
        links_idx_local : None | array_like, optional
            The indices of the links to get. If None, all links will be returned. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.

        Returns
        -------
        links_pos : array_like, shape (n_links, 3) or (n_envs, n_links, 3) or (len(envs_idx), n_links, 3)
            The positions of the links (link frame origins).
        links_quat : array_like, shape (n_links, 4) or (n_envs, n_links, 4) or (len(envs_idx), n_links, 4)
            The orientations of the links.
        """

        if self._solver.n_envs == 0:
            qpos = qpos[None]
            envs_idx = torch.zeros(1, dtype=gs.tc_int)
        else:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start)
        links_pos = torch.empty((len(envs_idx), len(links_idx), 3), dtype=gs.tc_float, device=gs.device)
        links_quat = torch.empty((len(envs_idx), len(links_idx), 4), dtype=gs.tc_float, device=gs.device)

        self._kernel_forward_kinematics(
            self._get_global_idx(qs_idx_local, self.n_qs, self._q_start),
            links_idx,
            envs_idx,
            links_pos,
            links_quat,
            qpos,
            self._solver.dyn_state,
            self._solver.dyn_info,
            self._solver.rigid_info,
            self._solver.rigid_config,
        )

        if self._solver.n_envs == 0:
            links_pos = links_pos[0]
            links_quat = links_quat[0]
        return links_pos, links_quat

    @qd.kernel
    def _kernel_forward_kinematics(
        self,
        qs_idx: qd.types.ndarray(),
        links_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        links_pos: qd.types.ndarray(),
        links_quat: qd.types.ndarray(),
        qpos: qd.types.ndarray(),
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        rigid_config: qd.template(),
    ):
        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_q_, i_b_ in qd.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
            # save original qpos
            # NOTE: reusing the IK_qpos_orig as cache (should not be a problem)
            self._IK_qpos_orig[qs_idx[i_q_], envs_idx[i_b_]] = rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]]
            # set new qpos
            rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos[i_b_, i_q_]

        # run FK
        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                self._idx_in_solver, envs_idx[i_b_], dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
            )

        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in qd.static(range(3)):
                links_pos[i_b_, i_l_, i] = dyn_state.links.pos[links_idx[i_l_], envs_idx[i_b_]][i]
            for i in qd.static(range(4)):
                links_quat[i_b_, i_l_, i] = dyn_state.links.quat[links_idx[i_l_], envs_idx[i_b_]][i]

        # restore original qpos
        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_q_, i_b_ in qd.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
            rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = self._IK_qpos_orig[qs_idx[i_q_], envs_idx[i_b_]]

        # run FK
        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                self._idx_in_solver, envs_idx[i_b_], dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
            )

    # ------------------------------------------------------------------------------------
    # --------------------------------- motion planing -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def plan_path(
        self,
        qpos_goal,
        qpos_start=None,
        max_nodes=2000,
        resolution=0.05,
        timeout=None,
        max_retry=1,
        smooth_path=True,
        num_waypoints=300,
        ignore_collision=False,
        planner="RRTConnect",
        envs_idx=None,
        return_valid_mask=False,
        *,
        ee_link_name=None,
        with_entity=None,
        **kwargs,
    ):
        """
        Plan a path from `qpos_start` to `qpos_goal`.

        Parameters
        ----------
        qpos_goal : array_like
            The goal state. [B, Nq] or [1, Nq]
        qpos_start : None | array_like, optional
            The start state. If None, the current state of the rigid entity will be used.
            Defaults to None. [B, Nq] or [1, Nq]
        resolution : float, optiona
            Joint-space resolution. It corresponds to the maximum distance between states to be checked
            for validity along a path segment.
        timeout : float, optional
            The max time to spend for each planning in seconds. Note that the timeout is not exact.
        max_retry : float, optional
            Maximum number of retry in case of timeout or convergence failure. Default to 1.
        smooth_path : bool, optional
            Whether to smooth the path after finding a solution. Defaults to True.
        num_waypoints : int, optional
            The number of waypoints to interpolate the path. If None, no interpolation will be performed.
            Defaults to 100.
        ignore_collision : bool, optional
            Whether to ignore collision checking during motion planning. Defaults to False.
        ignore_joint_limit : bool, optional
            This option has been deprecated and is not longer doing anything.
        planner : str, optional
            The name of the motion planning algorithm to use.
            Supported planners: 'RRT', 'RRTConnect'. Defaults to 'RRTConnect'.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.
        return_valid_mask: bool
            Obtain valid mask of the succesful planed path over batch.
        ee_link_name: str
            The name of the link, which we "attach" the object during the planning
        with_entity: RigidEntity
            The (non-articulated) object to "attach" during the planning

        Returns
        -------
        path : torch.Tensor
            A tensor of waypoints representing the planned path.
            Each waypoint is an array storing the entity's qpos of a single time step.
        is_invalid: torch.Tensor
            A tensor of boolean mask indicating the batch indices with failed plan.
        """
        if self._solver.n_envs > 0:
            n_envs = len(self._scene._sanitize_envs_idx(envs_idx))
        else:
            n_envs = 1

        if "ignore_joint_limit" in kwargs:
            gs.logger.warning("`ignore_joint_limit` is deprecated")

        ee_link_idx = None
        if ee_link_name is not None:
            assert with_entity is not None, "`with_entity` must be specified."
            ee_link_idx = self.get_link(ee_link_name).idx
        if with_entity is not None:
            assert ee_link_name is not None, "reference link of the robot must be specified."
            assert len(with_entity.links) == 1, "only non-articulated object is supported for now."

        # import here to avoid circular import
        from genesis.utils.path_planning import RRT, RRTConnect

        match planner:
            case "RRT":
                planner_obj = RRT(self)
            case "RRTConnect":
                planner_obj = RRTConnect(self)
            case _:
                gs.raise_exception(f"invalid planner {planner} specified.")

        path = torch.empty((num_waypoints, n_envs, self.n_qs), dtype=gs.tc_float, device=gs.device)
        is_invalid = torch.ones((n_envs,), dtype=torch.bool, device=gs.device)
        for i in range(1 + max_retry):
            retry_path, retry_is_invalid = planner_obj.plan(
                qpos_goal,
                qpos_start=qpos_start,
                resolution=resolution,
                timeout=timeout,
                max_nodes=max_nodes,
                smooth_path=smooth_path,
                num_waypoints=num_waypoints,
                ignore_collision=ignore_collision,
                envs_idx=envs_idx,
                ee_link_idx=ee_link_idx,
                obj_entity=with_entity,
            )
            # NOTE: update the previously failed path with the new results
            path[:, is_invalid] = retry_path[:, is_invalid]

            is_invalid &= retry_is_invalid
            if not is_invalid.any():
                break
            gs.logger.info(f"Planning failed. Retrying for {is_invalid.sum()} environments...")

        if self._solver.n_envs == 0:
            if return_valid_mask:
                return path.squeeze(1), ~is_invalid[0]
            return path.squeeze(1)

        if return_valid_mask:
            return path, ~is_invalid
        return path

    # ------------------------------------------------------------------------------------
    # ---------------------------------- control & io ------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_state(self):
        state = RigidEntityState(self, self._sim.cur_step_global)

        solver_state = self._solver.get_state()
        pos = solver_state.links_pos[:, self.base_link_idx]
        quat = solver_state.links_quat[:, self.base_link_idx]

        state._pos = pos
        state._quat = quat

        return state

    # ------------------------------------------------------------------------------------
    # -------------------------------- vertices / AABB -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_AABB(self, envs_idx=None, *, allow_fast_approx: bool = False):
        """
        Get the axis-aligned bounding box (AABB) of the entity in world frame by aggregating all the collision
        geometries associated with this entity.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        allow_fast_approx : bool
            Whether to allow fast approximation for efficiency if supported, i.e. 'LegacyCoupler' is enabled. In this
            case, each collision geometry is approximated by their pre-computed AABB in geometry-local frame, which is
            more efficiency but inaccurate.

        Returns
        -------
        aabb : torch.Tensor, shape (2, 3) or (n_envs, 2, 3)
            The AABB of the entity, where `[:, 0] = min_corner (x_min, y_min, z_min)` and
            `[:, 1] = max_corner (x_max, y_max, z_max)`.
        """
        from genesis.engine.couplers import LegacyCoupler

        if self.n_geoms == 0:
            gs.raise_exception("Entity has no collision geometries.")

        # Already computed internally by the solver. Let's access it directly for efficiency.
        if allow_fast_approx and isinstance(self.sim.coupler, LegacyCoupler):
            return self._solver.get_AABB(entities_idx=[self._idx_in_solver], envs_idx=envs_idx)[..., 0, :]

        # For heterogeneous entities, compute AABB per-environment respecting active_envs_idx.
        # FIXME: Remove this branch after implementing 'get_verts'.
        if self._enable_heterogeneous and self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            n_envs = len(envs_idx)
            aabb_min = torch.full((n_envs, 3), float("inf"), dtype=gs.tc_float, device=gs.device)
            aabb_max = torch.full((n_envs, 3), float("-inf"), dtype=gs.tc_float, device=gs.device)
            for geom in self.geoms:
                geom_aabb = geom.get_AABB()
                active_mask = geom.active_envs_mask[envs_idx] if geom.active_envs_mask is not None else ()
                aabb_min[active_mask] = torch.minimum(aabb_min[active_mask], geom_aabb[envs_idx[active_mask], 0])
                aabb_max[active_mask] = torch.maximum(aabb_max[active_mask], geom_aabb[envs_idx[active_mask], 1])
            return torch.stack((aabb_min, aabb_max), dim=-2)

        # Compute the AABB on-the-fly based on the positions of all the vertices
        verts = self.get_verts()[envs_idx if envs_idx is not None else ()]
        return torch.stack((verts.min(dim=-2).values, verts.max(dim=-2).values), dim=-2)

    def get_aabb(self):
        raise DeprecationError("This method has been removed. Please use 'get_AABB()' instead.")

    @gs.assert_built
    def get_links_pos(
        self,
        links_idx_local=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com", "root_com"] = "link_origin",
        relative=True,
    ):
        """
        Returns the position of a given reference point for all the entity's links.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        ref: "link_origin" | "link_com" | "root_com"
            The reference point being used to express the position of each link.
            * "root_com": center of mass of the sub-entities to which the link belongs. As a reminder, a single
              kinematic tree (aka. 'RigidEntity') may compromise multiple "physical" entities, i.e. a kinematic tree
              that may have at most one free joint, at its root.
        relative : bool, optional
            If True, strip the morph pose offset to return the user-frame position; this only affects
            ref="link_origin", since the offset is defined on the link origin. If False, return the world frame.
            Defaults to True.

        Returns
        -------
        pos : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The position of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_pos(links_idx, envs_idx, ref=ref, relative=relative)

    @gs.assert_built
    def get_links_vel(
        self, links_idx_local=None, envs_idx=None, *, ref: Literal["link_origin", "link_com"] = "link_origin"
    ):
        """
        Returns linear velocity of all the entity's links expressed at a given reference position in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        ref: "link_origin" | "link_com"
            The reference point being used to expressed the velocity of each link.

        Returns
        -------
        vel : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear velocity of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_vel(links_idx, envs_idx, ref=ref)

    @gs.assert_built
    def get_links_acc(self, links_idx_local=None, envs_idx=None):
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_acc(links_idx, envs_idx)

    @gs.assert_built
    def get_links_acc_ang(self, links_idx_local=None, envs_idx=None):
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_acc_ang(links_idx, envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------- links mass properties --------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_links_inertial_mass(self, links_idx_local=None, envs_idx=None):
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_inertial_mass(links_idx, envs_idx)

    @gs.assert_built
    def get_links_invweight(self, links_idx_local=None, envs_idx=None):
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_invweight(links_idx, envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------- base pos/quat get/set --------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    @tracked
    def set_pos(self, pos, envs_idx=None, *, zero_velocity=True, relative=True, skip_forward=False):
        """
        Set position of the entity's base link.

        Parameters
        ----------
        pos : array_like
            The position to set.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        relative : bool, optional
            Whether 'pos' is expressed in the user frame, with the morph pose offset and inertial alignment applied on
            top to reach the world frame used by the solver, rather than directly in the world frame. Defaults to True.
        skip_forward : bool, optional
            Whether to skip forward kinematics after setting position. Defaults to False.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type is not None and self.base_link.is_fixed:
            gs.raise_exception(
                "This method is only supported by `RigidMaterial.coup_type=None` for fixed-based rigid entities."
            )
        super().set_pos(pos, envs_idx, zero_velocity=zero_velocity, relative=relative, skip_forward=skip_forward)

    @gs.assert_built
    def set_pos_grad(self, envs_idx, relative, pos_grad):
        self._solver.set_base_links_pos_grad(self.base_link_idx, envs_idx, relative, pos_grad.data)

    @gs.assert_built
    @tracked
    def set_quat(self, quat, envs_idx=None, *, zero_velocity=True, relative=True, skip_forward=False):
        """
        Set quaternion of the entity's base link.

        Parameters
        ----------
        quat : array_like
            The quaternion to set.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        relative : bool, optional
            Whether 'quat' is expressed in the user frame, with the morph pose offset and inertial alignment applied on
            top to reach the world frame used by the solver, rather than directly in the world frame. Defaults to True.
        skip_forward : bool, optional
            Whether to skip forward kinematics after setting quaternion. Defaults to False.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type is not None and self.base_link.is_fixed:
            gs.raise_exception(
                "This method is only supported by `RigidMaterial.coup_type=None` for fixed-based rigid entities."
            )
        super().set_quat(quat, envs_idx, zero_velocity=zero_velocity, relative=relative, skip_forward=skip_forward)

    @gs.assert_built
    def set_quat_grad(self, envs_idx, relative, quat_grad):
        self._solver.set_base_links_quat_grad(self.base_link_idx, envs_idx, relative, quat_grad.data)

    @gs.assert_built
    def get_verts(self):
        """
        Get the all vertices of the entity based on collision geometries.

        Returns
        -------
        verts : torch.Tensor, shape (n_envs, n_verts, 3)
            The vertices of the entity.
        """
        if self._enable_heterogeneous:
            gs.raise_exception("This method is not supported by heterogeneous entities.")

        self._solver.update_verts_for_geoms(slice(self.geom_start, self.geom_end))

        n_fixed_verts, n_free_vertices = self._n_fixed_verts, self._n_free_verts
        tensor = torch.empty((self._solver._B, n_fixed_verts + n_free_vertices, 3), dtype=gs.tc_float, device=gs.device)

        if n_fixed_verts > 0:
            verts_idx = slice(self._fixed_verts_state_start, self._fixed_verts_state_start + n_fixed_verts)
            fixed_verts_state = qd_to_torch(self._solver.dyn_state.fixed_verts.pos, verts_idx)
            tensor[:, self._fixed_verts_idx_local] = fixed_verts_state
        if n_free_vertices > 0:
            verts_idx = slice(self._free_verts_state_start, self._free_verts_state_start + n_free_vertices)
            free_verts_state = qd_to_torch(self._solver.dyn_state.free_verts.pos, None, verts_idx, transpose=True)
            tensor[:, self._free_verts_idx_local] = free_verts_state

        if self._solver.n_envs == 0:
            tensor = tensor[0]
        return tensor

    # ------------------------------------------------------------------------------------
    # --------------------------------- qpos get/set -------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def set_qpos(self, qpos, qs_idx_local=None, envs_idx=None, *, zero_velocity=True, skip_forward=False):
        """
        Set the entity's qpos.

        Parameters
        ----------
        qpos : array_like
            The qpos to set.
        qs_idx_local : None | array_like, optional
            The indices of the qpos to set. If None, all qpos will be set. Note that here this uses the local `q_idx`,
            not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type == "external_articulation":
            gs.raise_exception("This method is not supported by `RigidMaterial.coup_type='external_articulation'`.")
        super().set_qpos(qpos, qs_idx_local, envs_idx, zero_velocity=zero_velocity, skip_forward=skip_forward)

    @gs.assert_built
    def set_dofs_kp(self, kp, dofs_idx_local=None, envs_idx=None):
        """
        Set the entity's dofs' positional gains for the PD controller.

        Parameters
        ----------
        kp : array_like
            The positional gains to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_kp(kp, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_kv(self, kv, dofs_idx_local=None, envs_idx=None):
        """
        Set the entity's dofs' velocity gains for the PD controller.

        Parameters
        ----------
        kv : array_like
            The velocity gains to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_kv(kv, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_act_gain(self, act_gain, dofs_idx_local=None, envs_idx=None):
        """
        Set the actuator gain for the entity's dofs. Invalidates PD-reducibility.

        Parameters
        ----------
        act_gain : array_like
            The actuator gain values.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_act_gain(act_gain, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_act_bias(self, bias0, bias1, bias2, dofs_idx_local=None, envs_idx=None):
        """
        Set the actuator bias for the entity's dofs.

        Parameters
        ----------
        bias0 : array_like
            Constant bias term.
        bias1 : array_like
            Position coefficient.
        bias2 : array_like
            Velocity coefficient.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_act_bias(bias0, bias1, bias2, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_force_range(self, lower, upper, dofs_idx_local=None, envs_idx=None):
        """
        Set the entity's dofs' force range.

        Parameters
        ----------
        lower : array_like
            The lower bounds of the force range.
        upper : array_like
            The upper bounds of the force range.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_force_range(lower, upper, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_stiffness(self, stiffness, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_stiffness(stiffness, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_invweight(self, invweight, dofs_idx_local=None, envs_idx=None):
        raise DeprecationError(
            "This method has been removed because dof invweights are supposed to be a by-product of link properties "
            "(mass, pose, and inertia matrix), joint placements, and dof armatures. Please consider using the "
            "considering setters instead."
        )

    @gs.assert_built
    def set_dofs_armature(self, armature, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_armature(armature, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_damping(self, damping, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_damping(damping, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_frictionloss(self, frictionloss, dofs_idx_local=None, envs_idx=None):
        """
        Set the entity's dofs' friction loss.
        Parameters
        ----------
        frictionloss : array_like
            The friction loss values to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_frictionloss(frictionloss, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_velocity_grad(self, dofs_idx_local, envs_idx, velocity_grad):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_velocity_grad(dofs_idx, envs_idx, velocity_grad.data)

    @gs.assert_built
    def set_dofs_force_grad(self, dofs_idx_local, envs_idx, force_grad):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_force_grad(dofs_idx, envs_idx, force_grad.data)

    # ------------------------------------------------------------------------------------
    # ----------------------------- DOF property setters ---------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def set_dofs_position(self, position, dofs_idx_local=None, envs_idx=None, *, zero_velocity=True):
        """
        Set the entity's dofs' position.

        Parameters
        ----------
        position : array_like
            The position to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`,
            not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type == "external_articulation":
            gs.raise_exception("This method is not supported by `RigidMaterial.coup_type='external_articulation'`.")
        super().set_dofs_position(position, dofs_idx_local, envs_idx, zero_velocity=zero_velocity)

    # ------------------------------------------------------------------------------------
    # ---------------------------------- PD control --------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    @tracked
    def control_dofs_force(self, force, dofs_idx_local=None, envs_idx=None):
        """
        Control the entity's dofs' motor force. This is used for force/torque control.

        Parameters
        ----------
        force : array_like
            The force to apply.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type == "ipc_only":
            gs.raise_exception("This method is not supported for `coup_type='ipc_only'` entities.")

        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_force(force, dofs_idx, envs_idx)

    @gs.assert_built
    @tracked
    def control_dofs_velocity(self, velocity, dofs_idx_local=None, envs_idx=None):
        """
        Set the PD controller's target velocity for the entity's dofs. This is used for velocity control.

        Parameters
        ----------
        velocity : array_like
            The target velocity to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type == "ipc_only":
            gs.raise_exception("This method is not supported for `coup_type='ipc_only'` entities.")

        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_velocity(velocity, dofs_idx, envs_idx)

    @gs.assert_built
    @tracked
    def control_dofs_position(self, position, dofs_idx_local=None, envs_idx=None):
        """
        Set the position controller's target position for the entity's dofs. The controller is a proportional term
        plus a velocity damping term (virtual friction).

        Parameters
        ----------
        position : array_like
            The target position to set.
        dofs_idx_local : array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type == "ipc_only":
            gs.raise_exception("This method is not supported for `coup_type='ipc_only'` entities.")

        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_position(position, dofs_idx, envs_idx)

    @gs.assert_built
    @tracked
    def control_dofs_position_velocity(self, position, velocity, dofs_idx_local=None, envs_idx=None):
        """
        Set a PD controller's target position and velocity for the entity's dofs. This is used for position control.

        Parameters
        ----------
        position : array_like
            The target position to set.
        velocity : array_like
            The target velocity
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type == "ipc_only":
            gs.raise_exception("This method is not supported for `coup_type='ipc_only'` entities.")

        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_position_velocity(position, velocity, dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_control_force(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' internal control force, computed based on the position/velocity control command.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        control_force : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' internal control force.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_control_force(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_force(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' internal force at the current time step.

        Note
        ----
        Different from `get_dofs_control_force`, this function returns the actual internal force experienced by all the dofs at the current time step.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        force : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' force.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_force(dofs_idx, envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------- DOF property getters ---------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_dofs_kp(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the positional gain (kp) for the entity's dofs used by the PD controller.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        kp : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The positional gain (kp) for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_kp(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_kv(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the velocity gain (kv) for the entity's dofs used by the PD controller.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        kv : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The velocity gain (kv) for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_kv(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_act_gain(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the actuator gain for the entity's dofs.

        Returns
        -------
        act_gain : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_act_gain(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_act_bias(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the actuator bias [constant, pos_coeff, vel_coeff] for the entity's dofs.

        Returns
        -------
        bias0, bias1, bias2 : tuple of torch.Tensor
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_act_bias(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_force_range(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the force range (min and max limits) for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The lower limit of the force range for the entity's dofs.
        upper_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The upper limit of the force range for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_force_range(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_stiffness(self, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_stiffness(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_invweight(self, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_invweight(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_armature(self, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_armature(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_damping(self, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_damping(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_frictionloss(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the friction loss for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        frictionloss : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The friction loss for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_frictionloss(dofs_idx, envs_idx)

    # ------------------------------------------------------------------------------------
    # -------------------------------- physics queries -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_mass_mat(self, envs_idx=None, decompose=False):
        dofs_idx = self._get_global_idx(None, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_mass_mat(dofs_idx, envs_idx, decompose)

    @gs.assert_built
    def get_kinetic_energy(self, envs_idx=None) -> torch.Tensor:
        """Get the total kinetic energy of the entity in Joules [J] (translational + rotational).

        Computed using the joint-space mass matrix: ``KE = 0.5 * dq^T * M(q) * dq``.
        The mass matrix is recomputed to include motor armature on the diagonal.

        Note
        ----
        When the ``approximate_implicitfast`` integrator is used, this method forces recomputation of the
        mass matrix to exclude implicit damping terms added during integration. Other integrators do not
        require this recomputation.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        kinetic_energy : torch.Tensor, shape () or (n_envs,)
        """
        if self._solver.rigid_config.integrator == gs.integrator.approximate_implicitfast:
            from genesis.engine.solvers.rigid.abd.forward_dynamics import kernel_compute_mass_matrix

            kernel_compute_mass_matrix(
                self._solver.dyn_state,
                self._solver.dyn_info,
                self._solver.rigid_info,
                self._solver.rigid_config,
                decompose=False,
            )
        mass_mat = self.get_mass_mat(envs_idx=envs_idx)
        dofs_vel = self.get_dofs_velocity(envs_idx=envs_idx)
        Mv = torch.matmul(mass_mat, dofs_vel.unsqueeze(-1)).squeeze(-1)
        return 0.5 * torch.sum(dofs_vel * Mv, dim=-1)

    @gs.assert_built
    def get_potential_energy(self, envs_idx=None) -> torch.Tensor:
        """Get the total gravitational potential energy of the entity in Joules [J].

        Computed as the sum over all links: ``PE = sum_i(m_i * g^T * p_i)``, where ``p_i`` is the
        center-of-mass position of link *i* and ``g`` is the gravity vector obtained from the solver.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        potential_energy : torch.Tensor, shape () or (n_envs,)
        """
        gravity = self._solver.get_gravity(envs_idx=envs_idx)  # (3,) or (n_envs, 3)
        links_pos = self.get_links_pos(envs_idx=envs_idx, ref="link_com")  # (..., n_links, 3)
        # Link masses are static properties (not batched per environment),
        # so always fetch without envs_idx to avoid indexing conflicts.
        links_mass = self.get_links_inertial_mass()  # (n_links,)

        # PE_i = m_i * g^T * p_i => PE = sum_i(m_i * (g . p_i))
        # g is (..., 3), links_pos is (..., n_links, 3) -> broadcast g to (..., 1, 3)
        g_dot_p = torch.sum(gravity.unsqueeze(-2) * links_pos, dim=-1)  # (..., n_links)
        return -torch.sum(links_mass * g_dot_p, dim=-1)

    @gs.assert_built
    def get_total_energy(self, envs_idx=None) -> torch.Tensor:
        """Get the total mechanical energy of the entity in Joules [J] (kinetic + potential).

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        total_energy : torch.Tensor, shape () or (n_envs,)
        """
        return self.get_kinetic_energy(envs_idx=envs_idx) + self.get_potential_energy(envs_idx=envs_idx)

    @gs.assert_built
    def detect_collision(self, env_idx=0):
        """
        Detects collision for the entity. This only supports a single environment.

        Note
        ----
        This function re-detects real-time collision for the entity, so it doesn't rely on scene.step() and can be used for applications like motion planning, which doesn't require physical simulation during state sampling.

        Parameters
        ----------
        env_idx : int, optional
            The index of the environment. Defaults to 0.
        """

        all_collision_pairs = self._solver.detect_collision(env_idx)
        collision_pairs = all_collision_pairs[
            np.logical_and(all_collision_pairs >= self.geom_start, all_collision_pairs < self.geom_end).any(axis=1)
        ]
        return collision_pairs

    @gs.assert_built
    def get_contacts(self, with_entity=None, exclude_self_contact=False):
        """
        Returns contact information computed during the most recent `scene.step()`.
        If `with_entity` is provided, only returns contact information involving the caller and the specified entity.
        Otherwise, returns all contact information involving the caller entity.
        When `with_entity` is `self`, it will return the self-collision only.

        The returned dict contains the following keys (a contact pair consists of two geoms: A and B):

        - 'geom_a'     : The global geom index of geom A in the contact pair.
                        (actual geom object can be obtained by scene.rigid_solver.geoms[geom_a])
        - 'geom_b'     : The global geom index of geom B in the contact pair.
                        (actual geom object can be obtained by scene.rigid_solver.geoms[geom_b])
        - 'link_a'     : The global link index of link A (that contains geom A) in the contact pair.
                        (actual link object can be obtained by scene.rigid_solver.links[link_a])
        - 'link_b'     : The global link index of link B (that contains geom B) in the contact pair.
                        (actual link object can be obtained by scene.rigid_solver.links[link_b])
        - 'position'   : The contact position in world frame.
        - 'force_a'    : The contact force applied to geom A.
        - 'force_b'    : The contact force applied to geom B.
        - 'valid_mask' : A boolean mask indicating whether the contact information is valid.
                        (Only when scene is parallelized)

        The shape of each entry is (n_envs, n_contacts, ...) for scene with parallel envs
                               and (n_contacts, ...) for non-parallelized scene.

        Parameters
        ----------
        with_entity : RigidEntity, optional
            The entity to check contact with. Defaults to None.
        exclude_self_contact: bool
            Exclude the self collision from the returning contacts. Defaults to False.

        Returns
        -------
        contact_info : dict
            The contact information.
        """
        contact_data = self._solver.collider.get_contacts(as_tensor=True, to_torch=True)

        logical_operation = torch.logical_xor if exclude_self_contact else torch.logical_or
        if with_entity is not None and self.idx == with_entity.idx:
            if exclude_self_contact:
                gs.raise_exception("`with_entity` is self but `exclude_self_contact` is True.")
            logical_operation = torch.logical_and

        valid_mask = logical_operation(
            torch.logical_and(contact_data["geom_a"] >= self.geom_start, contact_data["geom_a"] < self.geom_end),
            torch.logical_and(contact_data["geom_b"] >= self.geom_start, contact_data["geom_b"] < self.geom_end),
        )
        if with_entity is not None and self.idx != with_entity.idx:
            valid_mask = torch.logical_and(
                valid_mask,
                torch.logical_or(
                    torch.logical_and(
                        contact_data["geom_a"] >= with_entity.geom_start, contact_data["geom_a"] < with_entity.geom_end
                    ),
                    torch.logical_and(
                        contact_data["geom_b"] >= with_entity.geom_start, contact_data["geom_b"] < with_entity.geom_end
                    ),
                ),
            )

        if self._solver.n_envs == 0:
            contact_data = {key: value[valid_mask] for key, value in contact_data.items()}
        else:
            contact_data["valid_mask"] = valid_mask

        contact_data["force_a"] = -contact_data["force"]
        contact_data["force_b"] = +contact_data["force"]
        del contact_data["force"]

        return contact_data

    def get_links_net_contact_force(self, envs_idx=None):
        """
        Returns net force applied on each links due to direct external contacts.

        Returns
        -------
        entity_links_force : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The net force applied on each links due to direct external contacts.
        """
        links_idx = slice(self.link_start, self.link_end)
        tensor = qd_to_torch(self._solver.dyn_state.links.contact_force, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self._solver.n_envs == 0 else tensor

    # ------------------------------------------------------------------------------------
    # ----------------------------------- friction ---------------------------------------
    # ------------------------------------------------------------------------------------

    def set_friction_ratio(self, friction_ratio, links_idx_local=None, envs_idx=None):
        """
        Set the friction ratio of the geoms of the specified links.

        Parameters
        ----------
        friction_ratio : torch.Tensor, shape (n_envs, n_links)
            The friction ratio
        links_idx_local : array_like
            The indices of the links to set friction ratio.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx_local = self._get_global_idx(links_idx_local, self.n_links, 0, unsafe=True)

        links_n_geoms = torch.tensor(
            [self._links[i_l].n_geoms for i_l in links_idx_local], dtype=gs.tc_int, device=gs.device
        )
        links_friction_ratio = torch.as_tensor(friction_ratio, dtype=gs.tc_float, device=gs.device)
        geoms_friction_ratio = torch.repeat_interleave(links_friction_ratio, links_n_geoms, dim=-1)
        geoms_idx = [
            i_g for i_l in links_idx_local for i_g in range(self._links[i_l].geom_start, self._links[i_l].geom_end)
        ]

        self._solver.set_geoms_friction_ratio(geoms_friction_ratio, geoms_idx, envs_idx)

    def set_friction(self, friction):
        """
        Set the friction coefficient of all the links (and in turn, geometries) of the rigid entity.

        Note
        ----
        The friction coefficient associated with a pair of geometries in contact is defined as the maximum between
        their respective values, so one must be careful the set the friction coefficient properly for both of them.

        Warning
        -------
        The friction coefficient must be in range [1e-2, 5.0] for simulation stability.

        Parameters
        ----------
        friction : float
            The friction coefficient to set.
        """

        if friction < 1e-2 or friction > 5.0:
            gs.raise_exception("`friction` must be in the range [1e-2, 5.0] for simulation stability.")

        for link in self._links:
            link.set_friction(friction)

    def set_friction_torsional(self, friction_torsional):
        """
        Set the torsional friction coefficient of all the links (and in turn, geometries) of the rigid entity.

        Note
        ----
        The torsional friction coefficient associated with a pair of geometries in contact is defined as the maximum
        between their respective values (see 'gs.materials.Rigid'). Only effective when torsional friction is enabled
        at the scene level (see 'RigidOptions.enable_torsional_friction').

        Parameters
        ----------
        friction_torsional : float
            The torsional friction coefficient to set.
        """
        if friction_torsional < 0:
            gs.raise_exception("`friction_torsional` must be non-negative.")

        for link in self._links:
            link.set_friction_torsional(friction_torsional)

    def set_friction_rolling(self, friction_rolling):
        """
        Set the rolling friction coefficient of all the links (and in turn, geometries) of the rigid entity.

        Note
        ----
        The rolling friction coefficient associated with a pair of geometries in contact is defined as the maximum
        between their respective values (see 'gs.materials.Rigid'). Only effective when rolling friction is enabled
        at the scene level (see 'RigidOptions.enable_rolling_friction').

        Parameters
        ----------
        friction_rolling : float
            The rolling friction coefficient to set.
        """
        if friction_rolling < 0:
            gs.raise_exception("`friction_rolling` must be non-negative.")

        for link in self._links:
            link.set_friction_rolling(friction_rolling)

    # ------------------------------------------------------------------------------------
    # --------------------------------- mass / inertia -----------------------------------
    # ------------------------------------------------------------------------------------

    def set_mass_shift(self, mass_shift, links_idx_local=None, envs_idx=None):
        """
        Set the mass shift of specified links.

        Parameters
        ----------
        mass : torch.Tensor, shape (n_envs, n_links)
            The mass shift
        links_idx_local : array_like
            The indices of the links to set mass shift.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_mass_shift(mass_shift, links_idx, envs_idx)

    def set_COM_shift(self, com_shift, links_idx_local=None, envs_idx=None):
        """
        Set the center of mass (COM) shift of specified links.

        Parameters
        ----------
        com : torch.Tensor, shape (n_envs, n_links, 3)
            The COM shift
        links_idx_local : array_like
            The indices of the links to set COM shift.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_COM_shift(com_shift, links_idx, envs_idx)

    @gs.assert_built
    def set_links_inertial_mass(self, inertial_mass, links_idx_local=None, envs_idx=None):
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_inertial_mass(inertial_mass, links_idx, envs_idx)

    @gs.assert_built
    def set_links_invweight(self, invweight, links_idx_local=None, envs_idx=None):
        raise DeprecationError(
            "This method has been removed because links invweights are supposed to be a by-product of link properties "
            "(mass, pose, and inertia matrix), joint placements, and dof armatures. Please consider using the "
            "considering setters instead."
        )

    @gs.assert_built
    def set_mass(self, mass):
        """
        Set the mass of the entity.

        Parameters
        ----------
        mass : float
            The mass to set.
        """
        ratio = float(mass) / self.get_mass()
        for link in self.links:
            link.set_mass(link.get_mass() * ratio)

    @gs.assert_built
    def get_mass(self):
        """
        Get the total mass of the entity in kg.

        For heterogeneous entities, returns an array of masses for each environment.
        For non-heterogeneous entities, returns a scalar mass.

        Returns
        -------
        mass : float | np.ndarray
            The total mass of the entity in kg. For heterogeneous entities, returns
            an array of shape (n_envs,) with per-environment masses.
        """
        if self._enable_heterogeneous:
            links_idx = slice(self.link_start, self.link_end)
            links_mass = qd_to_numpy(self._solver.dyn_info.links.inertial_mass, None, links_idx, transpose=True)
            return links_mass.sum(axis=1)

        # Original behavior: sum link masses to scalar
        mass = 0.0
        for link in self.links:
            mass += link.get_mass()
        return mass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def visualize_contact(self):
        """Whether to visualize contact force."""
        return self._visualize_contact

    @property
    def n_geoms(self):
        """The number of collision geom `RigidGeom` in the entity."""
        if self._is_built:
            return self._n_geoms
        return sum(link.n_geoms for link in self._links)

    @property
    def n_cells(self):
        """The number of sdf cells in the entity."""
        return sum(link.n_cells for link in self._links)

    @property
    def n_verts(self):
        """The number of vertices (from collision geom `RigidGeom`) in the entity."""
        return sum(link.n_verts for link in self._links)

    @property
    def n_faces(self):
        """The number of faces (from collision geom `RigidGeom`) in the entity."""
        return sum(link.n_faces for link in self._links)

    @property
    def n_edges(self):
        """The number of edges (from collision geom `RigidGeom`) in the entity."""
        return sum(link.n_edges for link in self._links)

    @property
    def geom_start(self):
        """The index of the entity's first RigidGeom in the scene."""
        return self._geom_start

    @property
    def geom_end(self):
        """The index of the entity's last RigidGeom in the scene *plus one*."""
        return self._geom_start + self.n_geoms

    @property
    def cell_start(self):
        """The start index the entity's sdf cells in the scene."""
        return self._cell_start

    @property
    def cell_end(self):
        """The end index the entity's sdf cells in the scene *plus one*."""
        return self._cell_start + self.n_cells

    @property
    def gravity_compensation(self):
        """Apply a force to compensate gravity. A value of 1 will make a zero-gravity behavior. Default to 0"""
        return self.material.gravity_compensation

    @property
    def vert_start(self):
        """The index of the entity's first `vert` (collision vertex) in the scene."""
        return self._vert_start

    @property
    def face_start(self):
        """The index of the entity's first `face` (collision face) in the scene."""
        return self._face_start

    @property
    def edge_start(self):
        """The index of the entity's first `edge` (collision edge) in the scene."""
        return self._edge_start

    @property
    def geoms(self) -> list[RigidGeom]:
        """The list of collision geoms (`RigidGeom`) in the entity."""
        if self.is_built:
            return self._geoms
        return gs.List(geom for link in self._links for geom in link.geoms)

    @property
    def n_equalities(self):
        """The number of equality constraints in the entity."""
        return len(self._equalities)

    @property
    def equality_start(self):
        """The index of the entity's first RigidEquality in the scene."""
        return self._equality_start

    @property
    def equality_end(self):
        """The index of the entity's last RigidEquality in the scene *plus one*."""
        return self._equality_start + self.n_equalities

    @property
    def equalities(self):
        """The list of equality constraints (`RigidEquality`) in the entity."""
        return self._equalities

    @property
    def is_free(self) -> bool:
        raise DeprecationError("This property has been removed.")

    @property
    def is_local_collision_mask(self):
        """Whether the contype and conaffinity bitmasks of this entity only applies to self-collision."""
        return self._is_local_collision_mask
