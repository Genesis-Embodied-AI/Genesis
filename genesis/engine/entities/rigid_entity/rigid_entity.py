import inspect
import os
import xml.etree.ElementTree as ET
from copy import copy
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Any
from functools import wraps

import quadrants as qd
import numpy as np
import torch
import trimesh

import genesis as gs
from genesis.engine.materials.base import Material
from genesis.options.morphs import Morph
from genesis.options.surfaces import Surface
from genesis.utils import array_class
from genesis.utils import linalg as lu
from genesis.utils import geom as gu
from genesis.utils import mesh as mu
from genesis.utils import mjcf as mju
from genesis.utils import terrain as tu
from genesis.utils import urdf as uu
from genesis.utils.urdf import compose_inertial_properties, rotate_inertia
from genesis.utils.misc import DeprecationError, broadcast_tensor, qd_to_numpy, qd_to_torch
from genesis.engine.states.entities import RigidEntityState

from ..base_entity import Entity
from .rigid_equality import RigidEquality
from .rigid_geom import RigidGeom
from .rigid_joint import RigidJoint
from .rigid_link import RigidLink

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


# Wrapper to track the arguments of a function and save them in the target buffer
def tracked(fun):
    sig = inspect.signature(fun)

    @wraps(fun)
    def wrapper(self, *args, **kwargs):
        if self._update_tgt_while_set:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            args_dict = dict(tuple(bound.arguments.items())[1:])
            self._update_tgt(fun.__name__, args_dict)
        return fun(self, *args, **kwargs)

    return wrapper


def compute_inertial_from_geom_infos(cg_infos, vg_infos, rho):
    """
    Compute inertial properties (mass, center of mass, inertia tensor) from geometry infos.

    This is a standalone helper function that computes combined inertial properties from
    a collection of collision and/or visual geometry infos.

    Parameters
    ----------
    cg_infos : list[dict]
        List of collision geometry info dicts, each containing 'mesh', 'pos', 'quat'.
    vg_infos : list[dict]
        List of visual geometry info dicts, each containing 'vmesh', 'pos', 'quat'.
        Used as fallback if cg_infos is empty.
    rho : float
        Material density (kg/m^3) used to compute mass from volume.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        (total_mass, center_of_mass, inertia_tensor)
    """
    total_mass = gs.EPS
    total_com = np.zeros(3, dtype=gs.np_float)
    total_inertia = np.zeros((3, 3), dtype=gs.np_float)

    # Use collision geoms if available, otherwise fall back to visual geoms
    for g_info in cg_infos if cg_infos else vg_infos:
        mesh = g_info["mesh" if cg_infos else "vmesh"]
        if g_info["type"] == gs.GEOM_TYPE.PLANE:
            continue
        geom_pos = g_info.get("pos", gu.zero_pos())
        geom_quat = g_info.get("quat", gu.identity_quat())

        inertia_mesh = mesh.trimesh
        if not inertia_mesh.is_watertight:
            inertia_mesh = trimesh.convex.convex_hull(inertia_mesh)

        if inertia_mesh.volume < -gs.EPS:
            inertia_mesh.invert()

        geom_mass = inertia_mesh.volume * rho
        geom_com_local = np.array(inertia_mesh.center_mass, dtype=gs.np_float)
        geom_inertia_local = inertia_mesh.moment_inertia / inertia_mesh.mass * geom_mass

        # Transform geom properties to link frame
        geom_com_link = gu.transform_by_quat(geom_com_local, geom_quat) + geom_pos
        geom_inertia_link = rotate_inertia(geom_inertia_local, gu.quat_to_R(geom_quat))

        # Compose with existing properties
        total_mass, total_com, total_inertia = compose_inertial_properties(
            total_mass, total_com, total_inertia, geom_mass, geom_com_link, geom_inertia_link
        )

    return total_mass, total_com, total_inertia


@qd.data_oriented
class RigidEntity(Entity):
    """
    Entity class in rigid body systems. One rigid entity can be a robot, a terrain, a floating rigid body, etc.
    """

    # override typing
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
        equality_start=0,
        visualize_contact: bool = False,
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
        self._geom_start = geom_start
        self._cell_start = cell_start
        self._vert_start = vert_start
        self._face_start = face_start
        self._edge_start = edge_start
        self._free_verts_state_start = free_verts_state_start
        self._fixed_verts_state_start = fixed_verts_state_start
        self._vgeom_start = vgeom_start
        self._vvert_start = vvert_start
        self._vface_start = vface_start
        self._equality_start = equality_start

        self._free_verts_idx_local = torch.tensor([], dtype=gs.tc_int, device=gs.device)
        self._fixed_verts_idx_local = torch.tensor([], dtype=gs.tc_int, device=gs.device)

        self._batch_fixed_verts = morph.batch_fixed_verts

        self._visualize_contact: bool = visualize_contact

        self._is_built: bool = False
        self._is_attached: bool = False

        self._load_model()

        # Initialize target variables and checkpoint
        self._tgt_keys = ("pos", "quat", "qpos", "dofs_velocity")
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
        # Store g_infos for heterogeneous inertial computation
        self._first_g_infos = None
        if isinstance(morph, gs.morphs.Mesh):
            self._first_g_infos = self._load_mesh(morph, self._surface)
        elif isinstance(morph, (gs.morphs.MJCF, gs.morphs.URDF, gs.morphs.Drone, gs.morphs.USD)):
            self._load_scene(morph, self._surface)
        elif isinstance(morph, gs.morphs.Primitive):
            self._first_g_infos = self._load_primitive(morph, self._surface)
        elif isinstance(morph, gs.morphs.Terrain):
            self._load_terrain(morph, self._surface)
        else:
            gs.raise_exception(f"Unsupported morph: {morph}.")

        # Load heterogeneous variants (if any)
        self._load_heterogeneous_morphs()

    def _load_heterogeneous_morphs(self):
        """
        Load heterogeneous morphs (additional geometry variants for parallel environments).
        Each variant is loaded as additional geoms/vgeoms attached to the single link.
        """
        if not self._enable_heterogeneous:
            return

        # Initialize tracking lists for geom/vgeom ranges per variant.
        # These store the start/end indices for each variant's geoms and vgeoms,
        # enabling per-environment dispatch during simulation.
        self.variants_link_start = gs.List()
        self.variants_link_end = gs.List()
        self.variants_n_links = gs.List()
        self.variants_geom_start = gs.List()
        self.variants_geom_end = gs.List()
        self.variants_vgeom_start = gs.List()
        self.variants_vgeom_end = gs.List()
        self.variants_inertial_mass = gs.List()
        self.variants_inertial_pos = gs.List()
        self.variants_inertial_i = gs.List()

        # Record the first variant (the main morph)
        self.variants_link_start.append(self._link_start)
        self.variants_n_links.append(len(self._links))
        self.variants_link_end.append(self._link_start + len(self._links))
        self.variants_geom_start.append(self._geom_start)
        first_variant_geom_end = self._geom_start + len(self.geoms)
        self.variants_geom_end.append(first_variant_geom_end)
        self.variants_vgeom_start.append(self._vgeom_start)
        self.variants_vgeom_end.append(self._vgeom_start + len(self.vgeoms))
        # Store number of geoms in first variant for balanced block distribution across environments
        self._first_variant_n_geoms = len(self.geoms)
        self._first_variant_n_vgeoms = len(self.vgeoms)

        # Heterogeneous simulation only supports single-link entities.
        if len(self._links) != 1:
            gs.raise_exception("morph_heterogeneous only supports single-link entities.")

        link = self._links[0]

        # Compute first variant's inertial properties using stored g_infos
        cg_infos, vg_infos = self._convert_g_infos_to_cg_infos_and_vg_infos(self._morph, self._first_g_infos, False)
        het_mass, het_pos, het_i = compute_inertial_from_geom_infos(cg_infos, vg_infos, self.material.rho)
        self.variants_inertial_mass.append(het_mass)
        self.variants_inertial_pos.append(het_pos)
        self.variants_inertial_i.append(het_i)

        # Load additional heterogeneous variants
        for morph in self._morph_heterogeneous:
            if isinstance(morph, gs.morphs.Mesh):
                g_infos = self._load_mesh(morph, self._surface, load_geom_only_for_heterogeneous=True)
            elif isinstance(morph, gs.morphs.Primitive):
                g_infos = self._load_primitive(morph, self._surface, load_geom_only_for_heterogeneous=True)
            else:
                gs.raise_exception(
                    f"morph_heterogeneous only supports Primitive and Mesh, got: {type(morph).__name__}."
                )

            cg_infos, vg_infos = self._convert_g_infos_to_cg_infos_and_vg_infos(morph, g_infos, False)

            # Compute inertial properties for this variant from collision or visual geometries
            het_mass, het_pos, het_i = compute_inertial_from_geom_infos(cg_infos, vg_infos, self.material.rho)
            self.variants_inertial_mass.append(het_mass)
            self.variants_inertial_pos.append(het_pos)
            self.variants_inertial_i.append(het_i)

            # Add visual geometries
            for g_info in vg_infos:
                link._add_vgeom(
                    vmesh=g_info["vmesh"],
                    init_pos=g_info.get("pos", gu.zero_pos()),
                    init_quat=g_info.get("quat", gu.identity_quat()),
                )

            # Add collision geometries
            for g_info in cg_infos:
                friction = self.material.friction
                if friction is None:
                    friction = g_info.get("friction", gu.default_friction())
                link._add_geom(
                    mesh=g_info["mesh"],
                    init_pos=g_info.get("pos", gu.zero_pos()),
                    init_quat=g_info.get("quat", gu.identity_quat()),
                    type=g_info["type"],
                    friction=friction,
                    sol_params=g_info["sol_params"],
                    data=g_info.get("data"),
                    needs_coup=self.material.needs_coup,
                    contype=g_info["contype"],
                    conaffinity=g_info["conaffinity"],
                )

            # Record ranges for this variant
            self.variants_link_start.append(self.variants_link_end[-1])
            self.variants_link_end.append(self._link_start + len(self._links))
            self.variants_n_links.append(self.variants_link_end[-1] - self.variants_link_start[-1])
            self.variants_geom_start.append(self.variants_geom_end[-1])
            self.variants_geom_end.append(self.variants_geom_end[-1] + len(cg_infos))
            self.variants_vgeom_start.append(self.variants_vgeom_end[-1])
            self.variants_vgeom_end.append(self.variants_vgeom_end[-1] + len(vg_infos))

    def _load_model(self):
        self._links = gs.List()
        self._joints = gs.List()
        self._equalities = gs.List()

        self._load_morph(self._morph)

        self._requires_jac_and_IK = self._morph.requires_jac_and_IK
        self._is_local_collision_mask = isinstance(self._morph, gs.morphs.MJCF)

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
            geom_data = None
            geom_type = gs.GEOM_TYPE.MESH
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
                dict(
                    contype=0,
                    conaffinity=0,
                    vmesh=gs.Mesh.from_trimesh(tmesh, surface=surface, metadata=metadata),
                )
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

        # Load meshes
        meshes = gs.Mesh.from_morph_surface(morph, surface)

        g_infos = []
        if morph.visualization:
            for mesh in meshes:
                g_infos.append(
                    dict(
                        contype=0,
                        conaffinity=0,
                        vmesh=mesh,
                    )
                )
        if morph.collision:
            # Merge them as a single one if requested
            if morph.merge_submeshes_for_collision and len(meshes) > 1:
                tmesh = trimesh.util.concatenate([mesh.trimesh for mesh in meshes])
                mesh = gs.Mesh.from_trimesh(mesh=tmesh, surface=gs.surfaces.Collision())
                meshes = (mesh,)

            for mesh in meshes:
                g_infos.append(
                    dict(
                        contype=morph.contype,
                        conaffinity=morph.conaffinity,
                        mesh=mesh,
                        type=gs.GEOM_TYPE.MESH,
                        sol_params=gu.default_solver_params(),
                    )
                )

        # For heterogeneous simulation, only return geometry info without creating link/joint
        if load_geom_only_for_heterogeneous:
            return g_infos

        link_name = os.path.basename(morph.file).replace(".", "_")

        self._add_by_info(
            l_info=dict(
                is_robot=False,
                name=f"{link_name}_baselink",
                pos=np.array(morph.pos),
                quat=np.array(morph.quat),
                inertial_pos=None,  # we will compute the COM later based on the geometry
                inertial_quat=gu.identity_quat(),
                parent_idx=-1,
            ),
            j_infos=[
                dict(
                    name=f"{link_name}_baselink_joint",
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

    def _load_terrain(self, morph, surface):
        vmesh, mesh, self.terrain_hf = tu.parse_terrain(morph, surface)
        self.terrain_scale = np.array((morph.horizontal_scale, morph.vertical_scale), dtype=gs.np_float)

        g_infos = []
        if morph.visualization:
            g_infos.append(
                dict(
                    contype=0,
                    conaffinity=0,
                    vmesh=vmesh,
                )
            )
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
            j_infos=[
                dict(
                    name="joint_baselink",
                    n_qs=0,
                    n_dofs=0,
                    type=gs.JOINT_TYPE.FIXED,
                )
            ],
            g_infos=g_infos,
            morph=morph,
            surface=surface,
        )

    def _load_scene(self, morph, surface):
        from genesis.engine.couplers import IPCCoupler

        # Mujoco's unified MJCF+URDF parser is not good enough for now to be used for loading both MJCF and URDF files.
        # First, it would happen when loading visual meshes having supported format (i.e. Collada files '.dae').
        # Second, it does not take into account URDF 'mimic' joint constraints. However, it does a better job at
        # initialized undetermined physics parameters.
        if isinstance(morph, gs.morphs.MJCF):
            # Mujoco's unified MJCF+URDF parser systematically for MJCF files
            l_infos, links_j_infos, links_g_infos, eqs_info = mju.parse_xml(morph, surface)
        elif isinstance(morph, (gs.morphs.URDF, gs.morphs.Drone)):
            # Custom "legacy" URDF parser for loading geometries (visual and collision) and equality constraints.
            # This is necessary because Mujoco cannot parse visual geometries (meshes) reliably for URDF.
            l_infos, links_j_infos, links_g_infos, eqs_info = uu.parse_urdf(morph, surface)

            # Mujoco's unified MJCF+URDF parser for only link, joints, and collision geometries properties
            morph_ = copy(morph)
            morph_.visualization = False
            try:
                # Mujoco's unified MJCF+URDF parser for URDF files.
                # Note that Mujoco URDF parser completely ignores equality constraints.
                l_infos, links_j_infos_mj, links_g_infos_mj, _ = mju.parse_xml(morph_, surface)

                # Mujoco is not parsing actuators properties
                for j_info_gs in chain.from_iterable(links_j_infos):
                    for j_info_mj in chain.from_iterable(links_j_infos_mj):
                        if j_info_mj["name"] == j_info_gs["name"]:
                            for name in ("dofs_force_range", "dofs_armature", "dofs_kp", "dofs_kv"):
                                j_info_mj[name] = j_info_gs[name]
                links_j_infos = links_j_infos_mj

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
                mass_tot = sum(l_info["inertial_mass"] for l_info in l_infos)
                j_info["dofs_damping"][3:] = mass_tot * morph.default_base_ang_damping_scale
            j_info["dofs_armature"] = np.zeros(6)
            j_info["dofs_kp"] = np.zeros((6,), dtype=gs.np_float)
            j_info["dofs_kv"] = np.zeros((6,), dtype=gs.np_float)
            j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (6, 1))
            links_j_infos[0] = [j_info]

            # Shift root idx for all child links and replace root if no longer fixed wrt world
            for i_l in range(len(l_infos)):
                l_info = l_infos[i_l]
                if "root_idx" in l_info and l_info["root_idx"] in (1, i_l):
                    l_info["root_idx"] = 0

            # Must invalidate invweight for all child links and joints because the root joint was fixed when it was
            # initially computed. Re-initialize it to some strictly negative value to trigger recomputation in solver.
            for i_l in range(len(l_infos)):
                l_infos[i_l]["invweight"] = np.full((2,), fill_value=-1.0)
                for j_info in links_j_infos[i_l]:
                    j_info["dofs_invweight"] = np.full((j_info["n_dofs"],), fill_value=-1.0)

        # Force recomputing inertial information based on geometry if ill-defined for some reason
        is_inertia_invalid = False
        for l_info, link_j_infos in zip(l_infos, links_j_infos):
            if not all(j_info["type"] == gs.JOINT_TYPE.FIXED for j_info in link_j_infos) and (
                (l_info.get("inertial_mass") is None or l_info["inertial_mass"] <= 0.0)
                or (l_info.get("inertial_i") is None or (np.diag(l_info["inertial_i"]) <= 0.0).any())
            ):
                if l_info.get("inertial_mass") is not None or l_info.get("inertial_i") is not None:
                    gs.logger.debug(
                        f"Invalid or undefined inertia for link '{l_info['name']}'. Force recomputing it based on "
                        "geometry."
                    )
                l_info["inertial_i"] = None
                l_info["inertial_mass"] = None
                is_inertia_invalid = True
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

        # Make sure that the entity is not object
        if (
            isinstance(self.sim.coupler, IPCCoupler)
            and self.material.coupling_mode == "ipc_only"
            and any(l_info["is_robot"] for l_info in l_infos)
        ):
            gs.raise_exception(
                "`RigidMaterial.coupling_mode='ipc_only'` only supported by rigid non-articulated objects."
            )

        # Add (link, joints, geoms) tuples sequentially
        for l_info, link_j_infos, link_g_infos in zip(l_infos, links_j_infos, links_g_infos):
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
            link_j_infos = [j_info for j_info in link_j_infos if j_info["n_dofs"] > 0]

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
        for link in self._links:
            link._build()

        self._n_qs = self.n_qs
        self._n_dofs = self.n_dofs
        self._n_geoms = self.n_geoms
        self._geoms = self.geoms
        self._vgeoms = self.vgeoms
        self._is_built = True

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
        if not self._requires_jac_and_IK:
            return

        if self.n_dofs == 0:
            return

        self._jacobian = qd.field(dtype=gs.qd_float, shape=(6, self.n_dofs, self._solver._B))

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

        # for storing intermediate results
        self._IK_n_tgts = self._solver._options.IK_max_targets
        self._IK_error_dim = self._IK_n_tgts * 6
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

        # Add parent joints
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
                dofs_kp=j_info.get("dofs_kp", np.zeros(n_dofs)),
                dofs_kv=j_info.get("dofs_kv", np.zeros(n_dofs)),
                dofs_force_range=j_info.get("dofs_force_range", np.tile([[-np.inf, np.inf]], [n_dofs, 1])),
            )
            joints.append(joint)

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
        )
        self._links.append(link)

        if not link.is_fixed and isinstance(morph, gs.options.morphs.FileMorph) and morph.recompute_inertia:
            link._inertial_pos = None
            link._inertial_quat = None
            link._inertial_i = None
            link._inertial_mass = None

        # Separate collision from visual geometry for post-processing
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
            if l_info["is_robot"]:
                decompose_error_threshold = morph.decompose_robot_error_threshold
            else:
                decompose_error_threshold = morph.decompose_object_error_threshold

            cg_infos = mu.postprocess_collision_geoms(
                cg_infos,
                morph.decimate,
                morph.decimate_face_num,
                morph.decimate_aggressiveness,
                morph.convexify,
                decompose_error_threshold,
                morph.coacd_options,
            )

        # Randomize collision mesh colors. The is especially useful to check convex decomposition.
        for g_info in cg_infos:
            mesh = g_info["mesh"]
            mesh.set_color((*np.random.rand(3), 0.7))

        # Add visual geometries
        for g_info in vg_infos:
            link._add_vgeom(
                vmesh=g_info["vmesh"],
                init_pos=g_info.get("pos", gu.zero_pos()),
                init_quat=g_info.get("quat", gu.identity_quat()),
            )

        # Add collision geometries
        for g_info in cg_infos:
            friction = self.material.friction
            if friction is None:
                friction = g_info.get("friction", gu.default_friction())
            link._add_geom(
                mesh=g_info["mesh"],
                init_pos=g_info.get("pos", gu.zero_pos()),
                init_quat=g_info.get("quat", gu.identity_quat()),
                type=g_info["type"],
                friction=friction,
                sol_params=g_info["sol_params"],
                data=g_info.get("data"),
                needs_coup=self.material.needs_coup,
                contype=g_info["contype"],
                conaffinity=g_info["conaffinity"],
            )

        return link, joints

    @staticmethod
    def _convert_g_infos_to_cg_infos_and_vg_infos(morph, g_infos, is_robot):
        """
        Separate collision from visual geometry and post-process collision meshes.
        Used for both normal loading and heterogeneous simulation.
        """
        cg_infos, vg_infos = [], []
        for g_info in g_infos:
            is_col = g_info["contype"] or g_info["conaffinity"]
            if morph.collision and is_col:
                cg_infos.append(g_info)
            if morph.visualization and not is_col:
                vg_infos.append(g_info)

        # Post-process all collision meshes at once
        if isinstance(morph, gs.options.morphs.FileMorph):
            if is_robot:
                decompose_error_threshold = morph.decompose_robot_error_threshold
            else:
                decompose_error_threshold = morph.decompose_object_error_threshold

            cg_infos = mu.postprocess_collision_geoms(
                cg_infos,
                morph.decimate,
                morph.decimate_face_num,
                morph.decimate_aggressiveness,
                morph.convexify,
                decompose_error_threshold,
                morph.coacd_options,
            )

        # Randomize collision mesh colors
        for g_info in cg_infos:
            mesh = g_info["mesh"]
            mesh.set_color((*np.random.rand(3), 0.7))

        return cg_infos, vg_infos

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

    @gs.assert_unbuilt
    def attach(self, parent_entity, parent_link_name: str | None = None):
        """
        Merge two entities to act as single one, by attaching the base link of this entity as a child of a given link of
        another entity.

        Parameters
        ----------
        parent_entity : genesis.Entity
            The entity in the scene that will be a parent of kinematic tree.
        parent_link_name : str
            The name of the link in the parent entity to be linked. Default to the latest link the parent kinematic
            tree.
        """
        if self._is_attached:
            gs.raise_exception("Entity already attached.")

        if not isinstance(parent_entity, RigidEntity):
            gs.raise_exception("Parent entity must derive from 'RigidEntity'.")

        if parent_entity is self:
            gs.raise_exception("Cannot attach entity to itself.")

        if parent_entity.idx > self.idx:
            gs.raise_exception("Parent entity must be instantiated before child entity.")

        # Check if base link was fixed but no longer is
        base_link = self.links[0]
        parent_link = parent_entity.get_link(parent_link_name)
        if base_link.is_fixed and not parent_link.is_fixed:
            if not self._batch_fixed_verts:
                gs.raise_exception(
                    "Attaching fixed-based entity to parent link requires setting Morph option 'batch_fixed_verts=True'."
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
                joint._dof_start -= n_base_dofs
                joint._q_start -= n_base_qs
            for link in self._solver.links[(self.link_start + 1) :]:
                link._joint_start -= n_base_joints

        # Overwrite parent link
        base_link._parent_idx = parent_link.idx

        for link in self.links:
            # Break as soon as the root idx is -1, because the following links correspond to a different kinematic tree
            if link.root_idx == -1:
                break

            # Override root idx for child links
            assert link.root_idx == base_link.idx
            link._root_idx = parent_link.root_idx

            # Update fixed link flag
            link._is_fixed &= parent_link.is_fixed

            # Must invalidate invweight for all child links and joints
            link._invweight = None

        self._is_attached = True

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

        if local_point is None:
            sol = self._solver
            self._kernel_get_jacobian_zero(
                tgt_link_idx=link.idx,
                dofs_info=sol.dofs_info,
                joints_info=sol.joints_info,
                links_info=sol.links_info,
                links_state=sol.links_state,
            )
        else:
            p_local = torch.as_tensor(local_point, dtype=gs.tc_float, device=gs.device)
            if p_local.shape != (3,):
                gs.raise_exception("Must be a vector of length 3")
            sol = self._solver
            self._kernel_get_jacobian(
                tgt_link_idx=link.idx,
                p_local=p_local,
                dofs_info=sol.dofs_info,
                joints_info=sol.joints_info,
                links_info=sol.links_info,
                links_state=sol.links_state,
            )

        jacobian = qd_to_torch(self._jacobian, transpose=True, copy=True)
        if self._solver.n_envs == 0:
            jacobian = jacobian[0]

        return jacobian

    @qd.func
    def _impl_get_jacobian(
        self,
        tgt_link_idx,
        i_b,
        p_vec,
        dofs_info: array_class.DofsInfo,
        joints_info: array_class.JointsInfo,
        links_info: array_class.LinksInfo,
        links_state: array_class.LinksState,
    ):
        self._func_get_jacobian(
            tgt_link_idx=tgt_link_idx,
            i_b=i_b,
            p_local=p_vec,
            pos_mask=qd.Vector.one(gs.qd_int, 3),
            rot_mask=qd.Vector.one(gs.qd_int, 3),
            dofs_info=dofs_info,
            joints_info=joints_info,
            links_info=links_info,
            links_state=links_state,
        )

    @qd.kernel
    def _kernel_get_jacobian(
        self,
        tgt_link_idx: qd.i32,
        p_local: qd.types.ndarray(),
        dofs_info: array_class.DofsInfo,
        joints_info: array_class.JointsInfo,
        links_info: array_class.LinksInfo,
        links_state: array_class.LinksState,
    ):
        p_vec = qd.Vector([p_local[0], p_local[1], p_local[2]], dt=gs.qd_float)
        for i_b in range(self._solver._B):
            self._impl_get_jacobian(
                tgt_link_idx=tgt_link_idx,
                i_b=i_b,
                p_vec=p_vec,
                dofs_info=dofs_info,
                joints_info=joints_info,
                links_info=links_info,
                links_state=links_state,
            )

    @qd.kernel
    def _kernel_get_jacobian_zero(
        self,
        tgt_link_idx: qd.i32,
        dofs_info: array_class.DofsInfo,
        joints_info: array_class.JointsInfo,
        links_info: array_class.LinksInfo,
        links_state: array_class.LinksState,
    ):
        for i_b in range(self._solver._B):
            self._impl_get_jacobian(
                tgt_link_idx=tgt_link_idx,
                i_b=i_b,
                p_vec=qd.Vector.zero(gs.qd_float, 3),
                dofs_info=dofs_info,
                joints_info=joints_info,
                links_info=links_info,
                links_state=links_state,
            )

    @qd.func
    def _func_get_jacobian(
        self,
        tgt_link_idx,
        i_b,
        p_local,
        pos_mask,
        rot_mask,
        dofs_info: array_class.DofsInfo,
        joints_info: array_class.JointsInfo,
        links_info: array_class.LinksInfo,
        links_state: array_class.LinksState,
    ):
        for i_row, i_d in qd.ndrange(6, self.n_dofs):
            self._jacobian[i_row, i_d, i_b] = 0.0

        tgt_link_pos = links_state.pos[tgt_link_idx, i_b] + gu.qd_transform_by_quat(
            p_local, links_state.quat[tgt_link_idx, i_b]
        )
        i_l = tgt_link_idx
        while i_l > -1:
            I_l = [i_l, i_b] if qd.static(self.solver._options.batch_links_info) else i_l

            dof_offset = 0
            for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                I_j = [i_j, i_b] if qd.static(self.solver._options.batch_joints_info) else i_j

                if joints_info.type[I_j] == gs.JOINT_TYPE.FIXED:
                    pass

                elif joints_info.type[I_j] == gs.JOINT_TYPE.REVOLUTE:
                    i_d = joints_info.dof_start[I_j]
                    I_d = [i_d, i_b] if qd.static(self.solver._options.batch_dofs_info) else i_d
                    i_d_jac = i_d + dof_offset - self._dof_start
                    rotation = gu.qd_transform_by_quat(dofs_info.motion_ang[I_d], links_state.quat[i_l, i_b])
                    translation = rotation.cross(tgt_link_pos - links_state.pos[i_l, i_b])

                    self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                    self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                    self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]
                    self._jacobian[3, i_d_jac, i_b] = rotation[0] * rot_mask[0]
                    self._jacobian[4, i_d_jac, i_b] = rotation[1] * rot_mask[1]
                    self._jacobian[5, i_d_jac, i_b] = rotation[2] * rot_mask[2]

                elif joints_info.type[I_j] == gs.JOINT_TYPE.PRISMATIC:
                    i_d = joints_info.dof_start[I_j]
                    I_d = [i_d, i_b] if qd.static(self.solver._options.batch_dofs_info) else i_d
                    i_d_jac = i_d + dof_offset - self._dof_start
                    translation = gu.qd_transform_by_quat(dofs_info.motion_vel[I_d], links_state.quat[i_l, i_b])

                    self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                    self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                    self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]

                elif joints_info.type[I_j] == gs.JOINT_TYPE.FREE:
                    # translation
                    for i_d_ in qd.static(range(3)):
                        i_d = joints_info.dof_start[I_j] + i_d_
                        i_d_jac = i_d + dof_offset - self._dof_start

                        self._jacobian[i_d_, i_d_jac, i_b] = 1.0 * pos_mask[i_d_]

                    # rotation
                    for i_d_ in qd.static(range(3)):
                        i_d = joints_info.dof_start[I_j] + i_d_ + 3
                        i_d_jac = i_d + dof_offset - self._dof_start
                        I_d = [i_d, i_b] if qd.static(self.solver._options.batch_dofs_info) else i_d
                        rotation = dofs_info.motion_ang[I_d]
                        translation = rotation.cross(tgt_link_pos - links_state.pos[i_l, i_b])

                        self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                        self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                        self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]
                        self._jacobian[3, i_d_jac, i_b] = rotation[0] * rot_mask[0]
                        self._jacobian[4, i_d_jac, i_b] = rotation[1] * rot_mask[1]
                        self._jacobian[5, i_d_jac, i_b] = rotation[2] * rot_mask[2]

                dof_offset = dof_offset + joints_info.n_dofs[I_j]

            i_l = links_info.parent_idx[I_l]

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
        Compute inverse kinematics for  multiple target links.

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
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        if not self._requires_jac_and_IK:
            gs.raise_exception(
                "Inverse kinematics and jacobian are disabled for this entity. Set `morph.requires_jac_and_IK` to True if you need them."
            )

        if self.n_dofs == 0:
            gs.raise_exception("Entity has zero dofs.")

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
        links_idx_by_dofs = []
        for link in self.links:
            for joint in link.joints:
                if any(i in dofs_idx for i in joint.dofs_idx_local):
                    links_idx_by_dofs.append(link.idx_local)
                    break
        links_idx_by_dofs = self._get_global_idx(links_idx_by_dofs, self.n_links, self._link_start)
        n_links_by_dofs = len(links_idx_by_dofs)

        from genesis.engine.solvers.rigid.abd.inverse_kinematics import kernel_rigid_entity_inverse_kinematics

        kernel_rigid_entity_inverse_kinematics(
            self,
            links_idx,
            poss,
            quats,
            local_points,
            n_links,
            dofs_idx,
            n_dofs,
            links_idx_by_dofs,
            n_links_by_dofs,
            custom_init_qpos,
            init_qpos,
            max_samples,
            max_solver_iters,
            damping,
            pos_tol,
            rot_tol,
            pos_mask,
            rot_mask,
            link_pos_mask,
            link_rot_mask,
            max_step_size,
            respect_joint_limit,
            envs_idx,
            self._solver.links_state,
            self._solver.links_info,
            self._solver.joints_state,
            self._solver.joints_info,
            self._solver.dofs_state,
            self._solver.dofs_info,
            self._solver.entities_info,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
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
            links_pos,
            links_quat,
            qpos,
            self._get_global_idx(qs_idx_local, self.n_qs, self._q_start),
            links_idx,
            envs_idx,
            self._solver.links_state,
            self._solver.links_info,
            self._solver.joints_state,
            self._solver.joints_info,
            self._solver.dofs_state,
            self._solver.dofs_info,
            self._solver.entities_info,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

        if self._solver.n_envs == 0:
            links_pos = links_pos[0]
            links_quat = links_quat[0]
        return links_pos, links_quat

    @qd.kernel
    def _kernel_forward_kinematics(
        self,
        links_pos: qd.types.ndarray(),
        links_quat: qd.types.ndarray(),
        qpos: qd.types.ndarray(),
        qs_idx: qd.types.ndarray(),
        links_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        joints_state: array_class.JointsState,
        joints_info: array_class.JointsInfo,
        dofs_state: array_class.DofsState,
        dofs_info: array_class.DofsInfo,
        entities_info: array_class.EntitiesInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        static_rigid_sim_config: qd.template(),
    ):
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_q_, i_b_ in qd.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
            # save original qpos
            # NOTE: reusing the IK_qpos_orig as cache (should not be a problem)
            self._IK_qpos_orig[qs_idx[i_q_], envs_idx[i_b_]] = rigid_global_info.qpos[qs_idx[i_q_], envs_idx[i_b_]]
            # set new qpos
            rigid_global_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos[i_b_, i_q_]

        # run FK
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                self._idx_in_solver,
                envs_idx[i_b_],
                links_state,
                links_info,
                joints_state,
                joints_info,
                dofs_state,
                dofs_info,
                entities_info,
                rigid_global_info,
                static_rigid_sim_config,
                is_backward=False,
            )

        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in qd.static(range(3)):
                links_pos[i_b_, i_l_, i] = links_state.pos[links_idx[i_l_], envs_idx[i_b_]][i]
            for i in qd.static(range(4)):
                links_quat[i_b_, i_l_, i] = links_state.quat[links_idx[i_l_], envs_idx[i_b_]][i]

        # restore original qpos
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_q_, i_b_ in qd.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
            rigid_global_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = self._IK_qpos_orig[qs_idx[i_q_], envs_idx[i_b_]]

        # run FK
        qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                self._idx_in_solver,
                envs_idx[i_b_],
                links_state,
                links_info,
                joints_state,
                joints_info,
                dofs_state,
                dofs_info,
                entities_info,
                rigid_global_info,
                static_rigid_sim_config,
                is_backward=False,
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

            match key:
                case "set_pos":
                    self.set_pos(**data_kwargs)
                case "set_quat":
                    self.set_quat(**data_kwargs)
                case "set_dofs_velocity":
                    self.set_dofs_velocity(**data_kwargs)
                case _:
                    gs.raise_exception(f"Invalid target key: {key} not in {self._tgt_keys}")

        self._tgt = dict()
        self._update_tgt_while_set = update_tgt_while_set

    def process_input_grad(self):
        index = self._sim.cur_step_local - self._sim._steps_local
        for key in reversed(self._tgt_buffer[index].keys()):
            data_kwargs = self._tgt_buffer[index][key]

            match key:
                # We need to unpack the data_kwargs because [_backward_from_qd] only supports positional arguments
                case "set_pos":
                    pos = data_kwargs.pop("pos")
                    if pos.requires_grad:
                        pos._backward_from_qd(self.set_pos_grad, data_kwargs["envs_idx"], data_kwargs["relative"])

                case "set_quat":
                    quat = data_kwargs.pop("quat")
                    if quat.requires_grad:
                        quat._backward_from_qd(self.set_quat_grad, data_kwargs["envs_idx"], data_kwargs["relative"])

                case "set_dofs_velocity":
                    velocity = data_kwargs.pop("velocity")
                    # [velocity] could be None when we want to zero the velocity (see set_dofs_velocity of RigidSolver)
                    if velocity is not None and velocity.requires_grad:
                        velocity._backward_from_qd(
                            self.set_dofs_velocity_grad,
                            data_kwargs["dofs_idx_local"],
                            data_kwargs["envs_idx"],
                        )
                case _:
                    gs.raise_exception(f"Invalid target key: {key} not in {self._tgt_keys}")

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
    def get_pos(self, envs_idx=None):
        """
        Returns position of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        pos : torch.Tensor, shape (3,) or (n_envs, 3)
            The position of the entity's base link.
        """
        return self._solver.get_links_pos(self.base_link_idx, envs_idx)[..., 0, :]

    @gs.assert_built
    def get_quat(self, envs_idx=None):
        """
        Returns quaternion of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        quat : torch.Tensor, shape (4,) or (n_envs, 4)
            The quaternion of the entity's base link.
        """
        return self._solver.get_links_quat(self.base_link_idx, envs_idx)[..., 0, :]

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
    def get_links_pos(
        self,
        links_idx_local=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com", "root_com"] = "link_origin",
        unsafe=False,
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

        Returns
        -------
        pos : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The position of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_pos(links_idx, envs_idx, ref=ref)

    @gs.assert_built
    def get_links_quat(self, links_idx_local=None, envs_idx=None):
        """
        Returns quaternion of all the entity's links.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        quat : torch.Tensor, shape (n_links, 4) or (n_envs, n_links, 4)
            The quaternion of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_quat(links_idx, envs_idx)

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

    def get_aabb(self):
        raise DeprecationError("This method has been removed. Please use 'get_AABB()' instead.")

    @gs.assert_built
    def get_links_vel(
        self,
        links_idx_local=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com"] = "link_origin",
        unsafe=False,
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
    def get_links_acc(self, links_idx_local=None, envs_idx=None):
        """
        Returns true linear acceleration (aka. "classical acceleration") of the specified entity's links expressed at
        their respective origin in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        acc : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear classical acceleration of the specified entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_acc(links_idx, envs_idx)

    @gs.assert_built
    def get_links_acc_ang(self, links_idx_local=None, envs_idx=None):
        """
        Returns angular acceleration of the specified entity's links expressed at their respective origin in world
        coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        acc : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear classical acceleration of the specified entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_acc_ang(links_idx, envs_idx)

    @gs.assert_built
    def get_links_inertial_mass(self, links_idx_local=None, envs_idx=None):
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_inertial_mass(links_idx, envs_idx)

    @gs.assert_built
    def get_links_invweight(self, links_idx_local=None, envs_idx=None):
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_invweight(links_idx, envs_idx)

    @gs.assert_built
    @tracked
    def set_pos(self, pos, envs_idx=None, *, zero_velocity=True, relative=False):
        """
        Set position of the entity's base link.

        Parameters
        ----------
        pos : array_like
            The position to set.
        relative : bool, optional
            Whether the position to set is absolute or relative to the initial (not current!) position. Defaults to
            False.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        # Throw exception in entity no longer has a "true" base link becaused it has attached
        if self._is_attached:
            gs.raise_exception("Impossible to set position of an entity that has been attached.")
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_base_links_pos(pos, self.base_link_idx, envs_idx, relative=relative)

    @gs.assert_built
    def set_pos_grad(self, envs_idx, relative, pos_grad):
        self._solver.set_base_links_pos_grad(self.base_link_idx, envs_idx, relative, pos_grad.data)

    @gs.assert_built
    @tracked
    def set_quat(self, quat, envs_idx=None, *, zero_velocity=True, relative=False):
        """
        Set quaternion of the entity's base link.

        Parameters
        ----------
        quat : array_like
            The quaternion to set.
        relative : bool, optional
            Whether the quaternion to set is absolute or relative to the initial (not current!) quaternion. Defaults to
            False.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if self._is_attached:
            gs.raise_exception("Impossible to set position of an entity that has been attached.")
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_base_links_quat(quat, self.base_link_idx, envs_idx, relative=relative)

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
            fixed_verts_state = qd_to_torch(self._solver.fixed_verts_state.pos, verts_idx)
            tensor[:, self._fixed_verts_idx_local] = fixed_verts_state
        if n_free_vertices > 0:
            verts_idx = slice(self._free_verts_state_start, self._free_verts_state_start + n_free_vertices)
            free_verts_state = qd_to_torch(self._solver.free_verts_state.pos, None, verts_idx, transpose=True)
            tensor[:, self._free_verts_idx_local] = free_verts_state

        if self._solver.n_envs == 0:
            tensor = tensor[0]
        return tensor

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

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coupling_mode == "external_articulation":
            gs.raise_exception("This method is not supported by `RigidMaterial.coupling_mode='external_articulation'`.")

        qs_idx = self._get_global_idx(qs_idx_local, self.n_qs, self._q_start, unsafe=True)
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_qpos(qpos, qs_idx, envs_idx, skip_forward=skip_forward)

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
    def set_dofs_velocity_grad(self, dofs_idx_local, envs_idx, velocity_grad):
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_velocity_grad(dofs_idx, envs_idx, velocity_grad.data)

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

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coupling_mode == "external_articulation":
            gs.raise_exception("This method is not supported by `RigidMaterial.coupling_mode='external_articulation'`.")

        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_dofs_position(position, dofs_idx, envs_idx)

    @gs.assert_built
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
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_force(force, dofs_idx, envs_idx)

    @gs.assert_built
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
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_velocity(velocity, dofs_idx, envs_idx)

    @gs.assert_built
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
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_position(position, dofs_idx, envs_idx)

    @gs.assert_built
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
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_position_velocity(position, velocity, dofs_idx, envs_idx)

    @gs.assert_built
    def get_qpos(self, qs_idx_local=None, envs_idx=None):
        """
        Get the entity's qpos.

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

    @gs.assert_built
    def get_mass_mat(self, envs_idx=None, decompose=False):
        dofs_idx = self._get_global_idx(None, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_mass_mat(dofs_idx, envs_idx, decompose)

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
            np.logical_and(
                all_collision_pairs >= self.geom_start,
                all_collision_pairs < self.geom_end,
            ).any(axis=1)
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
            torch.logical_and(
                contact_data["geom_a"] >= self.geom_start,
                contact_data["geom_a"] < self.geom_end,
            ),
            torch.logical_and(
                contact_data["geom_b"] >= self.geom_start,
                contact_data["geom_b"] < self.geom_end,
            ),
        )
        if with_entity is not None and self.idx != with_entity.idx:
            valid_mask = torch.logical_and(
                valid_mask,
                torch.logical_or(
                    torch.logical_and(
                        contact_data["geom_a"] >= with_entity.geom_start,
                        contact_data["geom_a"] < with_entity.geom_end,
                    ),
                    torch.logical_and(
                        contact_data["geom_b"] >= with_entity.geom_start,
                        contact_data["geom_b"] < with_entity.geom_end,
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
        tensor = qd_to_torch(self._solver.links_state.contact_force, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self._solver.n_envs == 0 else tensor

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
            links_mass = qd_to_numpy(self._solver.links_info.inertial_mass, None, links_idx, transpose=True)
            return links_mass.sum(axis=1)

        # Original behavior: sum link masses to scalar
        mass = 0.0
        for link in self.links:
            mass += link.get_mass()
        return mass

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        if self._enable_heterogeneous:
            return "heterogeneous"

        morph = self._morph

        if isinstance(morph, gs.morphs.Box):
            return "box"
        if isinstance(morph, gs.morphs.Sphere):
            return "sphere"
        if isinstance(morph, gs.morphs.Cylinder):
            return "cylinder"
        if isinstance(morph, gs.morphs.Plane):
            return "plane"
        if isinstance(morph, gs.morphs.Mesh):
            return Path(morph.file).stem
        if isinstance(morph, gs.morphs.URDF):
            if isinstance(morph.file, str):
                # Try to get robot name from URDF file, fall back to filename stem
                try:
                    return uu.get_robot_name(morph.file)
                except (ValueError, ET.ParseError, FileNotFoundError, OSError) as e:
                    gs.logger.warning(f"Could not extract robot name from URDF: {e}. Using filename stem instead.")
                    return Path(morph.file).stem
            return morph.file.name
        if isinstance(morph, gs.morphs.MJCF):
            if isinstance(morph.file, str):
                # Try to get model name from MJCF file, fall back to filename stem
                model_name = mju.get_model_name(morph.file)
                if model_name:
                    return model_name
                return Path(morph.file).stem
            return morph.file.name
        if isinstance(morph, gs.morphs.Drone):
            if isinstance(morph.file, str):
                return Path(morph.file).stem
            return morph.file.name
        if isinstance(morph, gs.morphs.USD):
            if morph.prim_path:
                return morph.prim_path.rstrip("/").split("/")[-1]
            return Path(morph.file).stem
        if isinstance(morph, gs.morphs.Terrain):
            return morph.name if morph.name else "terrain"
        return "rigid"

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
    def visualize_contact(self):
        """Whether to visualize contact force."""
        return self._visualize_contact

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
    def main_morph(self):
        """The main morph of the entity (first morph for heterogeneous entities)."""
        return self._morph

    @property
    def morphs(self):
        """All morphs of the entity (main morph + heterogeneous variants if any)."""
        return (self._morph, *self._morph_heterogeneous)

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
    def n_geoms(self):
        """The number of `RigidGeom` in the entity."""
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
    def base_link_idx(self):
        """The index of the entity's base link in the scene."""
        return self._link_start

    @property
    def gravity_compensation(self):
        """Apply a force to compensate gravity. A value of 1 will make a zero-gravity behavior. Default to 0"""
        return self.material.gravity_compensation

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
    def vert_start(self):
        """The index of the entity's first `vert` (collision vertex) in the scene."""
        return self._vert_start

    @property
    def vvert_start(self):
        """The index of the entity's first `vvert` (visual vertex) in the scene."""
        return self._vvert_start

    @property
    def face_start(self):
        """The index of the entity's first `face` (collision face) in the scene."""
        return self._face_start

    @property
    def vface_start(self):
        """The index of the entity's first `vface` (visual face) in the scene."""
        return self._vface_start

    @property
    def edge_start(self):
        """The index of the entity's first `edge` (collision edge) in the scene."""
        return self._edge_start

    @property
    def q_start(self):
        """The index of the entity's first `q` (generalized coordinates) in the scene."""
        return self._q_start

    @property
    def q_end(self):
        """The index of the entity's last `q` (generalized coordinates) in the scene *plus one*."""
        return self._q_start + self.n_qs

    @property
    def geoms(self) -> list[RigidGeom]:
        """The list of collision geoms (`RigidGeom`) in the entity."""
        if self.is_built:
            return self._geoms
        return gs.List(geom for link in self._links for geom in link.geoms)

    @property
    def vgeoms(self):
        """The list of visual geoms (`RigidVisGeom`) in the entity."""
        if self.is_built:
            return self._vgeoms
        return gs.List(vgeom for link in self._links for vgeom in link.vgeoms)

    @property
    def links(self) -> list[RigidLink]:
        """The list of links (`RigidLink`) in the entity."""
        return self._links

    @property
    def joints(self):
        """The list of joints (`RigidJoint`) in the entity."""
        return tuple(chain.from_iterable(self._joints))

    @property
    def joints_by_links(self):
        """The list of joints (`RigidJoint`) in the entity grouped by parent links."""
        return self._joints

    @property
    def base_link(self):
        """The base link of the entity"""
        return self._links[0]

    @property
    def base_joint(self):
        """The base joint of the entity"""
        return self._joints[0][0]

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


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_get_free_verts(
    tensor: qd.types.ndarray(),
    free_verts_idx_local: qd.types.ndarray(),
    verts_state_start: qd.i32,
    free_verts_state: array_class.VertsState,
):
    n_verts = free_verts_idx_local.shape[0]
    _B = tensor.shape[0]
    for i_v_, i, i_b in qd.ndrange(n_verts, 3, _B):
        i_v = i_v_ + verts_state_start
        tensor[i_b, free_verts_idx_local[i_v_], i] = free_verts_state.pos[i_v, i_b][i]


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_get_fixed_verts(
    tensor: qd.types.ndarray(),
    fixed_verts_idx_local: qd.types.ndarray(),
    verts_state_start: qd.i32,
    fixed_verts_state: array_class.VertsState,
):
    n_verts = fixed_verts_idx_local.shape[0]
    _B = tensor.shape[0]
    for i_v_, i, i_b in qd.ndrange(n_verts, 3, _B):
        i_v = i_v_ + verts_state_start
        tensor[i_b, fixed_verts_idx_local[i_v_], i] = fixed_verts_state.pos[i_v][i]
