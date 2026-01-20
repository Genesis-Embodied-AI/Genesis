"""
Loader mixin for RigidEntity.

Contains methods for loading various morph types (mesh, URDF, MJCF, primitives, terrain).
"""

from copy import copy
from itertools import chain
from typing import Any

import numpy as np
import trimesh

import genesis as gs
from genesis.options.morphs import Morph
from genesis.utils import geom as gu
from genesis.utils import mesh as mu
from genesis.utils import mjcf as mju
from genesis.utils import terrain as tu
from genesis.utils import urdf as uu

from ..rigid_geom import RigidGeom
from ..rigid_joint import RigidJoint
from ..rigid_link import RigidLink


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
    from genesis.utils.urdf import compose_inertial_properties, rotate_inertia

    total_mass = gs.EPS
    total_com = np.zeros(3, dtype=gs.np_float)
    total_inertia = np.zeros((3, 3), dtype=gs.np_float)

    # Use collision geoms if available, otherwise fall back to visual geoms
    g_infos = cg_infos if cg_infos else vg_infos
    mesh_key = "mesh" if cg_infos else "vmesh"

    for g_info in g_infos:
        mesh = g_info.get(mesh_key)
        if mesh is None:
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
        geom_com_world = geom_pos + gu.quat_to_R(geom_quat) @ geom_com_local

        geom_inertia = np.array(inertia_mesh.moment_inertia, dtype=gs.np_float) * rho
        geom_inertia = rotate_inertia(geom_inertia, gu.quat_to_R(geom_quat))

        total_mass, total_com, total_inertia = compose_inertial_properties(
            total_mass, total_com, total_inertia, geom_mass, geom_com_world, geom_inertia
        )

    return total_mass, total_com, total_inertia


class RigidEntityLoaderMixin:
    """Mixin class providing morph loading functionality for RigidEntity."""

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
                mesh = gs.Mesh.from_trimesh(
                    mesh=tmesh,
                    surface=gs.surfaces.Collision(),
                )
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

        link_name = morph.file.rsplit("/", 1)[-1].replace(".", "_")

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
            # Mujoco's unified MJCF+URDF parser for only link, joints, and collision geometries properties.
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
            except (ValueError, AssertionError):
                gs.logger.info("Falling back to legacy URDF parser. Default values of physics properties may be off.")
        elif isinstance(morph, gs.morphs.USD):
            from genesis.utils.usd import parse_usd_rigid_entity

            # Unified parser handles both articulations and rigid bodies
            l_infos, links_j_infos, links_g_infos, eqs_info = parse_usd_rigid_entity(morph, surface)
        # Add free floating joint at root if necessary
        if (
            (isinstance(morph, gs.morphs.Drone) or (isinstance(morph, gs.morphs.URDF) and not morph.fixed))
            and links_j_infos
            and sum(j_info["n_dofs"] for j_info in links_j_infos[0]) == 0
        ):
            # Select the second joint down the kinematic tree if possible without messing up with fixed links to keep
            root_idx = 0
            for idx, (l_info, link_j_infos) in tuple(enumerate(zip(l_infos, links_j_infos)))[:2]:
                if (
                    l_info["name"] not in morph.links_to_keep
                    and l_info["parent_idx"] in (0, -1)
                    and sum(j_info["n_dofs"] for j_info in link_j_infos) == 0
                ):
                    root_idx = idx
                    continue
                break

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
            links_j_infos[root_idx] = [j_info]

            # Rename root link for clarity if relevant
            if root_idx == 0:
                l_infos[root_idx]["name"] = "base"

            # Shift root idx for all child links and replace root if no longer fixed wrt world
            for i_l in range(root_idx, len(l_infos)):
                l_info = l_infos[i_l]
                if "root_idx" in l_info and l_info["root_idx"] in (root_idx + 1, i_l):
                    l_info["root_idx"] = root_idx

            # Must invalidate invweight for all child links and joints because the root joint was fixed when it was
            # initially computed. Re-initialize it to some strictly negative value to trigger recomputation in solver.
            for i_l in range(root_idx, len(l_infos)):
                l_infos[i_l]["invweight"] = np.full((2,), fill_value=-1.0)
                for j_info in links_j_infos[i_l]:
                    j_info["dofs_invweight"] = np.full((j_info["n_dofs"],), fill_value=-1.0)

        # Remove the world link if deemed "useless", i.e. fixed joint without any geometry attached
        if not links_g_infos[0] and sum(j_info["n_dofs"] for j_info in links_j_infos[0]) == 0:
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

        # Force recomputing inertial information based on geometry if ill-defined for some reason
        is_inertia_invalid = False
        for l_info, link_j_infos in zip(l_infos, links_j_infos):
            if not all(j_info["type"] == gs.JOINT_TYPE.FIXED for j_info in link_j_infos) and (
                (l_info.get("inertial_mass") is None or l_info["inertial_mass"] <= 0.0)
                or (l_info.get("inertial_i") is None or np.diag(l_info["inertial_i"]) <= 0.0).any()
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
