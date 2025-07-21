from typing import Callable
import dataclasses
import os
import inspect
from typing import Any, Type, cast

import taichi as ti

import genesis as gs
import numpy as np

# as a temporary solution, we get is_ndarray from os's environment variable
use_ndarray = os.environ.get("GS_USE_NDARRAY", "0") == "1"
V = ti.ndarray if use_ndarray else ti.field
V_ANNOTATION = ti.types.ndarray() if use_ndarray else ti.template()
V_VEC = ti.Vector.ndarray if use_ndarray else ti.Vector.field
V_MAT = ti.Matrix.ndarray if use_ndarray else ti.Matrix.field

# =========================================== RigidGlobalInfo ===========================================


@dataclasses.dataclass
class StructRigidGlobalInfo:
    n_awake_dofs: V_ANNOTATION
    awake_dofs: V_ANNOTATION
    n_awake_entities: V_ANNOTATION
    awake_entities: V_ANNOTATION
    qpos0: V_ANNOTATION
    qpos: V_ANNOTATION
    links_T: V_ANNOTATION
    envs_offset: V_ANNOTATION
    geoms_init_AABB: V_ANNOTATION
    mass_mat: V_ANNOTATION
    mass_mat_L: V_ANNOTATION
    mass_mat_D_inv: V_ANNOTATION
    _mass_mat_mask: V_ANNOTATION
    meaninertia: V_ANNOTATION
    mass_parent_mask: V_ANNOTATION


def get_rigid_global_info(solver):
    f_batch = solver._batch_shape

    # Basic fields
    kwargs = {
        "n_awake_dofs": V(dtype=gs.ti_int, shape=f_batch()),
        "awake_dofs": V(dtype=gs.ti_int, shape=f_batch(solver.n_dofs_)),
        "n_awake_entities": V(dtype=gs.ti_int, shape=f_batch()),
        "awake_entities": V(dtype=gs.ti_int, shape=f_batch(solver.n_entities_)),
        "qpos0": V(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_qs_)),
        "qpos": V(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_qs_)),
        "links_T": V_MAT(n=4, m=4, dtype=gs.ti_float, shape=solver.n_links),
        "envs_offset": V_VEC(3, dtype=gs.ti_float, shape=f_batch()),
        "geoms_init_AABB": V_VEC(3, dtype=gs.ti_float, shape=(solver.n_geoms_, 8)),
        "mass_mat": V(dtype=gs.ti_float, shape=solver._batch_shape((solver.n_dofs_, solver.n_dofs_))),
        "mass_mat_L": V(dtype=gs.ti_float, shape=solver._batch_shape((solver.n_dofs_, solver.n_dofs_))),
        "mass_mat_D_inv": V(dtype=gs.ti_float, shape=solver._batch_shape((solver.n_dofs_,))),
        "_mass_mat_mask": V(dtype=gs.ti_int, shape=solver._batch_shape(solver.n_entities_)),
        "meaninertia": V(dtype=gs.ti_float, shape=solver._batch_shape()),
        "mass_parent_mask": V(dtype=gs.ti_float, shape=(solver.n_dofs_, solver.n_dofs_)),
    }

    if use_ndarray:
        obj = StructRigidGlobalInfo(**kwargs)
        # Initialize mass matrix data
        _init_mass_mat_data(obj, solver)
        return obj
    else:

        @ti.data_oriented
        class ClassRigidGlobalInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                _init_mass_mat_data(self, solver)

        return ClassRigidGlobalInfo()


def _init_mass_mat_data(obj, solver):
    # tree structure information
    mass_parent_mask = np.zeros((solver.n_dofs_, solver.n_dofs_), dtype=gs.np_float)

    for i in range(solver.n_links):
        j = i
        while j != -1:
            for i_d, j_d in ti.ndrange(
                (solver.links[i].dof_start, solver.links[i].dof_end),
                (solver.links[j].dof_start, solver.links[j].dof_end),
            ):
                mass_parent_mask[i_d, j_d] = 1.0
            j = solver.links[j].parent_idx

    obj.mass_parent_mask.from_numpy(mass_parent_mask)
    obj._mass_mat_mask.fill(1)
    obj.mass_mat_L.fill(0)
    obj.mass_mat_D_inv.fill(0)
    obj.meaninertia.fill(0)


# =========================================== Constraint ===========================================


@dataclasses.dataclass
class StructConstraintState:
    n_constraints: V_ANNOTATION
    ti_n_equalities: V_ANNOTATION
    jac: V_ANNOTATION
    diag: V_ANNOTATION
    aref: V_ANNOTATION
    jac_relevant_dofs: V_ANNOTATION
    jac_n_relevant_dofs: V_ANNOTATION
    n_constraints_equality: V_ANNOTATION
    improved: V_ANNOTATION
    Jaref: V_ANNOTATION
    Ma: V_ANNOTATION
    Ma_ws: V_ANNOTATION
    grad: V_ANNOTATION
    Mgrad: V_ANNOTATION
    search: V_ANNOTATION
    efc_D: V_ANNOTATION
    efc_force: V_ANNOTATION
    active: V_ANNOTATION
    prev_active: V_ANNOTATION
    qfrc_constraint: V_ANNOTATION
    qacc: V_ANNOTATION
    qacc_ws: V_ANNOTATION
    qacc_prev: V_ANNOTATION
    cost_ws: V_ANNOTATION
    gauss: V_ANNOTATION
    cost: V_ANNOTATION
    prev_cost: V_ANNOTATION
    gtol: V_ANNOTATION
    mv: V_ANNOTATION
    jv: V_ANNOTATION
    quad_gauss: V_ANNOTATION
    quad: V_ANNOTATION
    candidates: V_ANNOTATION
    ls_its: V_ANNOTATION
    ls_result: V_ANNOTATION
    # Optional CG fields
    cg_prev_grad: V_ANNOTATION
    cg_prev_Mgrad: V_ANNOTATION
    cg_beta: V_ANNOTATION
    cg_pg_dot_pMg: V_ANNOTATION
    # Optional Newton fields
    nt_H: V_ANNOTATION
    nt_vec: V_ANNOTATION


def get_constraint_state(constraint_solver, solver):
    f_batch = solver._batch_shape
    len_constraints = constraint_solver.len_constraints
    len_constraints_ = constraint_solver.len_constraints_

    jac_shape = solver._batch_shape((len_constraints_, solver.n_dofs_))
    if (jac_shape[0] * jac_shape[1] * jac_shape[2]) > np.iinfo(np.int32).max:
        raise ValueError(
            f"Jacobian shape {jac_shape} is too large for int32. "
            "Consider reducing the number of constraints or the number of degrees of freedom."
        )

    kwargs = {
        "n_constraints": ti.field(dtype=gs.ti_int, shape=f_batch()),
        "ti_n_equalities": ti.field(gs.ti_int, shape=solver._batch_shape()),
        "jac": ti.field(dtype=gs.ti_float, shape=solver._batch_shape((len_constraints_, solver.n_dofs_))),
        "diag": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(len_constraints_)),
        "aref": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(len_constraints_)),
        "jac_relevant_dofs": ti.field(gs.ti_int, shape=solver._batch_shape((len_constraints_, solver.n_dofs_))),
        "jac_n_relevant_dofs": ti.field(gs.ti_int, shape=solver._batch_shape(len_constraints_)),
        "n_constraints_equality": ti.field(gs.ti_int, shape=solver._batch_shape()),
        "improved": ti.field(gs.ti_int, shape=solver._batch_shape()),
        "Jaref": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(len_constraints_)),
        "Ma": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
        "Ma_ws": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
        "grad": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
        "Mgrad": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
        "search": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
        "efc_D": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(len_constraints_)),
        "efc_force": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(len_constraints_)),
        "active": ti.field(dtype=gs.ti_int, shape=solver._batch_shape(len_constraints_)),
        "prev_active": ti.field(dtype=gs.ti_int, shape=solver._batch_shape(len_constraints_)),
        "qfrc_constraint": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
        "qacc": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
        "qacc_ws": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
        "qacc_prev": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
        "cost_ws": ti.field(gs.ti_float, shape=solver._batch_shape()),
        "gauss": ti.field(gs.ti_float, shape=solver._batch_shape()),
        "cost": ti.field(gs.ti_float, shape=solver._batch_shape()),
        "prev_cost": ti.field(gs.ti_float, shape=solver._batch_shape()),
        "gtol": ti.field(gs.ti_float, shape=solver._batch_shape()),
        "mv": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
        "jv": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(len_constraints_)),
        "quad_gauss": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(3)),
        "quad": ti.field(dtype=gs.ti_float, shape=solver._batch_shape((len_constraints_, 3))),
        "candidates": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(12)),
        "ls_its": ti.field(gs.ti_float, shape=solver._batch_shape()),
        "ls_result": ti.field(gs.ti_int, shape=solver._batch_shape()),
    }

    # Add solver-specific fields
    if constraint_solver._solver_type == gs.constraint_solver.CG:
        kwargs.update(
            {
                "cg_prev_grad": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
                "cg_prev_Mgrad": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
                "cg_beta": ti.field(gs.ti_float, shape=solver._batch_shape()),
                "cg_pg_dot_pMg": ti.field(gs.ti_float, shape=solver._batch_shape()),
            }
        )

    if constraint_solver._solver_type == gs.constraint_solver.Newton:
        kwargs.update(
            {
                "nt_H": ti.field(dtype=gs.ti_float, shape=solver._batch_shape((solver.n_dofs_, solver.n_dofs_))),
                "nt_vec": ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_dofs_)),
            }
        )

    if use_ndarray:
        obj = StructConstraintState(**kwargs)
        # Initialize ti_n_equalities
        obj.ti_n_equalities.from_numpy(np.full((solver._B,), solver.n_equalities, dtype=gs.np_int))
        return obj
    else:

        @ti.data_oriented
        class ClassConstraintState:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                self.ti_n_equalities.from_numpy(np.full((solver._B,), solver.n_equalities, dtype=gs.np_int))

        return ClassConstraintState()


# =========================================== Collider ===========================================


@dataclasses.dataclass
class StructContactData:
    geom_a: ti.types.NDArray[ti.i32, 2]
    geom_b: ti.types.NDArray[ti.i32, 2]
    penetration: ti.types.NDArray[ti.f32, 2]
    normal: ti.types.NDArray[ti.types.vector(3, ti.f32), 2]
    pos: ti.types.NDArray[ti.types.vector(3, ti.f32), 2]
    friction: ti.types.NDArray[ti.f32, 2]
    sol_params: ti.types.NDArray[ti.types.vector(7, ti.f32), 2]
    force: ti.types.NDArray[ti.types.vector(3, ti.f32), 2]
    link_a: ti.types.NDArray[ti.i32, 2]
    link_b: ti.types.NDArray[ti.i32, 2]


def get_contact_data(solver, max_contact_pairs):
    f_batch = solver._batch_shape
    max_contact_pairs_ = max(1, max_contact_pairs)
    kwargs = {
        "geom_a": V(dtype=gs.ti_int, shape=f_batch(max_contact_pairs_)),
        "geom_b": V(dtype=gs.ti_int, shape=f_batch(max_contact_pairs_)),
        "penetration": V(dtype=gs.ti_float, shape=f_batch(max_contact_pairs_)),
        "normal": V(dtype=gs.ti_vec3, shape=f_batch(max_contact_pairs_)),
        "pos": V(dtype=gs.ti_vec3, shape=f_batch(max_contact_pairs_)),
        "friction": V(dtype=gs.ti_float, shape=f_batch(max_contact_pairs_)),
        "sol_params": V(dtype=gs.ti_vec7, shape=f_batch(max_contact_pairs_)),
        "force": V(dtype=gs.ti_vec3, shape=f_batch(max_contact_pairs_)),
        "link_a": V(dtype=gs.ti_int, shape=f_batch(max_contact_pairs_)),
        "link_b": V(dtype=gs.ti_int, shape=f_batch(max_contact_pairs_)),
    }

    if use_ndarray:
        obj = StructContactData(**kwargs)
        return obj
    else:

        @ti.data_oriented
        class ClassContactData:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassContactData()


@dataclasses.dataclass
class StructSortBuffer:
    value: V_ANNOTATION
    i_g: V_ANNOTATION
    is_max: V_ANNOTATION


def get_sort_buffer(solver):
    f_batch = solver._batch_shape
    kwargs = {
        "value": V(dtype=gs.ti_float, shape=f_batch(2 * solver.n_geoms_)),
        "i_g": V(dtype=gs.ti_int, shape=f_batch(2 * solver.n_geoms_)),
        "is_max": V(dtype=gs.ti_int, shape=f_batch(2 * solver.n_geoms_)),
    }
    if use_ndarray:
        return StructSortBuffer(**kwargs)
    else:

        @ti.data_oriented
        class ClassSortBuffer:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassSortBuffer()


@dataclasses.dataclass
class StructContactCache:
    normal: ti.types.NDArray[ti.types.vector(3, ti.f32), 3]
    # FIXME: cannot use V_ANNOTATION?
    # normal: V_ANNOTATION


def get_contact_cache(solver):
    f_batch = solver._batch_shape
    n_geoms = solver.n_geoms_
    kwargs = {
        "normal": V(dtype=gs.ti_vec3, shape=f_batch((n_geoms, n_geoms))),
    }
    if use_ndarray:
        return StructContactCache(**kwargs)
    else:

        @ti.data_oriented
        class ClassContactCache:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassContactCache()


@dataclasses.dataclass
class StructColliderState:
    # sort_buffer: StructSortBuffer
    contact_data: StructContactData
    active_buffer: V_ANNOTATION
    n_broad_pairs: V_ANNOTATION
    broad_collision_pairs: V_ANNOTATION
    active_buffer_awake: V_ANNOTATION
    active_buffer_hib: V_ANNOTATION
    box_depth: V_ANNOTATION
    box_points: V_ANNOTATION
    box_pts: V_ANNOTATION
    box_lines: V_ANNOTATION
    box_linesu: V_ANNOTATION
    box_axi: V_ANNOTATION
    box_ppts2: V_ANNOTATION
    box_pu: V_ANNOTATION
    xyz_max_min: V_ANNOTATION
    prism: V_ANNOTATION
    n_contacts: V_ANNOTATION
    n_contacts_hibernated: V_ANNOTATION
    first_time: V_ANNOTATION
    contact_cache: StructContactCache


def get_collider_state(solver, n_possible_pairs, collider_static_config):
    _B = solver._B
    f_batch = solver._batch_shape
    n_geoms = solver.n_geoms_
    max_collision_pairs = min(solver._max_collision_pairs, n_possible_pairs)
    max_collision_pairs_broad = max_collision_pairs * collider_static_config.max_collision_pairs_broad_k
    max_contact_pairs = max_collision_pairs * collider_static_config.n_contacts_per_pair

    ############## broad phase SAP ##############

    contact_data = get_contact_data(solver, max_contact_pairs)
    sort_buffer = get_sort_buffer(solver)
    contact_cache = get_contact_cache(solver)
    kwargs = {
        # "sort_buffer": sort_buffer,
        "contact_data": contact_data,
        "active_buffer": V(dtype=gs.ti_int, shape=f_batch(n_geoms)),
        "n_broad_pairs": V(dtype=gs.ti_int, shape=_B),
        "broad_collision_pairs": V(dtype=gs.ti_int, shape=f_batch(max(1, max_collision_pairs_broad))),
        "active_buffer_awake": V(dtype=gs.ti_int, shape=f_batch(n_geoms)),
        "active_buffer_hib": V(dtype=gs.ti_int, shape=f_batch(n_geoms)),
        "box_depth": V(dtype=gs.ti_float, shape=f_batch(collider_static_config.box_MAXCONPAIR)),
        "box_points": V(gs.ti_vec3, shape=f_batch(collider_static_config.box_MAXCONPAIR)),
        "box_pts": V(gs.ti_vec3, shape=f_batch(6)),
        "box_lines": V(gs.ti_vec6, shape=f_batch(4)),
        "box_linesu": V(gs.ti_vec6, shape=f_batch(4)),
        "box_axi": V(gs.ti_vec3, shape=f_batch(3)),
        "box_ppts2": V(dtype=gs.ti_float, shape=f_batch((4, 2))),
        "box_pu": V(gs.ti_vec3, shape=f_batch(4)),
        "xyz_max_min": V(dtype=gs.ti_float, shape=f_batch(6)),
        "prism": V(dtype=gs.ti_vec3, shape=f_batch(6)),
        "n_contacts": V(dtype=gs.ti_int, shape=_B),
        "n_contacts_hibernated": V(dtype=gs.ti_int, shape=_B),
        "first_time": V(dtype=gs.ti_int, shape=_B),
        "contact_cache": contact_cache,
    }

    if use_ndarray:
        return StructColliderState(**kwargs)
    else:

        @ti.data_oriented
        class ClassColliderState:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassColliderState()


@dataclasses.dataclass
class StructColliderInfo:
    vert_neighbors: V_ANNOTATION
    vert_neighbor_start: V_ANNOTATION
    vert_n_neighbors: V_ANNOTATION
    collision_pair_validity: V_ANNOTATION
    _max_possible_pairs: V_ANNOTATION
    _max_collision_pairs: V_ANNOTATION
    _max_contact_pairs: V_ANNOTATION
    _max_collision_pairs_broad: V_ANNOTATION
    # Terrain fields
    terrain_hf: V_ANNOTATION
    terrain_rc: V_ANNOTATION
    terrain_scale: V_ANNOTATION
    terrain_xyz_maxmin: V_ANNOTATION


def get_collider_info(solver, n_vert_neighbors, collider_static_config):
    n_geoms = solver.n_geoms_
    n_verts = solver.n_verts_

    ########## Terrain contact detection ##########
    terrain_hf_shape = 1
    if collider_static_config.has_terrain:
        links_idx = solver.geoms_info.link_idx.to_numpy()[solver.geoms_info.type.to_numpy() == gs.GEOM_TYPE.TERRAIN]
        entity = solver._entities[solver.links_info.entity_idx.to_numpy()[links_idx[0]]]
        terrain_hf_shape = entity.terrain_hf.shape

    kwargs = {
        "vert_neighbors": V(dtype=gs.ti_int, shape=max(1, n_vert_neighbors)),
        "vert_neighbor_start": V(dtype=gs.ti_int, shape=n_verts),
        "vert_n_neighbors": V(dtype=gs.ti_int, shape=n_verts),
        "collision_pair_validity": V(dtype=gs.ti_int, shape=(n_geoms, n_geoms)),
        "_max_possible_pairs": V(dtype=gs.ti_int, shape=()),
        "_max_collision_pairs": V(dtype=gs.ti_int, shape=()),
        "_max_contact_pairs": V(dtype=gs.ti_int, shape=()),
        "_max_collision_pairs_broad": V(dtype=gs.ti_int, shape=()),
        "terrain_hf": V(dtype=gs.ti_float, shape=terrain_hf_shape),
        "terrain_rc": V(dtype=gs.ti_int, shape=2),
        "terrain_scale": V(dtype=gs.ti_float, shape=2),
        "terrain_xyz_maxmin": V(dtype=gs.ti_float, shape=6),
    }

    if use_ndarray:
        return StructColliderInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassColliderInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassColliderInfo()


# =========================================== MPR ===========================================


@dataclasses.dataclass
class StructMPRState:
    simplex_support: V_ANNOTATION
    simplex_size: V_ANNOTATION


def get_mpr_state(f_batch):
    struct_support = ti.types.struct(
        v1=gs.ti_vec3,
        v2=gs.ti_vec3,
        v=gs.ti_vec3,
    )

    kwargs = {
        "simplex_support": struct_support.field(shape=f_batch(4)),
        "simplex_size": ti.field(gs.ti_int, shape=f_batch()),
    }

    if use_ndarray:
        return StructMPRState(**kwargs)
    else:

        @ti.data_oriented
        class ClassMPRState:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassMPRState()


# =========================================== GJK ===========================================


@dataclasses.dataclass
class StructGJKState:
    support_mesh_prev_vertex_id: V_ANNOTATION
    simplex_vertex: V_ANNOTATION
    simplex_buffer: V_ANNOTATION
    simplex: V_ANNOTATION
    simplex_vertex_intersect: V_ANNOTATION
    simplex_buffer_intersect: V_ANNOTATION
    nsimplex: V_ANNOTATION
    last_searched_simplex_vertex_id: V_ANNOTATION
    polytope: V_ANNOTATION
    polytope_verts: V_ANNOTATION
    polytope_faces: V_ANNOTATION
    polytope_horizon_data: V_ANNOTATION
    polytope_faces_map: V_ANNOTATION
    polytope_horizon_stack: V_ANNOTATION
    contact_faces: V_ANNOTATION
    contact_normals: V_ANNOTATION
    contact_halfspaces: V_ANNOTATION
    contact_clipped_polygons: V_ANNOTATION
    multi_contact_flag: V_ANNOTATION
    witness: V_ANNOTATION
    n_witness: V_ANNOTATION
    n_contacts: V_ANNOTATION
    contact_pos: V_ANNOTATION
    normal: V_ANNOTATION
    is_col: V_ANNOTATION
    penetration: V_ANNOTATION
    distance: V_ANNOTATION


def get_gjk_state(solver, static_rigid_sim_config, gjk_static_config):
    _B = solver._B
    polytope_max_faces = gjk_static_config.polytope_max_faces
    max_contacts_per_pair = gjk_static_config.max_contacts_per_pair
    max_contact_polygon_verts = gjk_static_config.max_contact_polygon_verts

    # Struct definitions
    struct_simplex_vertex = ti.types.struct(
        obj1=gs.ti_vec3,
        obj2=gs.ti_vec3,
        id1=gs.ti_int,
        id2=gs.ti_int,
        mink=gs.ti_vec3,
    )
    struct_simplex = ti.types.struct(
        nverts=gs.ti_int,
        dist=gs.ti_float,
    )
    struct_simplex_buffer = ti.types.struct(
        normal=gs.ti_vec3,
        sdist=gs.ti_float,
    )

    kwargs = {
        "support_mesh_prev_vertex_id": ti.field(dtype=gs.ti_int, shape=(_B, 2)),
        "simplex_vertex": struct_simplex_vertex.field(shape=(_B, 4)),
        "simplex_buffer": struct_simplex_buffer.field(shape=(_B, 4)),
        "simplex": struct_simplex.field(shape=(_B,)),
        "last_searched_simplex_vertex_id": ti.field(dtype=gs.ti_int, shape=(_B,)),
    }

    # MuJoCo compatibility fields
    if static_rigid_sim_config.enable_mujoco_compatibility:
        kwargs.update(
            {
                "simplex_vertex_intersect": struct_simplex_vertex.field(shape=(_B, 4)),
                "simplex_buffer_intersect": struct_simplex_buffer.field(shape=(_B, 4)),
                "nsimplex": ti.field(dtype=gs.ti_int, shape=(_B,)),
            }
        )

    ### EPA polytope
    struct_polytope_vertex = struct_simplex_vertex
    struct_polytope_face = ti.types.struct(
        verts_idx=gs.ti_ivec3,
        adj_idx=gs.ti_ivec3,
        normal=gs.ti_vec3,
        dist2=gs.ti_float,
        map_idx=gs.ti_int,
    )
    struct_polytope_horizon_data = ti.types.struct(
        face_idx=gs.ti_int,
        edge_idx=gs.ti_int,
    )
    struct_polytope = ti.types.struct(
        nverts=gs.ti_int,
        nfaces=gs.ti_int,
        nfaces_map=gs.ti_int,
        horizon_nedges=gs.ti_int,
        horizon_w=gs.ti_vec3,
    )

    kwargs.update(
        {
            "polytope": struct_polytope.field(shape=(_B,)),
            "polytope_verts": struct_polytope_vertex.field(shape=(_B, 5 + gjk_static_config.epa_max_iterations)),
            "polytope_faces": struct_polytope_face.field(shape=(_B, polytope_max_faces)),
            "polytope_horizon_data": struct_polytope_horizon_data.field(
                shape=(_B, 6 + gjk_static_config.epa_max_iterations)
            ),
            "polytope_faces_map": ti.Vector.field(n=polytope_max_faces, dtype=gs.ti_int, shape=(_B,)),
            "polytope_horizon_stack": struct_polytope_horizon_data.field(shape=(_B, polytope_max_faces * 3)),
        }
    )

    # Multi-contact detection
    if gjk_static_config.enable_mujoco_multi_contact:
        struct_contact_face = ti.types.struct(
            vert1=gs.ti_vec3,
            vert2=gs.ti_vec3,
            endverts=gs.ti_vec3,
            normal1=gs.ti_vec3,
            normal2=gs.ti_vec3,
            id1=gs.ti_int,
            id2=gs.ti_int,
        )
        struct_contact_normal = ti.types.struct(
            endverts=gs.ti_vec3,
            normal=gs.ti_vec3,
            id=gs.ti_int,
        )
        struct_contact_halfspace = ti.types.struct(
            normal=gs.ti_vec3,
            dist=gs.ti_float,
        )
        kwargs.update(
            {
                "contact_faces": struct_contact_face.field(shape=(_B, max_contact_polygon_verts)),
                "contact_normals": struct_contact_normal.field(shape=(_B, max_contact_polygon_verts)),
                "contact_halfspaces": struct_contact_halfspace.field(shape=(_B, max_contact_polygon_verts)),
                "contact_clipped_polygons": gs.ti_vec3.field(shape=(_B, 2, max_contact_polygon_verts)),
            }
        )

    kwargs.update(
        {
            "multi_contact_flag": ti.field(dtype=gs.ti_int, shape=(_B,)),
        }
    )

    ### Final results
    struct_witness = ti.types.struct(
        point_obj1=gs.ti_vec3,
        point_obj2=gs.ti_vec3,
    )

    kwargs.update(
        {
            "witness": struct_witness.field(shape=(_B, max_contacts_per_pair)),
            "n_witness": ti.field(dtype=gs.ti_int, shape=(_B,)),
            "n_contacts": ti.field(dtype=gs.ti_int, shape=(_B,)),
            "contact_pos": gs.ti_vec3.field(shape=(_B, max_contacts_per_pair)),
            "normal": gs.ti_vec3.field(shape=(_B, max_contacts_per_pair)),
            "is_col": ti.field(dtype=gs.ti_int, shape=(_B,)),
            "penetration": ti.field(dtype=gs.ti_float, shape=(_B,)),
            "distance": ti.field(dtype=gs.ti_float, shape=(_B,)),
        }
    )

    if use_ndarray:
        return StructGJKState(**kwargs)
    else:

        @ti.data_oriented
        class ClassGJKState:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassGJKState()


# =========================================== SupportField ===========================================


@dataclasses.dataclass
class StructSupportFieldInfo:
    support_cell_start: V_ANNOTATION
    support_v: V_ANNOTATION
    support_vid: V_ANNOTATION


def get_support_field_info(n_geoms, n_support_cells):
    kwargs = {
        "support_cell_start": ti.field(dtype=gs.ti_int, shape=n_geoms),
        "support_v": ti.Vector.field(3, dtype=gs.ti_float, shape=max(1, n_support_cells)),
        "support_vid": ti.field(dtype=gs.ti_int, shape=max(1, n_support_cells)),
    }

    if use_ndarray:
        return StructSupportFieldInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassSupportFieldInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassSupportFieldInfo()


# =========================================== DofsInfo and DofsState ===========================================


@dataclasses.dataclass
class StructDofsInfo:
    stiffness: V_ANNOTATION
    invweight: V_ANNOTATION
    armature: V_ANNOTATION
    damping: V_ANNOTATION
    motion_ang: V_ANNOTATION
    motion_vel: V_ANNOTATION
    limit: V_ANNOTATION
    dof_start: V_ANNOTATION
    kp: V_ANNOTATION
    kv: V_ANNOTATION
    force_range: V_ANNOTATION


def get_dofs_info(solver):
    shape = solver._batch_shape(solver.n_dofs_) if solver._options.batch_dofs_info else solver.n_dofs_
    kwargs = {
        "stiffness": V(dtype=gs.ti_float, shape=shape),
        "invweight": V(dtype=gs.ti_float, shape=shape),
        "armature": V(dtype=gs.ti_float, shape=shape),
        "damping": V(dtype=gs.ti_float, shape=shape),
        "motion_ang": V(dtype=gs.ti_vec3, shape=shape),
        "motion_vel": V(dtype=gs.ti_vec3, shape=shape),
        "limit": V(dtype=gs.ti_vec2, shape=shape),
        "dof_start": V(dtype=gs.ti_int, shape=shape),
        "kp": V(dtype=gs.ti_float, shape=shape),
        "kv": V(dtype=gs.ti_float, shape=shape),
        "force_range": V(dtype=gs.ti_vec2, shape=shape),
    }

    if use_ndarray:
        return StructDofsInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassDofsInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassDofsInfo()


@dataclasses.dataclass
class StructDofsState:
    force: V_ANNOTATION
    qf_bias: V_ANNOTATION
    qf_passive: V_ANNOTATION
    qf_actuator: V_ANNOTATION
    qf_applied: V_ANNOTATION
    act_length: V_ANNOTATION
    pos: V_ANNOTATION
    vel: V_ANNOTATION
    acc: V_ANNOTATION
    acc_smooth: V_ANNOTATION
    qf_smooth: V_ANNOTATION
    qf_constraint: V_ANNOTATION
    cdof_ang: V_ANNOTATION
    cdof_vel: V_ANNOTATION
    cdofvel_ang: V_ANNOTATION
    cdofvel_vel: V_ANNOTATION
    cdofd_ang: V_ANNOTATION
    cdofd_vel: V_ANNOTATION
    f_vel: V_ANNOTATION
    f_ang: V_ANNOTATION
    ctrl_force: V_ANNOTATION
    ctrl_pos: V_ANNOTATION
    ctrl_vel: V_ANNOTATION
    ctrl_mode: V_ANNOTATION
    hibernated: V_ANNOTATION


def get_dofs_state(solver):
    shape = solver._batch_shape(solver.n_dofs_)
    kwargs = {
        "force": V(dtype=gs.ti_float, shape=shape),
        "qf_bias": V(dtype=gs.ti_float, shape=shape),
        "qf_passive": V(dtype=gs.ti_float, shape=shape),
        "qf_actuator": V(dtype=gs.ti_float, shape=shape),
        "qf_applied": V(dtype=gs.ti_float, shape=shape),
        "act_length": V(dtype=gs.ti_float, shape=shape),
        "pos": V(dtype=gs.ti_float, shape=shape),
        "vel": V(dtype=gs.ti_float, shape=shape),
        "acc": V(dtype=gs.ti_float, shape=shape),
        "acc_smooth": V(dtype=gs.ti_float, shape=shape),
        "qf_smooth": V(dtype=gs.ti_float, shape=shape),
        "qf_constraint": V(dtype=gs.ti_float, shape=shape),
        "cdof_ang": V(dtype=gs.ti_vec3, shape=shape),
        "cdof_vel": V(dtype=gs.ti_vec3, shape=shape),
        "cdofvel_ang": V(dtype=gs.ti_vec3, shape=shape),
        "cdofvel_vel": V(dtype=gs.ti_vec3, shape=shape),
        "cdofd_ang": V(dtype=gs.ti_vec3, shape=shape),
        "cdofd_vel": V(dtype=gs.ti_vec3, shape=shape),
        "f_vel": V(dtype=gs.ti_vec3, shape=shape),
        "f_ang": V(dtype=gs.ti_vec3, shape=shape),
        "ctrl_force": V(dtype=gs.ti_float, shape=shape),
        "ctrl_pos": V(dtype=gs.ti_float, shape=shape),
        "ctrl_vel": V(dtype=gs.ti_float, shape=shape),
        "ctrl_mode": V(dtype=gs.ti_int, shape=shape),
        "hibernated": V(dtype=gs.ti_int, shape=shape),
    }

    if use_ndarray:
        return StructDofsState(**kwargs)
    else:

        @ti.data_oriented
        class ClassDofsState:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassDofsState()


# =========================================== LinksState and LinksInfo ===========================================


@dataclasses.dataclass
class StructLinksState:
    cinr_inertial: V_ANNOTATION
    cinr_pos: V_ANNOTATION
    cinr_quat: V_ANNOTATION
    cinr_mass: V_ANNOTATION
    crb_inertial: V_ANNOTATION
    crb_pos: V_ANNOTATION
    crb_quat: V_ANNOTATION
    crb_mass: V_ANNOTATION
    cdd_vel: V_ANNOTATION
    cdd_ang: V_ANNOTATION
    pos: V_ANNOTATION
    quat: V_ANNOTATION
    i_pos: V_ANNOTATION
    i_quat: V_ANNOTATION
    j_pos: V_ANNOTATION
    j_quat: V_ANNOTATION
    j_vel: V_ANNOTATION
    j_ang: V_ANNOTATION
    cd_ang: V_ANNOTATION
    cd_vel: V_ANNOTATION
    mass_sum: V_ANNOTATION
    COM: V_ANNOTATION
    mass_shift: V_ANNOTATION
    i_pos_shift: V_ANNOTATION
    cacc_ang: V_ANNOTATION
    cacc_lin: V_ANNOTATION
    cfrc_ang: V_ANNOTATION
    cfrc_vel: V_ANNOTATION
    cfrc_applied_ang: V_ANNOTATION
    cfrc_applied_vel: V_ANNOTATION
    contact_force: V_ANNOTATION
    hibernated: V_ANNOTATION


def get_links_state(solver):
    shape = solver._batch_shape(solver.n_links_)
    kwargs = {
        "cinr_inertial": V(dtype=gs.ti_mat3, shape=shape),
        "cinr_pos": V(dtype=gs.ti_vec3, shape=shape),
        "cinr_quat": V(dtype=gs.ti_vec4, shape=shape),
        "cinr_mass": V(dtype=gs.ti_float, shape=shape),
        "crb_inertial": V(dtype=gs.ti_mat3, shape=shape),
        "crb_pos": V(dtype=gs.ti_vec3, shape=shape),
        "crb_quat": V(dtype=gs.ti_vec4, shape=shape),
        "crb_mass": V(dtype=gs.ti_float, shape=shape),
        "cdd_vel": V(dtype=gs.ti_vec3, shape=shape),
        "cdd_ang": V(dtype=gs.ti_vec3, shape=shape),
        "pos": V(dtype=gs.ti_vec3, shape=shape),
        "quat": V(dtype=gs.ti_vec4, shape=shape),
        "i_pos": V(dtype=gs.ti_vec3, shape=shape),
        "i_quat": V(dtype=gs.ti_vec4, shape=shape),
        "j_pos": V(dtype=gs.ti_vec3, shape=shape),
        "j_quat": V(dtype=gs.ti_vec4, shape=shape),
        "j_vel": V(dtype=gs.ti_vec3, shape=shape),
        "j_ang": V(dtype=gs.ti_vec3, shape=shape),
        "cd_ang": V(dtype=gs.ti_vec3, shape=shape),
        "cd_vel": V(dtype=gs.ti_vec3, shape=shape),
        "mass_sum": V(dtype=gs.ti_float, shape=shape),
        "COM": V(dtype=gs.ti_vec3, shape=shape),
        "mass_shift": V(dtype=gs.ti_float, shape=shape),
        "i_pos_shift": V(dtype=gs.ti_vec3, shape=shape),
        "cacc_ang": V(dtype=gs.ti_vec3, shape=shape),
        "cacc_lin": V(dtype=gs.ti_vec3, shape=shape),
        "cfrc_ang": V(dtype=gs.ti_vec3, shape=shape),
        "cfrc_vel": V(dtype=gs.ti_vec3, shape=shape),
        "cfrc_applied_ang": V(dtype=gs.ti_vec3, shape=shape),
        "cfrc_applied_vel": V(dtype=gs.ti_vec3, shape=shape),
        "contact_force": V(dtype=gs.ti_vec3, shape=shape),
        "hibernated": V(dtype=gs.ti_int, shape=shape),
    }

    if use_ndarray:
        return StructLinksState(**kwargs)
    else:

        @ti.data_oriented
        class ClassLinksState:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassLinksState()


@dataclasses.dataclass
class StructLinksInfo:
    parent_idx: V_ANNOTATION
    root_idx: V_ANNOTATION
    q_start: V_ANNOTATION
    dof_start: V_ANNOTATION
    joint_start: V_ANNOTATION
    q_end: V_ANNOTATION
    dof_end: V_ANNOTATION
    joint_end: V_ANNOTATION
    n_dofs: V_ANNOTATION
    pos: V_ANNOTATION
    quat: V_ANNOTATION
    invweight: V_ANNOTATION
    is_fixed: V_ANNOTATION
    inertial_pos: V_ANNOTATION
    inertial_quat: V_ANNOTATION
    inertial_i: V_ANNOTATION
    inertial_mass: V_ANNOTATION
    entity_idx: V_ANNOTATION


def get_links_info(solver):
    links_info_shape = solver._batch_shape(solver.n_links_) if solver._options.batch_links_info else solver.n_links_
    kwargs = {
        "parent_idx": V(dtype=gs.ti_int, shape=links_info_shape),
        "root_idx": V(dtype=gs.ti_int, shape=links_info_shape),
        "q_start": V(dtype=gs.ti_int, shape=links_info_shape),
        "dof_start": V(dtype=gs.ti_int, shape=links_info_shape),
        "joint_start": V(dtype=gs.ti_int, shape=links_info_shape),
        "q_end": V(dtype=gs.ti_int, shape=links_info_shape),
        "dof_end": V(dtype=gs.ti_int, shape=links_info_shape),
        "joint_end": V(dtype=gs.ti_int, shape=links_info_shape),
        "n_dofs": V(dtype=gs.ti_int, shape=links_info_shape),
        "pos": V(dtype=gs.ti_vec3, shape=links_info_shape),
        "quat": V(dtype=gs.ti_vec4, shape=links_info_shape),
        "invweight": V(dtype=gs.ti_vec2, shape=links_info_shape),
        "is_fixed": V(dtype=gs.ti_int, shape=links_info_shape),
        "inertial_pos": V(dtype=gs.ti_vec3, shape=links_info_shape),
        "inertial_quat": V(dtype=gs.ti_vec4, shape=links_info_shape),
        "inertial_i": V(dtype=gs.ti_mat3, shape=links_info_shape),
        "inertial_mass": V(dtype=gs.ti_float, shape=links_info_shape),
        "entity_idx": V(dtype=gs.ti_int, shape=links_info_shape),
    }

    if use_ndarray:
        return StructLinksInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassLinksInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassLinksInfo()


# =========================================== JointsInfo and JointsState ===========================================


@dataclasses.dataclass
class StructJointsInfo:
    type: V_ANNOTATION
    sol_params: V_ANNOTATION
    q_start: V_ANNOTATION
    dof_start: V_ANNOTATION
    q_end: V_ANNOTATION
    dof_end: V_ANNOTATION
    n_dofs: V_ANNOTATION
    pos: V_ANNOTATION


def get_joints_info(solver):
    shape = solver._batch_shape(solver.n_joints_) if solver._options.batch_joints_info else solver.n_joints_
    kwargs = {
        "type": V(dtype=gs.ti_int, shape=shape),
        "sol_params": V(dtype=gs.ti_vec7, shape=shape),
        "q_start": V(dtype=gs.ti_int, shape=shape),
        "dof_start": V(dtype=gs.ti_int, shape=shape),
        "q_end": V(dtype=gs.ti_int, shape=shape),
        "dof_end": V(dtype=gs.ti_int, shape=shape),
        "n_dofs": V(dtype=gs.ti_int, shape=shape),
        "pos": V(dtype=gs.ti_vec3, shape=shape),
    }

    if use_ndarray:
        return StructJointsInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassJointsInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassJointsInfo()


@dataclasses.dataclass
class StructJointsState:
    xanchor: V_ANNOTATION
    xaxis: V_ANNOTATION


def get_joints_state(solver):
    shape = solver._batch_shape(solver.n_joints_)
    kwargs = {
        "xanchor": V(dtype=gs.ti_vec3, shape=shape),
        "xaxis": V(dtype=gs.ti_vec3, shape=shape),
    }

    if use_ndarray:
        return StructJointsState(**kwargs)
    else:

        @ti.data_oriented
        class ClassJointsState:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassJointsState()


# =========================================== GeomsInfo and GeomsState ===========================================


@dataclasses.dataclass
class StructGeomsInfo:
    pos: V_ANNOTATION
    center: V_ANNOTATION
    quat: V_ANNOTATION
    data: V_ANNOTATION
    link_idx: V_ANNOTATION
    type: V_ANNOTATION
    friction: V_ANNOTATION
    sol_params: V_ANNOTATION
    vert_num: V_ANNOTATION
    vert_start: V_ANNOTATION
    vert_end: V_ANNOTATION
    verts_state_start: V_ANNOTATION
    verts_state_end: V_ANNOTATION
    face_num: V_ANNOTATION
    face_start: V_ANNOTATION
    face_end: V_ANNOTATION
    edge_num: V_ANNOTATION
    edge_start: V_ANNOTATION
    edge_end: V_ANNOTATION
    is_convex: V_ANNOTATION
    contype: V_ANNOTATION
    conaffinity: V_ANNOTATION
    is_free: V_ANNOTATION
    is_decomposed: V_ANNOTATION
    needs_coup: V_ANNOTATION
    coup_friction: V_ANNOTATION
    coup_softness: V_ANNOTATION
    coup_restitution: V_ANNOTATION


def get_geoms_info(solver):
    shape = (solver.n_geoms_,)
    kwargs = {
        "pos": V(dtype=gs.ti_vec3, shape=shape),
        "center": V(dtype=gs.ti_vec3, shape=shape),
        "quat": V(dtype=gs.ti_vec4, shape=shape),
        "data": V(dtype=gs.ti_vec7, shape=shape),
        "link_idx": V(dtype=gs.ti_int, shape=shape),
        "type": V(dtype=gs.ti_int, shape=shape),
        "friction": V(dtype=gs.ti_float, shape=shape),
        "sol_params": V(dtype=gs.ti_vec7, shape=shape),
        "vert_num": V(dtype=gs.ti_int, shape=shape),
        "vert_start": V(dtype=gs.ti_int, shape=shape),
        "vert_end": V(dtype=gs.ti_int, shape=shape),
        "verts_state_start": V(dtype=gs.ti_int, shape=shape),
        "verts_state_end": V(dtype=gs.ti_int, shape=shape),
        "face_num": V(dtype=gs.ti_int, shape=shape),
        "face_start": V(dtype=gs.ti_int, shape=shape),
        "face_end": V(dtype=gs.ti_int, shape=shape),
        "edge_num": V(dtype=gs.ti_int, shape=shape),
        "edge_start": V(dtype=gs.ti_int, shape=shape),
        "edge_end": V(dtype=gs.ti_int, shape=shape),
        "is_convex": V(dtype=gs.ti_int, shape=shape),
        "contype": V(dtype=gs.ti_int, shape=shape),
        "conaffinity": V(dtype=gs.ti_int, shape=shape),
        "is_free": V(dtype=gs.ti_int, shape=shape),
        "is_decomposed": V(dtype=gs.ti_int, shape=shape),
        "needs_coup": V(dtype=gs.ti_int, shape=shape),
        "coup_friction": V(dtype=gs.ti_float, shape=shape),
        "coup_softness": V(dtype=gs.ti_float, shape=shape),
        "coup_restitution": V(dtype=gs.ti_float, shape=shape),
    }

    if use_ndarray:
        return StructGeomsInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassGeomsInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassGeomsInfo()


@dataclasses.dataclass
class StructGeomsState:
    pos: V_ANNOTATION
    quat: V_ANNOTATION
    aabb_min: V_ANNOTATION
    aabb_max: V_ANNOTATION
    verts_updated: V_ANNOTATION
    min_buffer_idx: V_ANNOTATION
    max_buffer_idx: V_ANNOTATION
    hibernated: V_ANNOTATION
    friction_ratio: V_ANNOTATION


def get_geoms_state(solver):
    shape = solver._batch_shape(solver.n_geoms_)
    kwargs = {
        "pos": V(dtype=gs.ti_vec3, shape=shape),
        "quat": V(dtype=gs.ti_vec4, shape=shape),
        "aabb_min": V(dtype=gs.ti_vec3, shape=shape),
        "aabb_max": V(dtype=gs.ti_vec3, shape=shape),
        "verts_updated": V(dtype=gs.ti_int, shape=shape),
        "min_buffer_idx": V(dtype=gs.ti_int, shape=shape),
        "max_buffer_idx": V(dtype=gs.ti_int, shape=shape),
        "hibernated": V(dtype=gs.ti_int, shape=shape),
        "friction_ratio": V(dtype=gs.ti_float, shape=shape),
    }

    if use_ndarray:
        return StructGeomsState(**kwargs)
    else:

        @ti.data_oriented
        class ClassGeomsState:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassGeomsState()


# =========================================== VertsInfo ===========================================


@dataclasses.dataclass
class StructVertsInfo:
    init_pos: V_ANNOTATION
    init_normal: V_ANNOTATION
    geom_idx: V_ANNOTATION
    init_center_pos: V_ANNOTATION
    verts_state_idx: V_ANNOTATION
    is_free: V_ANNOTATION


def get_verts_info(solver):
    shape = (solver.n_verts_,)
    kwargs = {
        "init_pos": V(dtype=gs.ti_vec3, shape=shape),
        "init_normal": V(dtype=gs.ti_vec3, shape=shape),
        "geom_idx": V(dtype=gs.ti_int, shape=shape),
        "init_center_pos": V(dtype=gs.ti_vec3, shape=shape),
        "verts_state_idx": V(dtype=gs.ti_int, shape=shape),
        "is_free": V(dtype=gs.ti_int, shape=shape),
    }

    if use_ndarray:
        return StructVertsInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassVertsInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassVertsInfo()


# =========================================== FacesInfo ===========================================


@dataclasses.dataclass
class StructFacesInfo:
    verts_idx: V_ANNOTATION
    geom_idx: V_ANNOTATION


def get_faces_info(solver):
    shape = (solver.n_faces_,)
    kwargs = {
        "verts_idx": V(dtype=gs.ti_ivec3, shape=shape),
        "geom_idx": V(dtype=gs.ti_int, shape=shape),
    }

    if use_ndarray:
        return StructFacesInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassFacesInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassFacesInfo()


# =========================================== EdgesInfo ===========================================


@dataclasses.dataclass
class StructEdgesInfo:
    v0: V_ANNOTATION
    v1: V_ANNOTATION
    length: V_ANNOTATION


def get_edges_info(solver):
    shape = (solver.n_edges_,)
    kwargs = {
        "v0": V(dtype=gs.ti_int, shape=shape),
        "v1": V(dtype=gs.ti_int, shape=shape),
        "length": V(dtype=gs.ti_float, shape=shape),
    }

    if use_ndarray:
        return StructEdgesInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassEdgesInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassEdgesInfo()


# =========================================== FreeVertsState ===========================================


@dataclasses.dataclass
class StructFreeVertsState:
    pos: V_ANNOTATION


def get_free_verts_state(solver):
    shape = solver._batch_shape(solver.n_free_verts_)
    kwargs = {
        "pos": V(dtype=gs.ti_vec3, shape=shape),
    }

    if use_ndarray:
        return StructFreeVertsState(**kwargs)
    else:

        @ti.data_oriented
        class ClassFreeVertsState:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassFreeVertsState()


# =========================================== FixedVertsState ===========================================


@dataclasses.dataclass
class StructFixedVertsState:
    pos: V_ANNOTATION


def get_fixed_verts_state(solver):
    shape = solver.n_fixed_verts_
    kwargs = {
        "pos": V(dtype=gs.ti_vec3, shape=shape),
    }

    if use_ndarray:
        return StructFixedVertsState(**kwargs)
    else:

        @ti.data_oriented
        class ClassFixedVertsState:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassFixedVertsState()


# =========================================== VvertsInfo ===========================================


@dataclasses.dataclass
class StructVvertsInfo:
    init_pos: V_ANNOTATION
    init_vnormal: V_ANNOTATION
    vgeom_idx: V_ANNOTATION


def get_vverts_info(solver):
    shape = (solver.n_vverts_,)
    kwargs = {
        "init_pos": V(dtype=gs.ti_vec3, shape=shape),
        "init_vnormal": V(dtype=gs.ti_vec3, shape=shape),
        "vgeom_idx": V(dtype=gs.ti_int, shape=shape),
    }

    if use_ndarray:
        return StructVvertsInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassVvertsInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassVvertsInfo()


# =========================================== VfacesInfo ===========================================


@dataclasses.dataclass
class StructVfacesInfo:
    vverts_idx: V_ANNOTATION
    vgeom_idx: V_ANNOTATION


def get_vfaces_info(solver):
    shape = (solver.n_vfaces_,)
    kwargs = {
        "vverts_idx": V(dtype=gs.ti_ivec3, shape=shape),
        "vgeom_idx": V(dtype=gs.ti_int, shape=shape),
    }

    if use_ndarray:
        return StructVfacesInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassVfacesInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassVfacesInfo()


# =========================================== VgeomsInfo ===========================================


@dataclasses.dataclass
class StructVgeomsInfo:
    pos: V_ANNOTATION
    quat: V_ANNOTATION
    link_idx: V_ANNOTATION
    vvert_num: V_ANNOTATION
    vvert_start: V_ANNOTATION
    vvert_end: V_ANNOTATION
    vface_num: V_ANNOTATION
    vface_start: V_ANNOTATION
    vface_end: V_ANNOTATION


def get_vgeoms_info(solver):
    shape = (solver.n_vgeoms_,)
    kwargs = {
        "pos": V(dtype=gs.ti_vec3, shape=shape),
        "quat": V(dtype=gs.ti_vec4, shape=shape),
        "link_idx": V(dtype=gs.ti_int, shape=shape),
        "vvert_num": V(dtype=gs.ti_int, shape=shape),
        "vvert_start": V(dtype=gs.ti_int, shape=shape),
        "vvert_end": V(dtype=gs.ti_int, shape=shape),
        "vface_num": V(dtype=gs.ti_int, shape=shape),
        "vface_start": V(dtype=gs.ti_int, shape=shape),
        "vface_end": V(dtype=gs.ti_int, shape=shape),
    }

    if use_ndarray:
        return StructVgeomsInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassVgeomsInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassVgeomsInfo()


# =========================================== VgeomsState ===========================================


@dataclasses.dataclass
class StructVgeomsState:
    pos: V_ANNOTATION
    quat: V_ANNOTATION


def get_vgeoms_state(solver):
    shape = solver._batch_shape(solver.n_vgeoms_)
    kwargs = {
        "pos": V(dtype=gs.ti_vec3, shape=shape),
        "quat": V(dtype=gs.ti_vec4, shape=shape),
    }

    if use_ndarray:
        return StructVgeomsState(**kwargs)
    else:

        @ti.data_oriented
        class ClassVgeomsState:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassVgeomsState()


# =========================================== EqualitiesInfo ===========================================


@dataclasses.dataclass
class StructEqualitiesInfo:
    eq_obj1id: V_ANNOTATION
    eq_obj2id: V_ANNOTATION
    eq_data: V_ANNOTATION
    eq_type: V_ANNOTATION
    sol_params: V_ANNOTATION


def get_equalities_info(solver):
    shape = solver._batch_shape(solver.n_equalities_candidate)
    kwargs = {
        "eq_obj1id": V(dtype=gs.ti_int, shape=shape),
        "eq_obj2id": V(dtype=gs.ti_int, shape=shape),
        "eq_data": V(dtype=gs.ti_vec11, shape=shape),
        "eq_type": V(dtype=gs.ti_int, shape=shape),
        "sol_params": V(dtype=gs.ti_vec7, shape=shape),
    }

    if use_ndarray:
        return StructEqualitiesInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassEqualitiesInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassEqualitiesInfo()


# =========================================== EntitiesInfo ===========================================


@dataclasses.dataclass
class StructEntitiesInfo:
    dof_start: V_ANNOTATION
    dof_end: V_ANNOTATION
    n_dofs: V_ANNOTATION
    link_start: V_ANNOTATION
    link_end: V_ANNOTATION
    n_links: V_ANNOTATION
    geom_start: V_ANNOTATION
    geom_end: V_ANNOTATION
    n_geoms: V_ANNOTATION
    gravity_compensation: V_ANNOTATION


def get_entities_info(solver):
    shape = solver.n_entities_
    kwargs = {
        "dof_start": V(dtype=gs.ti_int, shape=shape),
        "dof_end": V(dtype=gs.ti_int, shape=shape),
        "n_dofs": V(dtype=gs.ti_int, shape=shape),
        "link_start": V(dtype=gs.ti_int, shape=shape),
        "link_end": V(dtype=gs.ti_int, shape=shape),
        "n_links": V(dtype=gs.ti_int, shape=shape),
        "geom_start": V(dtype=gs.ti_int, shape=shape),
        "geom_end": V(dtype=gs.ti_int, shape=shape),
        "n_geoms": V(dtype=gs.ti_int, shape=shape),
        "gravity_compensation": V(dtype=gs.ti_float, shape=shape),
    }

    if use_ndarray:
        return StructEntitiesInfo(**kwargs)
    else:

        @ti.data_oriented
        class ClassEntitiesInfo:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassEntitiesInfo()


# =========================================== EntitiesState ===========================================


@dataclasses.dataclass
class StructEntitiesState:
    hibernated: V_ANNOTATION


def get_entities_state(solver):
    shape = solver._batch_shape(solver.n_entities_)
    kwargs = {
        "hibernated": V(dtype=gs.ti_int, shape=shape),
    }

    if use_ndarray:
        return StructEntitiesState(**kwargs)
    else:

        @ti.data_oriented
        class ClassEntitiesState:
            def __init__(self):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ClassEntitiesState()


# =========================================== DataManager ===========================================


@ti.data_oriented
class DataManager:
    def __init__(self, solver):
        # self.doughs = {}
        self.rigid_global_info = get_rigid_global_info(solver)
        self.dofs_info = get_dofs_info(solver)
        self.dofs_state = get_dofs_state(solver)
        self.links_info = get_links_info(solver)
        self.links_state = get_links_state(solver)
        self.joints_info = get_joints_info(solver)
        self.joints_state = get_joints_state(solver)
        self.geoms_info = get_geoms_info(solver)
        self.geoms_state = get_geoms_state(solver)

        self.verts_info = get_verts_info(solver)
        self.faces_info = get_faces_info(solver)
        self.edges_info = get_edges_info(solver)

        self.free_verts_state = get_free_verts_state(solver)
        self.fixed_verts_state = get_fixed_verts_state(solver)

        self.vverts_info = get_vverts_info(solver)
        self.vfaces_info = get_vfaces_info(solver)

        self.vgeoms_info = get_vgeoms_info(solver)
        self.vgeoms_state = get_vgeoms_state(solver)

        self.equalities_info = get_equalities_info(solver)

        self.entities_info = get_entities_info(solver)
        self.entities_state = get_entities_state(solver)


# we will use struct for DofsState and DofsInfo after Hugh adds array_struct feature to taichi
DofsState = ti.template() if not use_ndarray else StructDofsState
DofsInfo = ti.template() if not use_ndarray else StructDofsInfo
GeomsState = ti.template() if not use_ndarray else StructGeomsState
GeomsInfo = ti.template() if not use_ndarray else StructGeomsInfo
GeomsInitAABB = ti.template() if not use_ndarray else ti.types.ndarray()
LinksState = ti.template() if not use_ndarray else StructLinksState
LinksInfo = ti.template() if not use_ndarray else StructLinksInfo
JointsInfo = ti.template() if not use_ndarray else StructJointsInfo
JointsState = ti.template() if not use_ndarray else StructJointsState
FreeVertsState = ti.template() if not use_ndarray else StructFreeVertsState
FixedVertsState = ti.template() if not use_ndarray else StructFixedVertsState
VertsInfo = ti.template() if not use_ndarray else StructVertsInfo
EdgesInfo = ti.template() if not use_ndarray else StructEdgesInfo
FacesInfo = ti.template() if not use_ndarray else StructFacesInfo
VVertsInfo = ti.template() if not use_ndarray else StructVvertsInfo
VFacesInfo = ti.template() if not use_ndarray else StructVfacesInfo
VGeomsInfo = ti.template() if not use_ndarray else StructVgeomsInfo
VGeomsState = ti.template() if not use_ndarray else StructVgeomsState
EntitiesState = ti.template() if not use_ndarray else StructEntitiesState
EntitiesInfo = ti.template() if not use_ndarray else StructEntitiesInfo
EqualitiesInfo = ti.template() if not use_ndarray else StructEqualitiesInfo
RigidGlobalInfo = ti.template() if not use_ndarray else StructRigidGlobalInfo
ColliderState = ti.template() if not use_ndarray else StructColliderState
ColliderInfo = ti.template() if not use_ndarray else StructColliderInfo
