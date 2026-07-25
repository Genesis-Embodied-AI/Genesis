from typing import NamedTuple

import numpy as np

import genesis as gs
import genesis.utils.geom as gu


class GeomSphereProxy(NamedTuple):
    """Conservative sphere cover of a single geom, expressed in its link frame."""

    pos: np.ndarray
    radius: np.ndarray


# Coverage targets and sphere candidates are deterministically subsampled to these budgets so the greedy set cover
# stays a few 1e6 pairwise distances per geom. The final certificate runs against the FULL target set, so
# subsampling can only cost extra spheres or padding, never coverage.
_MAX_CANDIDATES = 512
_MAX_TARGETS = 2048


def _subsample(points, budget):
    if len(points) <= budget:
        return points
    step = -(-len(points) // budget)
    return points[::step]


def _sdf_node_positions_mesh(geom, nodes_idx):
    T_sdf_to_mesh = np.linalg.inv(geom.T_mesh_to_sdf)
    return nodes_idx @ T_sdf_to_mesh[:3, :3].T + T_sdf_to_mesh[:3, 3]


def _greedy_sphere_cover(candidates_pos, candidates_radius, targets, delta_cover, n_max_spheres):
    """Greedy max-coverage set cover: pick the candidate covering the most yet-uncovered targets.

    A target t is covered by sphere (c, r) iff ||t - c|| <= r - delta_cover, so that after inflating every target
    into a delta_cover-ball (the sampling density bound), the sphere union still contains it.
    """
    dist = np.linalg.norm(targets[None, :, :] - candidates_pos[:, None, :], axis=-1)
    is_covering = dist <= (candidates_radius[:, None] - delta_cover)

    spheres_idx = []
    is_uncovered = np.ones(len(targets), dtype=np.bool_)
    while is_uncovered.any() and len(spheres_idx) < n_max_spheres:
        gains = (is_covering & is_uncovered).sum(axis=1)
        i_best = np.argmax(gains)
        if gains[i_best] == 0:
            break
        spheres_idx.append(i_best)
        is_uncovered &= ~is_covering[i_best]
    return spheres_idx, not is_uncovered.any()


def build_geom_sphere_proxy(geom, n_max_spheres, sphere_pad):
    """Build a conservative sphere cover of a geom's collision volume, in the link frame.

    The union of the returned spheres contains the geom volume: sphere centers are interior signed distance field
    (SDF) grid nodes (radius = interior depth + padding) or surface vertices for thin geoms, greedily selected to
    cover every interior node and surface vertex, then certified against the full target set. On certification
    failure the padding is doubled and the fill retried, so the contract holds for any input at the cost of a
    looser proxy.

    Parameters
    ----------
    geom : RigidGeom
        The collision geom to cover.
    n_max_spheres : int
        Budget of spheres for this geom.
    sphere_pad : float
        Base padding added to every sphere radius [m].

    Returns
    -------
    proxy : GeomSphereProxy
        Sphere centers (link frame) and radii.
    """
    # An exact single sphere beats any fill for spherical geoms.
    if geom.type == gs.GEOM_TYPE.SPHERE:
        pos_geom = np.zeros((1, 3), dtype=gs.np_float)
        radius = np.array([geom.data[0]], dtype=gs.np_float)
    elif geom.type == gs.GEOM_TYPE.PLANE:
        gs.raise_exception("Sphere proxies cannot cover unbounded plane geoms.")
    else:
        pos_geom, radius = _fill_from_sdf(geom, n_max_spheres, sphere_pad)

    pos_link = gu.transform_by_trans_quat(pos_geom, geom.init_pos, geom.init_quat)
    return GeomSphereProxy(pos=pos_link.astype(gs.np_float), radius=radius.astype(gs.np_float))


def _fill_from_sdf(geom, n_max_spheres, sphere_pad):
    sdf_val = geom.sdf_val
    # Half grid-cell diagonal: no point of the geom volume is farther than this from some coverage target.
    delta_cover = 0.5 * np.sqrt(3.0) * np.max(geom.sdf_cell_size)

    nodes_idx = np.argwhere(sdf_val < 0.0)
    interior_pos = _sdf_node_positions_mesh(geom, nodes_idx)
    interior_depth = -sdf_val[tuple(nodes_idx.T)] if len(nodes_idx) else np.zeros(0, dtype=gs.np_float)
    surface_pos = np.asarray(geom.init_verts, dtype=gs.np_float)

    targets = np.concatenate([interior_pos, surface_pos], axis=0)
    targets_sub = _subsample(targets, _MAX_TARGETS)

    # Deepest-first candidate subsampling keeps the large inscribed spheres available under the budget.
    order = np.argsort(-interior_depth, kind="stable")
    candidates_pos = np.concatenate([interior_pos[order], surface_pos], axis=0)
    candidates_depth = np.concatenate([interior_depth[order], np.zeros(len(surface_pos), dtype=gs.np_float)])
    candidates_pos = _subsample(candidates_pos, _MAX_CANDIDATES)
    candidates_depth = _subsample(candidates_depth, _MAX_CANDIDATES)

    pad = sphere_pad + delta_cover
    while True:
        candidates_radius = candidates_depth + pad
        spheres_idx, is_covered = _greedy_sphere_cover(
            candidates_pos, candidates_radius, targets_sub, delta_cover, n_max_spheres
        )
        if is_covered and spheres_idx:
            # Certificate against the FULL target set: every point of the geom volume lies within delta_cover of
            # some target, and every target sits strictly inside a sphere, so the union contains the volume.
            pos, radius = candidates_pos[spheres_idx], candidates_radius[spheres_idx]
            dist = np.linalg.norm(targets[None, :, :] - pos[:, None, :], axis=-1)
            if (dist <= (radius[:, None] - delta_cover)).any(axis=0).all():
                return pos, radius
        pad *= 2.0
        gs.logger.debug(f"Sphere proxy of geom {geom.idx} not certified; retrying with padding {pad:.4f}.")
