"""System-unaware host tools shared by kinematic model owners."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import indices_to_mask

if TYPE_CHECKING:
    from genesis.engine.entities import KinematicEntity
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom, RigidVisGeom

HostArray: TypeAlias = NDArray[np.floating] | torch.Tensor
HostIndex: TypeAlias = ArrayLike | torch.Tensor | None


def balanced_variant_mapping(n_variants: int, batch_size: int) -> NDArray[np.integer]:
    """Map variants to environments using balanced contiguous blocks."""
    if batch_size >= n_variants:
        base = batch_size // n_variants
        extra = batch_size % n_variants
        sizes = np.r_[np.full(extra, base + 1), np.full(n_variants - extra, base)]
        return np.repeat(np.arange(n_variants), sizes)
    return np.arange(batch_size)


def select_links_offset(offset: HostArray, links_idx: HostIndex, envs_idx: HostIndex) -> HostArray:
    """Index a base-link forward offset with the same mask as a transposed state query."""
    if offset.ndim == 2:
        return offset[indices_to_mask(links_idx)]
    return offset[indices_to_mask(envs_idx, links_idx)]


def offset_world_shift(offset_pos: HostArray, offset_quat: HostArray, world_quat: HostArray) -> HostArray:
    """Return the world displacement contributed by a body-frame offset position."""
    authored_quat = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), world_quat)
    return gu.transform_by_quat(offset_pos, authored_quat)


def fill_base_link_geom_offsets(
    offset_pos: NDArray[np.floating],
    offset_quat: NDArray[np.floating],
    entity: "KinematicEntity",
    geoms: Sequence["RigidGeom | RigidVisGeom"],
    ranges: Sequence[tuple[int, int]] | None,
) -> None:
    """Fill per-geom forward offsets for one entity's collision or visual geoms."""
    for geom in geoms:
        if ranges is None:
            link_off_pos = geom.link.desc.offset_pos
            link_off_quat = geom.link.desc.offset_quat
            if np.allclose(link_off_pos, 0.0, atol=gs.EPS) and np.allclose(
                link_off_quat, gu.identity_quat(), atol=gs.EPS
            ):
                continue
            morph = entity._morph
        else:
            i_variant = next((i for i, (start, end) in enumerate(ranges) if start <= geom.idx < end), None)
            if i_variant is None:
                continue
            link_off_pos = entity._desc.variants[i_variant].offset_pos
            link_off_quat = entity._desc.variants[i_variant].offset_quat
            morph = entity._morph if i_variant == 0 else entity._morph_heterogeneous[i_variant - 1]
        frame_pos, frame_quat = gu.transform_pos_quat_by_trans_quat(
            geom.init_pos,
            geom.init_quat,
            link_off_pos,
            link_off_quat,
        )
        offset_frame_pos, offset_frame_quat = gu.transform_pos_quat_by_trans_quat(
            frame_pos,
            frame_quat,
            np.array(morph.offset_pos, dtype=gs.np_float),
            np.array(morph.offset_quat, dtype=gs.np_float),
        )
        offset_pos[geom.idx], offset_quat[geom.idx] = gu.inv_transform_pos_quat_by_trans_quat(
            offset_frame_pos,
            offset_frame_quat,
            frame_pos,
            frame_quat,
        )
