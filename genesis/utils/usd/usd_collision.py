"""Translate USD collision filtering (CollisionGroup / FilteredPairsAPI) into contype/conaffinity."""

from typing import Dict, List, Set

from pxr import Usd, UsdPhysics

import genesis as gs
from genesis.utils.collision import solve_contype_conaffinity


def _geoms_under(cg_infos: List[Dict], path: str) -> List[int]:
    """Indices of collision g_infos whose source prim is `path` or a descendant of it."""
    prefix = path.rstrip("/") + "/"
    return [
        i for i, g in enumerate(cg_infos) if g.get("prim_path") == path or g.get("prim_path", "").startswith(prefix)
    ]


def _collision_group_records(stage: Usd.Stage):
    """Gather CollisionGroup records, unifying groups that share a non-empty mergeGroupName.

    Returns (canonical_of, membership_queries, filtered_of):
    - canonical_of: group prim path -> canonical group id (mergeGroupName if set, else the path)
    - membership_queries: list of (canonical_id, UsdCollectionMembershipQuery)
    - filtered_of: canonical id -> set of canonical ids it does not collide with (symmetric use)
    """
    canonical_of: Dict[str, str] = {}
    raw = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.CollisionGroup):
            continue
        group = UsdPhysics.CollisionGroup(prim)
        path = str(prim.GetPath())
        merge_name = group.GetMergeGroupNameAttr().Get() or ""
        canonical_of[path] = merge_name if merge_name else path
        raw.append((path, group))

    membership_queries = []
    filtered_of: Dict[str, Set[str]] = {}
    for path, group in raw:
        canonical = canonical_of[path]
        if group.GetInvertFilteredGroupsAttr().Get():
            # Inverted semantics (collide ONLY with the listed groups) are uncommon and not modeled;
            # skip this group's filtering rather than risk disabling the wrong pairs.
            gs.logger.warning(
                f"USD CollisionGroup '{path}' uses invertFilteredGroups, which is not supported; "
                "its collision filtering is ignored."
            )
            continue
        membership_queries.append((canonical, group.GetCollidersCollectionAPI().ComputeMembershipQuery()))
        filtered = filtered_of.setdefault(canonical, set())
        for target in group.GetFilteredGroupsRel().GetTargets():
            filtered.add(canonical_of.get(str(target), str(target)))
    return canonical_of, membership_queries, filtered_of


def _is_external(external_colliders: Set[str], path: str) -> bool:
    """Whether `path` is, or contains, a collider that belongs to another entity."""
    prefix = path.rstrip("/") + "/"
    return any(p == path or p.startswith(prefix) for p in external_colliders)


def apply_collision_filtering(context, cg_infos: List[Dict]):
    """
    Set contype/conaffinity on collision g_infos from USD collision filtering.

    Honors ``UsdPhysics.CollisionGroup`` (collider membership + ``filteredGroups`` + ``mergeGroupName``)
    and per-prim ``UsdPhysics.FilteredPairsAPI``, for the geoms in ``cg_infos`` (a single entity). The
    resulting "must not collide" pairs are realized as contype/conaffinity bitmasks by the shared
    solver; if they cannot be expressed as bitmasks, a warning is logged and defaults are kept.

    Filtering that spans entities is not expressible per entity, so it is detected against the stage's
    full collider set and reported once via ``context.note_unsupported_cross_entity_filtering()``.

    ``cg_infos`` must each carry a ``prim_path``; it is consumed (popped) here.
    """
    if not cg_infos:
        return

    stage: Usd.Stage = context.stage
    n = len(cg_infos)
    invalid_pairs: Set[frozenset] = set()

    # Colliders that belong to other entities (add_stage splits the stage): used to detect filtering
    # relationships this per-entity pass cannot express.
    entity_paths = {g["prim_path"] for g in cg_infos}
    stage_collider_paths = {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)}
    external_colliders = stage_collider_paths - entity_paths
    cross_entity = False

    # 1) CollisionGroup filtering: members of groups that filter each other must not collide. Membership
    # is resolved over every collider in the stage (this entity's geoms are a subset) so cross-entity
    # relationships are visible.
    canonical_of, membership_queries, filtered_of = _collision_group_records(stage)
    if filtered_of:
        group_members: Dict[str, Set[str]] = {}
        for canonical, query in membership_queries:
            group_members.setdefault(canonical, set()).update(
                p for p in stage_collider_paths if query.IsPathIncluded(p)
            )
        geom_groups: List[Set[str]] = [
            {canonical for canonical, members in group_members.items() if g["prim_path"] in members} for g in cg_infos
        ]
        for i in range(n):
            for j in range(i + 1, n):
                if any(
                    (gb in filtered_of.get(ga, ())) or (ga in filtered_of.get(gb, ()))
                    for ga in geom_groups[i]
                    for gb in geom_groups[j]
                ):
                    invalid_pairs.add(frozenset((i, j)))
        if external_colliders:
            for ca, partners in filtered_of.items():
                for cb in partners:
                    members = group_members.get(ca, set()) | group_members.get(cb, set())
                    if (members & entity_paths) and (members & external_colliders):
                        cross_entity = True

    # 2) FilteredPairsAPI: explicit prim-subtree pairs that must not collide.
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            continue
        geoms_a = _geoms_under(cg_infos, str(prim.GetPath()))
        if not geoms_a:
            continue
        for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
            target_geoms = _geoms_under(cg_infos, str(target))
            if not target_geoms and _is_external(external_colliders, str(target)):
                cross_entity = True  # target collider lives in another entity
            for i in geoms_a:
                for j in target_geoms:
                    if i != j:
                        invalid_pairs.add(frozenset((i, j)))

    if invalid_pairs:
        masks = solve_contype_conaffinity(n, invalid_pairs)
        if masks is None:
            gs.logger.warning(
                "USD collision filtering cannot be expressed with 'contype'/'conaffinity' bitmasks. "
                "Keeping default (all-colliding) values."
            )
        else:
            for g_info, (contype, conaffinity) in zip(cg_infos, masks):
                g_info["contype"], g_info["conaffinity"] = contype, conaffinity

    if cross_entity:
        context.note_unsupported_cross_entity_filtering()

    for g_info in cg_infos:
        g_info.pop("prim_path", None)
