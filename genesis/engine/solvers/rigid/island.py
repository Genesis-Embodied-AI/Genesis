import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class


# Partition dof-carrying entities into islands: connected components of the inter-entity coupling
# graph. Edges come from ALL couplings that put two entities in the same constraint block - contacts
# and equality constraints (CONNECT/WELD on links, JOINT on joints) - so each island is an exactly
# decoupled (block-diagonal) sub-problem of the constraint solve. Union-by-min makes the labels
# independent of edge-processing order, which keeps per-island solving deterministic.


@qd.func
def func_find_root(island_state: array_class.IslandState, e, i_b):
    # Path-halving find.
    root = e
    while island_state.entity_parent[root, i_b] != root:
        island_state.entity_parent[root, i_b] = island_state.entity_parent[island_state.entity_parent[root, i_b], i_b]
        root = island_state.entity_parent[root, i_b]
    return root


@qd.func
def func_union(island_state: array_class.IslandState, a, b, i_b):
    # Union by minimum index: the root of a component is its smallest entity index, regardless of the
    # order edges are processed.
    ra = func_find_root(island_state, a, i_b)
    rb = func_find_root(island_state, b, i_b)
    if ra < rb:
        island_state.entity_parent[rb, i_b] = ra
    elif rb < ra:
        island_state.entity_parent[ra, i_b] = rb


@qd.func
def func_link_entity(links_info: array_class.LinksInfo, link, i_b, static_rigid_sim_config: qd.template()):
    link_idx = [link, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link
    return links_info.entity_idx[link_idx]


@qd.func
def func_joint_entity(
    joints_info: array_class.JointsInfo,
    entities_info: array_class.EntitiesInfo,
    joint,
    i_b,
    n_entities,
    static_rigid_sim_config: qd.template(),
):
    # JointsInfo carries no entity/link mapping, so locate the entity whose dof range owns the joint's
    # first dof. Joint equalities are rare and entity counts are small, so the linear scan is cheap.
    joint_idx = [joint, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else joint
    i_dof = joints_info.dof_start[joint_idx]
    entity = -1
    for i_e in range(n_entities):
        if entities_info.dof_start[i_e] <= i_dof < entities_info.dof_end[i_e]:
            entity = i_e
    return entity


@qd.func
def func_equality_entities(
    equalities_info: array_class.EqualitiesInfo,
    joints_info: array_class.JointsInfo,
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    i_eq,
    i_b,
    n_entities,
    static_rigid_sim_config: qd.template(),
):
    # Map an equality constraint to the pair of entities it couples. CONNECT/WELD reference links;
    # JOINT references joints.
    obj1 = equalities_info.eq_obj1id[i_eq, i_b]
    obj2 = equalities_info.eq_obj2id[i_eq, i_b]
    eq_type = equalities_info.eq_type[i_eq, i_b]
    ea = -1
    eb = -1
    if eq_type == gs.EQUALITY_TYPE.JOINT:
        ea = func_joint_entity(joints_info, entities_info, obj1, i_b, n_entities, static_rigid_sim_config)
        eb = func_joint_entity(joints_info, entities_info, obj2, i_b, n_entities, static_rigid_sim_config)
    else:
        ea = func_link_entity(links_info, obj1, i_b, static_rigid_sim_config)
        eb = func_link_entity(links_info, obj2, i_b, static_rigid_sim_config)
    return ea, eb


@qd.func
def func_constraint_island(
    constraint_state: array_class.ConstraintState,
    island_state: array_class.IslandState,
    i_c,
    i_b,
    n_dofs,
    EPS,
    static_rigid_sim_config: qd.template(),
):
    # A constraint couples dofs of a single island, so its island is that of its first nonzero
    # Jacobian dof. With the sparse Jacobian representation that dof is jac_dofs_idx[i_c, 0] directly
    # (O(1)); otherwise scan the dense Jacobian row for the first nonzero entry (O(n_dofs)).
    i_island = -1
    if qd.static(static_rigid_sim_config.sparse_solve):
        if constraint_state.jac_n_dofs[i_c, i_b] > 0:
            i_island = island_state.dof_island[constraint_state.jac_dofs_idx[i_c, 0, i_b], i_b]
    else:
        for i_d in range(n_dofs):
            if qd.abs(constraint_state.jac[i_c, i_d, i_b]) > EPS:
                i_island = island_state.dof_island[i_d, i_b]
                break
    return i_island


@qd.kernel
def kernel_group_constraints_by_island(
    island_state: array_class.IslandState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    # Run AFTER constraint assembly: group the assembled constraints by island into contiguous ranges
    # in constraint_id, so the per-island solve can iterate its own constraints. Built as a separate
    # post-assembly pass to avoid reworking the intricate (cooperative/atomic) assembly kernels.
    EPS = rigid_global_info.EPS[None]
    n_dofs = constraint_state.jac.shape[1]
    _B = constraint_state.n_constraints.shape[0]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        n_islands = island_state.n_islands[i_b]
        for i_island in range(n_islands):
            island_state.island_constraint.n[i_island, i_b] = 0

        n_con = constraint_state.n_constraints[i_b]
        for i_c in range(n_con):
            i_island = func_constraint_island(
                constraint_state, island_state, i_c, i_b, n_dofs, EPS, static_rigid_sim_config
            )
            if i_island >= 0:
                island_state.island_constraint.n[i_island, i_b] = island_state.island_constraint.n[i_island, i_b] + 1

        con_list_start = 0
        for i_island in range(n_islands):
            island_state.island_constraint.start[i_island, i_b] = con_list_start
            island_state.island_constraint.curr[i_island, i_b] = con_list_start
            con_list_start = con_list_start + island_state.island_constraint.n[i_island, i_b]

        for i_c in range(n_con):
            i_island = func_constraint_island(
                constraint_state, island_state, i_c, i_b, n_dofs, EPS, static_rigid_sim_config
            )
            if i_island >= 0:
                island_state.constraint_id[island_state.island_constraint.curr[i_island, i_b], i_b] = i_c
                island_state.island_constraint.curr[i_island, i_b] = (
                    island_state.island_constraint.curr[i_island, i_b] + 1
                )


@qd.kernel
def kernel_build_islands(
    entities_info: array_class.EntitiesInfo,
    entities_state: array_class.EntitiesState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    island_state: array_class.IslandState,
    static_rigid_sim_config: qd.template(),
):
    n_entities = entities_info.n_dofs.shape[0]
    _B = island_state.entity_island.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        # Init: every dof-entity is its own component; fixed (0-dof) entities are not partitioned.
        for i_e in range(n_entities):
            island_state.entity_parent[i_e, i_b] = i_e
            island_state.entity_island[i_e, i_b] = -1
            island_state.island_entity.n[i_e, i_b] = 0
            island_state.island_dof.n[i_e, i_b] = 0
            island_state.island_contact.n[i_e, i_b] = 0

        # Edges from contacts (read through contact_sort_idx so pruning/sorting is honored). Only
        # couple two dof-entities; a contact against a fixed body adds no edge.
        for i_c in range(collider_state.n_contacts[i_b]):
            i_col = collider_state.contact_sort_idx[i_c, i_b]
            link_a = collider_state.contact_data.link_a[i_col, i_b]
            link_b = collider_state.contact_data.link_b[i_col, i_b]
            ea = func_link_entity(links_info, link_a, i_b, static_rigid_sim_config)
            eb = func_link_entity(links_info, link_b, i_b, static_rigid_sim_config)
            if entities_info.n_dofs[ea] > 0 and entities_info.n_dofs[eb] > 0 and ea != eb:
                func_union(island_state, ea, eb, i_b)

        # Edges from equality constraints (model + dynamically registered welds).
        for i_eq in range(constraint_state.qd_n_equalities[i_b]):
            ea, eb = func_equality_entities(
                equalities_info,
                joints_info,
                entities_info,
                links_info,
                i_eq,
                i_b,
                n_entities,
                static_rigid_sim_config,
            )
            if ea >= 0 and eb >= 0 and entities_info.n_dofs[ea] > 0 and entities_info.n_dofs[eb] > 0 and ea != eb:
                func_union(island_state, ea, eb, i_b)

        # Hibernated islands: re-union along the daisy chain so a sleeping group (which generates no live
        # contacts to union it) stays one island across steps, matching the partition the wakeup walks.
        if qd.static(static_rigid_sim_config.use_hibernation):
            for i_e in range(n_entities):
                next_e = island_state.entity_idx_to_next_entity_idx_in_hibernated_island[i_e, i_b]
                if 0 <= next_e < n_entities and next_e != i_e:
                    if entities_info.n_dofs[i_e] > 0 and entities_info.n_dofs[next_e] > 0:
                        func_union(island_state, i_e, next_e, i_b)

        # Label components: each root (min entity index of its component) gets the next island id.
        n_islands = 0
        for i_e in range(n_entities):
            if entities_info.n_dofs[i_e] > 0 and func_find_root(island_state, i_e, i_b) == i_e:
                island_state.entity_island[i_e, i_b] = n_islands
                n_islands = n_islands + 1
        island_state.n_islands[i_b] = n_islands

        # Propagate the root's label to every dof-entity in its component.
        for i_e in range(n_entities):
            if entities_info.n_dofs[i_e] > 0:
                root = func_find_root(island_state, i_e, i_b)
                island_state.entity_island[i_e, i_b] = island_state.entity_island[root, i_b]

        # Mark islands whose every dof-entity is asleep (read by the hibernation decision on the next step to
        # skip already-sleeping islands). An island is hibernated unless it has at least one awake dof-entity.
        if qd.static(static_rigid_sim_config.use_hibernation):
            for i_island in range(n_islands):
                island_state.island_is_hibernated[i_island, i_b] = 1
            for i_e in range(n_entities):
                i_island = island_state.entity_island[i_e, i_b]
                if i_island >= 0 and not entities_state.is_hibernated[i_e, i_b]:
                    island_state.island_is_hibernated[i_island, i_b] = 0

        # Build the per-island entity list (island -> entity-idx ranges).
        for i_e in range(n_entities):
            i_island = island_state.entity_island[i_e, i_b]
            if i_island >= 0:
                island_state.island_entity.n[i_island, i_b] = island_state.island_entity.n[i_island, i_b] + 1
        entity_list_start = 0
        for i_island in range(n_islands):
            island_state.island_entity.start[i_island, i_b] = entity_list_start
            island_state.island_entity.curr[i_island, i_b] = entity_list_start
            entity_list_start = entity_list_start + island_state.island_entity.n[i_island, i_b]
        for i_e in range(n_entities):
            i_island = island_state.entity_island[i_e, i_b]
            if i_island >= 0:
                island_state.entity_id[island_state.island_entity.curr[i_island, i_b], i_b] = i_e
                island_state.island_entity.curr[i_island, i_b] = island_state.island_entity.curr[i_island, i_b] + 1

        # Build the per-island dof list (the block-gather map: local dof -> global dof). dof_id is
        # grouped by island; for the monolith (one island over all entities in order) it is the
        # identity permutation.
        for i_e in range(n_entities):
            i_island = island_state.entity_island[i_e, i_b]
            if i_island >= 0:
                island_state.island_dof.n[i_island, i_b] = (
                    island_state.island_dof.n[i_island, i_b] + entities_info.n_dofs[i_e]
                )
        # dof_list_start lays the block-gather dof_id contiguously; tile_start packs each island's dense
        # n_i x n_i Hessian tile contiguously into the shared n_dofs^2 buffer (offset = running Sum n_j^2).
        dof_list_start = 0
        tile_start = 0
        for i_island in range(n_islands):
            island_state.island_dof.start[i_island, i_b] = dof_list_start
            island_state.island_dof.curr[i_island, i_b] = dof_list_start
            island_state.island_tile_start[i_island, i_b] = tile_start
            n_isl_dofs = island_state.island_dof.n[i_island, i_b]
            dof_list_start = dof_list_start + n_isl_dofs
            tile_start = tile_start + n_isl_dofs * n_isl_dofs
        for i_e in range(n_entities):
            i_island = island_state.entity_island[i_e, i_b]
            if i_island >= 0:
                for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    local = island_state.island_dof.curr[i_island, i_b] - island_state.island_dof.start[i_island, i_b]
                    island_state.dof_id[island_state.island_dof.curr[i_island, i_b], i_b] = i_d
                    island_state.dof_iisland[i_d, i_b] = local
                    island_state.dof_island[i_d, i_b] = i_island
                    island_state.island_dof.curr[i_island, i_b] = island_state.island_dof.curr[i_island, i_b] + 1

        # Build the per-island contact list (island -> contact ranges in contact_id). A contact
        # belongs to the island of its dof-carrying endpoint (both endpoints share an island when both
        # carry dofs, since the contact unioned them; otherwise one side is a fixed body).
        for i_c in range(collider_state.n_contacts[i_b]):
            i_col = collider_state.contact_sort_idx[i_c, i_b]
            ea = func_link_entity(
                links_info, collider_state.contact_data.link_a[i_col, i_b], i_b, static_rigid_sim_config
            )
            eb = func_link_entity(
                links_info, collider_state.contact_data.link_b[i_col, i_b], i_b, static_rigid_sim_config
            )
            i_island = island_state.entity_island[ea, i_b]
            if i_island < 0:
                i_island = island_state.entity_island[eb, i_b]
            if i_island >= 0:
                island_state.island_contact.n[i_island, i_b] = island_state.island_contact.n[i_island, i_b] + 1
        contact_list_start = 0
        for i_island in range(n_islands):
            island_state.island_contact.start[i_island, i_b] = contact_list_start
            island_state.island_contact.curr[i_island, i_b] = contact_list_start
            contact_list_start = contact_list_start + island_state.island_contact.n[i_island, i_b]
        for i_c in range(collider_state.n_contacts[i_b]):
            i_col = collider_state.contact_sort_idx[i_c, i_b]
            ea = func_link_entity(
                links_info, collider_state.contact_data.link_a[i_col, i_b], i_b, static_rigid_sim_config
            )
            eb = func_link_entity(
                links_info, collider_state.contact_data.link_b[i_col, i_b], i_b, static_rigid_sim_config
            )
            i_island = island_state.entity_island[ea, i_b]
            if i_island < 0:
                i_island = island_state.entity_island[eb, i_b]
            if i_island >= 0:
                island_state.contact_id[island_state.island_contact.curr[i_island, i_b], i_b] = i_col
                island_state.island_contact.curr[i_island, i_b] = island_state.island_contact.curr[i_island, i_b] + 1


@qd.kernel
def kernel_build_island_worklist(island_state: array_class.IslandState):
    """Flatten the per-env islands into a single (env, island) work-list for the decomposed arm's
    warp-cooperative per-island dispatch.

    Runs serially in one thread (cheap: one write per island) so the running cursor stays consistent; the
    dispatch does not depend on ordering. Resets the atomic steal cursor for the step.
    """
    qd.loop_config(serialize=True)
    for _ in range(1):
        _B = island_state.n_islands.shape[0]
        k = 0
        for i_b in range(_B):
            for i_island in range(island_state.n_islands[i_b]):
                island_state.work_i_b[k] = i_b
                island_state.work_i_island[k] = i_island
                k = k + 1
        island_state.work_size[0] = k
        island_state.work_counter[0] = 0
