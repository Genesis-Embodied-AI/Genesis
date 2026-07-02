import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class


# Partition LINKS into islands: connected components of the coupling graph whose edges are (1) kinematic - every link
# to its parent, so a floating-base subtree collapses to one component - plus (2) contacts and (3) equality constraints
# (CONNECT/WELD on links, JOINT on joints). A single Genesis entity holding several free bodies (common in MJCF) thus
# splits into one island per free body, while an articulated body's links stay one island. Each island is an exactly
# decoupled (block-diagonal) sub-problem of the constraint solve. Union-by-min makes the labels independent of
# edge-processing order, which keeps per-island solving deterministic.


@qd.func
def func_find_root(island_state: array_class.IslandState, i_l, i_b):
    # Path-halving find (over links).
    root = i_l
    while island_state.links_parent_idx[root, i_b] != root:
        island_state.links_parent_idx[root, i_b] = island_state.links_parent_idx[
            island_state.links_parent_idx[root, i_b], i_b
        ]
        root = island_state.links_parent_idx[root, i_b]
    return root


@qd.func
def func_union(island_state: array_class.IslandState, i_la, i_lb, i_b):
    # Union by minimum index: the root of a component is its smallest link index, regardless of the order edges are
    # processed.
    root_a = func_find_root(island_state, i_la, i_b)
    root_b = func_find_root(island_state, i_lb, i_b)
    if root_a < root_b:
        island_state.links_parent_idx[root_b, i_b] = root_a
    elif root_b < root_a:
        island_state.links_parent_idx[root_a, i_b] = root_b


@qd.func
def func_joint_link(
    joints_info: array_class.JointsInfo,
    links_info: array_class.LinksInfo,
    i_joint,
    i_b,
    n_links,
    static_rigid_sim_config: qd.template(),
):
    # JointsInfo carries no link mapping, so locate the link whose dof range owns the joint's first dof. Joint
    # equalities are rare and link counts are small, so the linear scan is cheap.
    joint_idx = [i_joint, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_joint
    i_dof = joints_info.dof_start[joint_idx]
    link = -1
    for i_l in range(n_links):
        link_idx = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
        if links_info.dof_start[link_idx] <= i_dof < links_info.dof_end[link_idx]:
            link = i_l
            break
    return link


@qd.func
def func_equality_links(
    equalities_info: array_class.EqualitiesInfo,
    joints_info: array_class.JointsInfo,
    links_info: array_class.LinksInfo,
    i_eq,
    i_b,
    n_links,
    static_rigid_sim_config: qd.template(),
):
    # Map an equality constraint to the pair of links it couples. CONNECT/WELD reference links; JOINT references joints.
    obj1 = equalities_info.eq_obj1id[i_eq, i_b]
    obj2 = equalities_info.eq_obj2id[i_eq, i_b]
    eq_type = equalities_info.eq_type[i_eq, i_b]
    la = -1
    lb = -1
    if eq_type == gs.EQUALITY_TYPE.JOINT:
        la = func_joint_link(joints_info, links_info, obj1, i_b, n_links, static_rigid_sim_config)
        lb = func_joint_link(joints_info, links_info, obj2, i_b, n_links, static_rigid_sim_config)
    else:
        la = obj1
        lb = obj2
    return la, lb


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
    # A constraint couples dofs of a single island, so its island is that of its first nonzero Jacobian dof. With the
    # sparse Jacobian representation that dof is jac_dofs_idx[i_c, 0] directly (O(1)); otherwise scan the dense Jacobian
    # row for the first nonzero entry (O(n_dofs)).
    i_island = -1
    if qd.static(static_rigid_sim_config.sparse_solve):
        if constraint_state.jac_n_dofs[i_c, i_b] > 0:
            i_island = island_state.dofs_island_idx[constraint_state.jac_dofs_idx[i_c, 0, i_b], i_b]
    else:
        for i_d in range(n_dofs):
            if qd.abs(constraint_state.jac[i_c, i_d, i_b]) > EPS:
                i_island = island_state.dofs_island_idx[i_d, i_b]
                break
    return i_island


@qd.func
def func_group_constraints_by_island(
    i_b,
    island_state: array_class.IslandState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    # Group one env's constraints into contiguous per-island ranges in constraint_id, so the per-island solve can
    # iterate its own constraints. Reads constraint_island_idx (resolved by the parallel pass in the caller). The
    # island label is read in O(1), and the fill walks constraints in index order, so each island's constraint list
    # stays order-deterministic.
    n_islands = island_state.n_islands[i_b]
    n_con = constraint_state.n_constraints[i_b]
    if n_islands == 1:
        # A single island spans the whole env, so every constraint belongs to island 0 in index order: the grouping is
        # the identity. Skip the per-constraint island lookup and the two-pass bucketing (the caller also skips the
        # per-constraint resolve pass for this env). Any constraint that touches no DOF carries jac == 0, so listing it
        # in island 0 is harmless. This is the common case for a scene whose free bodies have settled into one contact
        # component, where the per-island bookkeeping would otherwise be pure overhead.
        island_state.constraint_slices.start[0, i_b] = 0
        island_state.constraint_slices.n[0, i_b] = n_con
        island_state.constraint_slices.curr[0, i_b] = n_con
        for i_c in range(n_con):
            island_state.constraint_id[i_c, i_b] = i_c
    else:
        for i_island in range(n_islands):
            island_state.constraint_slices.n[i_island, i_b] = 0

        for i_c in range(n_con):
            i_island = island_state.constraint_island_idx[i_c, i_b]
            if i_island >= 0:
                island_state.constraint_slices.n[i_island, i_b] = island_state.constraint_slices.n[i_island, i_b] + 1

        con_list_start = 0
        for i_island in range(n_islands):
            island_state.constraint_slices.start[i_island, i_b] = con_list_start
            island_state.constraint_slices.curr[i_island, i_b] = con_list_start
            con_list_start = con_list_start + island_state.constraint_slices.n[i_island, i_b]

        for i_c in range(n_con):
            i_island = island_state.constraint_island_idx[i_c, i_b]
            if i_island >= 0:
                island_state.constraint_id[island_state.constraint_slices.curr[i_island, i_b], i_b] = i_c
                island_state.constraint_slices.curr[i_island, i_b] = (
                    island_state.constraint_slices.curr[i_island, i_b] + 1
                )


@qd.func
def func_contact_tree_slots(
    i_b,
    i_col,
    links_info: array_class.LinksInfo,
    collider_state: array_class.ColliderState,
    island_state: array_class.IslandState,
    static_rigid_sim_config: qd.template(),
):
    """Island-local tree slots (rcm_tree_pos) of a contact's two endpoints, -1 for a fixed / dof-less side."""
    link_a = collider_state.contact_data.link_a[i_col, i_b]
    link_b = collider_state.contact_data.link_b[i_col, i_b]
    i_ta = -1
    i_tb = -1
    if island_state.links_island_idx[link_a, i_b] >= 0:
        link_idx = [link_a, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link_a
        i_ta = island_state.rcm_tree_pos[links_info.root_idx[link_idx], i_b]
    if island_state.links_island_idx[link_b, i_b] >= 0:
        link_idx = [link_b, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link_b
        i_tb = island_state.rcm_tree_pos[links_info.root_idx[link_idx], i_b]
    return i_ta, i_tb


@qd.func
def func_build_islands(
    i_b,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    equalities_info: array_class.EqualitiesInfo,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    island_state: array_class.IslandState,
    static_rigid_sim_config: qd.template(),
):
    # Partition one env's links into islands (kinematic tree + contact + equality edges) via union-find, then build the
    # per-island link/dof/contact lists. Run before the Newton solve so it factors each island's block independently.
    n_links = island_state.links_island_idx.shape[0]
    # Init: every link is its own component.
    for i_l in range(n_links):
        island_state.links_parent_idx[i_l, i_b] = i_l
        island_state.links_island_idx[i_l, i_b] = -1
        island_state.link_slices.n[i_l, i_b] = 0
        island_state.dof_slices.n[i_l, i_b] = 0
        island_state.contact_slices.n[i_l, i_b] = 0

    # Kinematic edges: union every link with its parent. This collapses an articulated body (and a free body's
    # subtree) into one component, while sibling free bodies (parent_idx == -1) stay separate components - so a
    # single Genesis entity holding several free bodies splits into one island per free body.
    for i_l in range(n_links):
        link_idx = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
        i_p = links_info.parent_idx[link_idx]
        if i_p >= 0:
            func_union(island_state, i_l, i_p, i_b)

    # Mark each kinematic component that carries at least one dof as dynamic (links_island_idx[root] = -2, a
    # transient marker overwritten by the labeling pass below). A contact/equality couples two links only when
    # BOTH sit in a dynamic component: a contact against a static link adds no edge - even a 0-dof link that
    # belongs to a dof-carrying entity (e.g. a plane geom welded to the worldbody of a multi-free-body entity),
    # which an entity-level dof check would wrongly treat as dynamic and merge every body through it.
    for i_l in range(n_links):
        link_idx = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
        if links_info.n_dofs[link_idx] > 0:
            island_state.links_island_idx[func_find_root(island_state, i_l, i_b), i_b] = -2

    # Edges from contacts (read through contact_sort_idx so pruning/sorting is honored).
    for i_c in range(collider_state.n_contacts[i_b]):
        i_col = collider_state.contact_sort_idx[i_c, i_b]
        link_a = collider_state.contact_data.link_a[i_col, i_b]
        link_b = collider_state.contact_data.link_b[i_col, i_b]
        root_a = func_find_root(island_state, link_a, i_b)
        root_b = func_find_root(island_state, link_b, i_b)
        if island_state.links_island_idx[root_a, i_b] == -2 and island_state.links_island_idx[root_b, i_b] == -2:
            func_union(island_state, link_a, link_b, i_b)

    # Edges from equality constraints (model + dynamically registered welds).
    for i_eq in range(constraint_state.qd_n_equalities[i_b]):
        la, lb = func_equality_links(
            equalities_info,
            joints_info,
            links_info,
            i_eq,
            i_b,
            n_links,
            static_rigid_sim_config,
        )
        if la >= 0 and lb >= 0:
            root_a = func_find_root(island_state, la, i_b)
            root_b = func_find_root(island_state, lb, i_b)
            if island_state.links_island_idx[root_a, i_b] == -2 and island_state.links_island_idx[root_b, i_b] == -2:
                func_union(island_state, la, lb, i_b)

    # Hibernated islands: re-union along the daisy chain so a sleeping group (which generates no live
    # contacts to union it) stays one island across steps, matching the partition the wakeup walks.
    if qd.static(static_rigid_sim_config.use_hibernation):
        for i_l in range(n_links):
            i_next_l = island_state.hibernated_next_link[i_l, i_b]
            if 0 <= i_next_l < n_links and i_next_l != i_l:
                func_union(island_state, i_l, i_next_l, i_b)

    # Label each dynamic component (root marked -2 above). A component (root = min link index) is labeled the first
    # time one of its dof-links is seen, in ascending link order, so labels are deterministic and each island's
    # gathered global DOFs end up ascending (until the fill-reducing reorder below, where enabled) - the per-island
    # Hessian block lives in constraint_state.nt_H at those global rows/cols, triangle-oriented by local position.
    n_islands = 0
    for i_l in range(n_links):
        link_idx = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
        if links_info.n_dofs[link_idx] > 0:
            root = func_find_root(island_state, i_l, i_b)
            if island_state.links_island_idx[root, i_b] == -2:
                island_state.links_island_idx[root, i_b] = n_islands
                n_islands = n_islands + 1
    island_state.n_islands[i_b] = n_islands

    # Propagate the root's label to every link in its component (links in dof-less components stay -1).
    for i_l in range(n_links):
        root = func_find_root(island_state, i_l, i_b)
        island_state.links_island_idx[i_l, i_b] = island_state.links_island_idx[root, i_b]

    # Mark islands whose every link is asleep (read by the hibernation decision on the next step to skip
    # already-sleeping islands). An island is hibernated unless it has at least one awake link.
    if qd.static(static_rigid_sim_config.use_hibernation):
        for i_island in range(n_islands):
            island_state.is_hibernated[i_island, i_b] = 1
        for i_l in range(n_links):
            i_island = island_state.links_island_idx[i_l, i_b]
            if i_island >= 0 and not links_state.is_hibernated[i_l, i_b]:
                island_state.is_hibernated[i_island, i_b] = 0

    # Build the per-island link list (island -> link-idx ranges).
    for i_l in range(n_links):
        i_island = island_state.links_island_idx[i_l, i_b]
        if i_island >= 0:
            island_state.link_slices.n[i_island, i_b] = island_state.link_slices.n[i_island, i_b] + 1
    link_list_start = 0
    for i_island in range(n_islands):
        island_state.link_slices.start[i_island, i_b] = link_list_start
        island_state.link_slices.curr[i_island, i_b] = link_list_start
        link_list_start = link_list_start + island_state.link_slices.n[i_island, i_b]
    for i_l in range(n_links):
        i_island = island_state.links_island_idx[i_l, i_b]
        if i_island >= 0:
            island_state.link_id[island_state.link_slices.curr[i_island, i_b], i_b] = i_l
            island_state.link_slices.curr[i_island, i_b] = island_state.link_slices.curr[i_island, i_b] + 1

    # Build the per-island dof list (the block-gather map: local dof -> global dof, ascending). dof_id is grouped
    # by island; for the monolith (one island over all dofs in order) it is the identity permutation. Links are
    # visited in ascending index order and dof ranges grow with link index, so each island's global DOFs end up
    # ascending even when a component's links are non-contiguous (e.g. an entity's free bodies interleaved).
    for i_l in range(n_links):
        i_island = island_state.links_island_idx[i_l, i_b]
        if i_island >= 0:
            link_idx = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
            island_state.dof_slices.n[i_island, i_b] = (
                island_state.dof_slices.n[i_island, i_b] + links_info.n_dofs[link_idx]
            )
    dof_list_start = 0
    for i_island in range(n_islands):
        island_state.dof_slices.start[i_island, i_b] = dof_list_start
        island_state.dof_slices.curr[i_island, i_b] = dof_list_start
        dof_list_start = dof_list_start + island_state.dof_slices.n[i_island, i_b]
    for i_l in range(n_links):
        i_island = island_state.links_island_idx[i_l, i_b]
        if i_island >= 0:
            link_idx = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
            for i_d in range(links_info.dof_start[link_idx], links_info.dof_end[link_idx]):
                island_state.dof_id[island_state.dof_slices.curr[i_island, i_b], i_b] = i_d
                island_state.dof_local_pos[i_d, i_b] = (
                    island_state.dof_slices.curr[i_island, i_b] - island_state.dof_slices.start[i_island, i_b]
                )
                island_state.dofs_island_idx[i_d, i_b] = i_island
                island_state.dof_slices.curr[i_island, i_b] = island_state.dof_slices.curr[i_island, i_b] + 1

    # Build the per-island contact list (island -> contact ranges in contact_id). A contact belongs to the island of
    # its dof-carrying endpoint (both endpoints share an island when both carry dofs, since the contact unioned
    # them; otherwise one side is a fixed body).
    for i_c in range(collider_state.n_contacts[i_b]):
        i_col = collider_state.contact_sort_idx[i_c, i_b]
        link_a = collider_state.contact_data.link_a[i_col, i_b]
        link_b = collider_state.contact_data.link_b[i_col, i_b]
        i_island = island_state.links_island_idx[link_a, i_b]
        if i_island < 0:
            i_island = island_state.links_island_idx[link_b, i_b]
        if i_island >= 0:
            island_state.contact_slices.n[i_island, i_b] = island_state.contact_slices.n[i_island, i_b] + 1
    contact_list_start = 0
    for i_island in range(n_islands):
        island_state.contact_slices.start[i_island, i_b] = contact_list_start
        island_state.contact_slices.curr[i_island, i_b] = contact_list_start
        contact_list_start = contact_list_start + island_state.contact_slices.n[i_island, i_b]
    for i_c in range(collider_state.n_contacts[i_b]):
        i_col = collider_state.contact_sort_idx[i_c, i_b]
        link_a = collider_state.contact_data.link_a[i_col, i_b]
        link_b = collider_state.contact_data.link_b[i_col, i_b]
        i_island = island_state.links_island_idx[link_a, i_b]
        if i_island < 0:
            i_island = island_state.links_island_idx[link_b, i_b]
        if i_island >= 0:
            island_state.contact_id[island_state.contact_slices.curr[i_island, i_b], i_b] = i_col
            island_state.contact_slices.curr[i_island, i_b] = island_state.contact_slices.curr[i_island, i_b] + 1

    # Fill-reducing DOF reordering for the CPU per-island skyline path: rebuild each island's dof_id in reverse
    # Cuthill-McKee order of its kinematic trees over the contact adjacency, instead of ascending global order. The
    # build-time DOF order says nothing about which bodies end up in contact, so a settled pile otherwise produces a
    # near-dense skyline; ordering trees by contact adjacency shrinks the envelope the factor, rank-1 updates and
    # triangular solves sweep. Trees stay contiguous with their DOFs in original relative order, preserving the
    # local contiguity of the mass blocks. Every per-island consumer addresses nt_H through (dof_id, dof_local_pos)
    # pairs with island-local triangle orientation, so a non-monotonic dof_id only changes where blocks are stored.
    if qd.static(
        static_rigid_sim_config.sparse_solve
        and static_rigid_sim_config.enable_per_island_solve
        and not static_rigid_sim_config.sparse_envelope
    ):
        for i_island in range(n_islands):
            link_base = island_state.link_slices.start[i_island, i_b]
            n_isl_links = island_state.link_slices.n[i_island, i_b]
            con_base = island_state.contact_slices.start[i_island, i_b]
            n_isl_cons = island_state.contact_slices.n[i_island, i_b]

            # Collect the island's kinematic-tree roots (island-local tree slots, in ascending link order).
            n_trees = 0
            for i_l_ in range(n_isl_links):
                i_l = island_state.link_id[link_base + i_l_, i_b]
                link_idx = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                i_root = links_info.root_idx[link_idx]
                island_state.rcm_tree_pos[i_root, i_b] = -1
            for i_l_ in range(n_isl_links):
                i_l = island_state.link_id[link_base + i_l_, i_b]
                link_idx = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                i_root = links_info.root_idx[link_idx]
                if island_state.rcm_tree_pos[i_root, i_b] == -1:
                    island_state.rcm_tree_pos[i_root, i_b] = n_trees
                    n_trees = n_trees + 1

            # Reordering cannot shrink the envelope of at most two trees; keep the ascending build order there so
            # small islands stay bit-identical to the unordered path (and the common tiny-island case costs nothing).
            if n_trees <= 2:
                continue

            # Contact degree per tree (duplicate contacts inflate degrees, which only biases the tie-break).
            for i_t in range(n_trees):
                island_state.rcm_tree_degree[link_base + i_t, i_b] = 0
                island_state.rcm_tree_is_ordered[link_base + i_t, i_b] = False
            for i_c_ in range(n_isl_cons):
                i_col = island_state.contact_id[con_base + i_c_, i_b]
                i_ta, i_tb = func_contact_tree_slots(
                    i_b, i_col, links_info, collider_state, island_state, static_rigid_sim_config
                )
                if i_ta >= 0 and i_tb >= 0 and i_ta != i_tb:
                    island_state.rcm_tree_degree[link_base + i_ta, i_b] = (
                        island_state.rcm_tree_degree[link_base + i_ta, i_b] + 1
                    )
                    island_state.rcm_tree_degree[link_base + i_tb, i_b] = (
                        island_state.rcm_tree_degree[link_base + i_tb, i_b] + 1
                    )

            # Cuthill-McKee: BFS from the lowest-degree unordered tree, appending each frontier sorted by degree.
            n_ordered = 0
            i_head = 0
            while n_ordered < n_trees:
                i_t_start = -1
                for i_t in range(n_trees):
                    if not island_state.rcm_tree_is_ordered[link_base + i_t, i_b] and (
                        i_t_start == -1
                        or island_state.rcm_tree_degree[link_base + i_t, i_b]
                        < island_state.rcm_tree_degree[link_base + i_t_start, i_b]
                    ):
                        i_t_start = i_t
                island_state.rcm_tree_order[link_base + n_ordered, i_b] = i_t_start
                island_state.rcm_tree_is_ordered[link_base + i_t_start, i_b] = True
                n_ordered = n_ordered + 1
                while i_head < n_ordered:
                    i_t_head = island_state.rcm_tree_order[link_base + i_head, i_b]
                    n_frontier_start = n_ordered
                    for i_c_ in range(n_isl_cons):
                        i_col = island_state.contact_id[con_base + i_c_, i_b]
                        i_ta, i_tb = func_contact_tree_slots(
                            i_b, i_col, links_info, collider_state, island_state, static_rigid_sim_config
                        )
                        i_t_next = -1
                        if i_ta == i_t_head and i_tb >= 0:
                            i_t_next = i_tb
                        elif i_tb == i_t_head and i_ta >= 0:
                            i_t_next = i_ta
                        if i_t_next >= 0 and not island_state.rcm_tree_is_ordered[link_base + i_t_next, i_b]:
                            # Insert into the current frontier keeping it sorted by ascending degree.
                            i_ins = n_ordered
                            while i_ins > n_frontier_start and (
                                island_state.rcm_tree_degree[
                                    link_base + island_state.rcm_tree_order[link_base + i_ins - 1, i_b], i_b
                                ]
                                > island_state.rcm_tree_degree[link_base + i_t_next, i_b]
                            ):
                                island_state.rcm_tree_order[link_base + i_ins, i_b] = island_state.rcm_tree_order[
                                    link_base + i_ins - 1, i_b
                                ]
                                i_ins = i_ins - 1
                            island_state.rcm_tree_order[link_base + i_ins, i_b] = i_t_next
                            island_state.rcm_tree_is_ordered[link_base + i_t_next, i_b] = True
                            n_ordered = n_ordered + 1
                    i_head = i_head + 1

            # Rebuild dof_id with trees in REVERSE Cuthill-McKee order, each tree's links in ascending link order.
            # The per-tree link scan is O(n_trees * n_isl_links), dominated by the BFS neighbor sweep above
            # (O(n_trees * n_isl_cons)), so it is not worth a per-tree cursor buffer.
            i_dof_curr = island_state.dof_slices.start[i_island, i_b]
            for i_t_ in range(n_trees):
                i_t = island_state.rcm_tree_order[link_base + (n_trees - 1 - i_t_), i_b]
                for i_l_ in range(n_isl_links):
                    i_l = island_state.link_id[link_base + i_l_, i_b]
                    link_idx = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                    if island_state.rcm_tree_pos[links_info.root_idx[link_idx], i_b] == i_t:
                        for i_d in range(links_info.dof_start[link_idx], links_info.dof_end[link_idx]):
                            island_state.dof_id[i_dof_curr, i_b] = i_d
                            island_state.dof_local_pos[i_d, i_b] = (
                                i_dof_curr - island_state.dof_slices.start[i_island, i_b]
                            )
                            i_dof_curr = i_dof_curr + 1


@qd.func
def _sort_island_contacts(
    i_b,
    start,
    n,
    contact_idx: qd.Tensor,
    contacts_pos: qd.Tensor,
    contacts_geom_a: qd.Tensor,
    contacts_geom_b: qd.Tensor,
):
    """Insertion-sort the contact-index slice contact_idx[start : start + n] by a deterministic total order.

    The order (pos_x, geom_a, geom_b, pos_y, pos_z) is a pure function of contact data, so it is independent of the
    racy atomic_add narrowphase layout. contact_idx is island_state.contact_id for the per-island sort (disjoint
    island slices sort concurrently, one warp lane per island) or collider_state.contact_sort_idx for the global
    islands-off sort. The contact-data tensors are passed as leaves rather than the whole collider_state struct so
    that contact_sort_idx can be sorted in place without the struct-expansion aliasing its own field.
    """
    for i_s in range(start + 1, start + n):
        i_p = contact_idx[i_s, i_b]
        pos_p = contacts_pos[i_p, i_b]
        geom_a_p = contacts_geom_a[i_p, i_b]
        geom_b_p = contacts_geom_b[i_p, i_b]
        j_s = i_s - 1
        while j_s >= start:
            i_q = contact_idx[j_s, i_b]
            pos_q = contacts_pos[i_q, i_b]
            precedes = pos_q[0] < pos_p[0]
            if not precedes and pos_q[0] == pos_p[0]:
                geom_a_q = contacts_geom_a[i_q, i_b]
                if geom_a_q < geom_a_p:
                    precedes = True
                elif geom_a_q == geom_a_p:
                    geom_b_q = contacts_geom_b[i_q, i_b]
                    if geom_b_q < geom_b_p:
                        precedes = True
                    elif geom_b_q == geom_b_p:
                        if pos_q[1] < pos_p[1]:
                            precedes = True
                        elif pos_q[1] == pos_p[1]:
                            precedes = pos_q[2] <= pos_p[2]
            if precedes:
                break
            contact_idx[j_s + 1, i_b] = i_q
            j_s = j_s - 1
        contact_idx[j_s + 1, i_b] = i_p


@qd.func
def func_island_contacts_total(i_b, island_state: array_class.IslandState):
    # Total in-island contact count = sum of per-island contact slice sizes (== n_contacts when every contact touches a
    # dof-carrying body, which holds whenever a static-static pair is not collided).
    total = 0
    for i_island in range(island_state.n_islands[i_b]):
        total = total + island_state.contact_slices.n[i_island, i_b]
    return total
