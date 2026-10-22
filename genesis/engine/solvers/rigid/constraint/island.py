import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.simt as su

from ..abd.misc import func_wakeup_link
from ..collider.contact import func_contact_order_key, func_promote_woken_contacts


# Partition the dof-carrying kinematic trees (see trees_root_idx in array_class.py) into islands: the connected
# components of the trees under the edges of (1) the contacts, (2) the equality constraints (CONNECT/WELD on links,
# JOINT on joints) and (3) the hibernation chains. A single Genesis entity holding several free bodies (common in MJCF)
# thus splits into one island per free body, while an articulated body's links stay one island. Each island is an
# exactly decoupled (block-diagonal) sub-problem of the constraint solve. A component is labeled by its smallest tree,
# whatever the order its edges are met in, which keeps the partition, hence the per-island solve, deterministic.


@qd.func
def func_find_tree_root(i_t: int, i_b: int, island_state: array_class.IslandState):
    # Path-halving find over the trees
    root = i_t
    while island_state.trees_parent_idx[root, i_b] != root:
        island_state.trees_parent_idx[root, i_b] = island_state.trees_parent_idx[
            island_state.trees_parent_idx[root, i_b], i_b
        ]
        root = island_state.trees_parent_idx[root, i_b]
    return root


@qd.func
def func_union_trees(i_ta: int, i_tb: int, i_b: int, island_state: array_class.IslandState):
    # Union by minimum index: the root of a component is its smallest tree, regardless of the order edges are processed
    root_a = func_find_tree_root(i_ta, i_b, island_state)
    root_b = func_find_tree_root(i_tb, i_b, island_state)
    if root_a < root_b:
        island_state.trees_parent_idx[root_b, i_b] = root_a
    elif root_b < root_a:
        island_state.trees_parent_idx[root_a, i_b] = root_b


@qd.func
def func_hook_trees(i_ta: int, i_tb: int, i_b: int, island_state: array_class.IslandState):
    # One hook of the cooperative build (see func_build_islands_coop): the larger root of two components points at the
    # smaller one, and n_islands holds the change flag of the round
    root_a = func_tree_component(i_ta, i_b, island_state)
    root_b = func_tree_component(i_tb, i_b, island_state)
    if root_a != root_b:
        qd.atomic_min(island_state.trees_parent_idx[qd.max(root_a, root_b), i_b], qd.min(root_a, root_b))
        island_state.n_islands[i_b] = 1


@qd.func
def func_joint_link(i_joint: int, i_b: int, n_links: int, dyn_info: array_class.DynInfo, rigid_config: qd.template()):
    # JointsInfo carries no link mapping, so locate the link whose dof range owns the joint's first dof. Joint
    # equalities are rare and link counts are small, so the linear scan is cheap.
    I_j = [i_joint, i_b] if qd.static(rigid_config.batch_joints_info) else i_joint
    i_dof = dyn_info.joints.dof_start[I_j]
    link = -1
    for i_l in range(n_links):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.dof_start[I_l] <= i_dof < dyn_info.links.dof_end[I_l]:
            link = i_l
            break
    return link


@qd.func
def func_equality_links(i_eq: int, i_b: int, n_links: int, dyn_info: array_class.DynInfo, rigid_config: qd.template()):
    # Map an equality constraint to the pair of links it couples. CONNECT/WELD reference links; JOINT references joints.
    obj1 = dyn_info.equalities.eq_obj1id[i_eq, i_b]
    obj2 = dyn_info.equalities.eq_obj2id[i_eq, i_b]
    eq_type = dyn_info.equalities.eq_type[i_eq, i_b]
    i_l_a = -1
    i_l_b = -1
    if eq_type == gs.EQUALITY_TYPE.JOINT:
        i_l_a = func_joint_link(obj1, i_b, n_links, dyn_info, rigid_config)
        if obj2 >= 0:
            i_l_b = func_joint_link(obj2, i_b, n_links, dyn_info, rigid_config)
    else:
        i_l_a = obj1
        i_l_b = obj2
    return i_l_a, i_l_b


@qd.func
def func_union_link_trees(
    link_a: int,
    link_b: int,
    i_b: int,
    island_state: array_class.IslandState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Join the trees of two links an edge of env i_b couples: the serial union, or one hook of a round of the
    cooperative build (see func_build_islands_coop).

    An edge couples nothing when either link is -1 or static: a contact against a fixed body, a link with no chain
    successor.
    """
    if link_a >= 0 and link_b >= 0:
        # A static link belongs to no tree (links_tree_idx -1) and joins no island
        i_ta = rigid_info.links_tree_idx[link_a]
        i_tb = rigid_info.links_tree_idx[link_b]
        if i_ta >= 0 and i_tb >= 0:
            if qd.static(rigid_config.enable_cooperative_constraint_kernels):
                func_hook_trees(i_ta, i_tb, i_b, island_state)
            else:
                func_union_trees(i_ta, i_tb, i_b, island_state)


@qd.func
def func_union_model_edges(
    i_b: int,
    tid: int,
    has_sleepers: bool,
    island_state: array_class.IslandState,
    qd_n_equalities: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Join the trees of env i_b the equality constraints couple, then under hibernation the chain successor of every
    link, from lane tid of the cooperative build or from the one thread of the serial build (tid 0)."""
    _K = qd.static(32 if rigid_config.enable_cooperative_constraint_kernels else 1)
    n_links = rigid_info.links_tree_idx.shape[0]
    i_e = tid
    while i_e < qd_n_equalities[i_b]:
        link_a, link_b = func_equality_links(i_e, i_b, n_links, dyn_info, rigid_config)
        func_union_link_trees(link_a, link_b, i_b, island_state, rigid_info, rigid_config)
        i_e = i_e + _K
    if qd.static(rigid_config.use_hibernation):
        if has_sleepers:
            i_l = tid
            while i_l < n_links:
                link_b = island_state.hibernated_next_link[i_l, i_b]
                if link_b >= n_links or link_b == i_l:
                    link_b = -1
                func_union_link_trees(i_l, link_b, i_b, island_state, rigid_info, rigid_config)
                i_l = i_l + _K


@qd.func
def func_union_live_contacts(
    i_b: int,
    tid: int,
    collider_state: array_class.ColliderState,
    island_state: array_class.IslandState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Join the trees the live contacts of env i_b couple, in contact_sort_idx order, from lane tid of the cooperative
    build or from the one thread of the serial build (tid 0). The live contacts follow the kept ones of the sleepers
    (see n_contacts_hibernated in array_class.py)."""
    n_first_contact = collider_state.n_contacts_hibernated[i_b]
    n_contacts = collider_state.n_contacts[i_b] - n_first_contact
    i_e = tid
    while i_e < n_contacts:
        i_col = collider_state.contact_sort_idx[n_first_contact + i_e, i_b]
        func_union_link_trees(
            collider_state.contact_data.link_a[i_col, i_b],
            collider_state.contact_data.link_b[i_col, i_b],
            i_b,
            island_state,
            rigid_info,
            rigid_config,
        )
        i_e = i_e + qd.static(32 if rigid_config.enable_cooperative_constraint_kernels else 1)


@qd.func
def func_union_contact_edges(
    contact_data: qd.template(),
    island_data: qd.template(),
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    i_b: int,
    tid: int,
):
    """The edge source of the contacts in the island partition (see RigidIsland.on_partition_edges)."""
    func_union_live_contacts(
        i_b, tid, contact_data.collider_state, island_data.island, rigid_data.rigid_info, rigid_config
    )


@qd.func
def func_constraint_island(i_c: int, i_b: int, constraint_state: array_class.ConstraintState):
    # A constraint couples dofs of a single island, so its island is that of any dof of its support, read from the
    # sparse dof list every assembly site fills (jac_dofs_idx, see _append_relevant_dof in solver.py); a row with an
    # empty support belongs to no island.
    i_island = -1
    if constraint_state.jac_n_dofs[i_c, i_b] > 0:
        i_d = constraint_state.jac_dofs_idx[i_c, 0, i_b]
        i_island = constraint_state.island.dofs_island_idx[i_d, i_b]
    return i_island


@qd.func
def func_group_constraints_by_island(
    i_b: int, constraint_state: array_class.ConstraintState, rigid_config: qd.template()
):
    """Group the constraints of one env by island and start every island iterating.

    The island of each constraint is resolved and the constraints are listed in contiguous per-island ranges of
    constraint_id, so the per-island solve iterates its own constraints. Every island's improved flag is then raised
    (see func_exit_decision in linesearch.py). The fill walks constraints in index order, so each island's constraint
    list stays order-deterministic. The whole pass is serial per env, folded into the per-env init loop of
    func_solve_init so the partition adds no kernel launch of its own: its work is a few integer operations per
    constraint.
    """
    n_islands = constraint_state.island.n_islands[i_b]
    n_con = constraint_state.n_constraints[i_b]
    if n_islands == 1:
        # A single island spans the whole env, so every constraint belongs to island 0 in index order and the grouping
        # is the identity. A constraint touching no dof carries jac == 0, so listing it in island 0 is harmless.
        constraint_state.island_solve.constraint_slices.start[0, i_b] = 0
        constraint_state.island_solve.constraint_slices.n[0, i_b] = n_con
        constraint_state.island_solve.constraint_slices.curr[0, i_b] = n_con
        for i_c in range(n_con):
            constraint_state.island_solve.constraint_id[i_c, i_b] = i_c
            constraint_state.island_solve.constraint_island_idx[i_c, i_b] = 0
    else:
        for i_island in range(n_islands):
            constraint_state.island_solve.constraint_slices.n[i_island, i_b] = 0

        for i_c in range(n_con):
            i_island = func_constraint_island(i_c, i_b, constraint_state)
            constraint_state.island_solve.constraint_island_idx[i_c, i_b] = i_island
            if i_island >= 0:
                constraint_state.island_solve.constraint_slices.n[i_island, i_b] = (
                    constraint_state.island_solve.constraint_slices.n[i_island, i_b] + 1
                )

        con_list_start = 0
        for i_island in range(n_islands):
            constraint_state.island_solve.constraint_slices.start[i_island, i_b] = con_list_start
            constraint_state.island_solve.constraint_slices.curr[i_island, i_b] = con_list_start
            con_list_start = con_list_start + constraint_state.island_solve.constraint_slices.n[i_island, i_b]

        for i_c in range(n_con):
            i_island = constraint_state.island_solve.constraint_island_idx[i_c, i_b]
            if i_island >= 0:
                constraint_state.island_solve.constraint_id[
                    constraint_state.island_solve.constraint_slices.curr[i_island, i_b], i_b
                ] = i_c
                constraint_state.island_solve.constraint_slices.curr[i_island, i_b] = (
                    constraint_state.island_solve.constraint_slices.curr[i_island, i_b] + 1
                )

    # Every awake island starts iterating, a constraint-free one included: a warm start leaves its acceleration at the
    # previous step's, which its own iteration brings back to the unconstrained one.
    for i_island in range(n_islands):
        is_awake = n_con > 0
        if qd.static(rigid_config.use_hibernation):
            is_awake = is_awake and constraint_state.island.is_hibernated[i_island, i_b] == 0
        constraint_state.island_solve.improved[i_island, i_b] = is_awake


@qd.func
def func_chunk_island_rank(tid: int, i_island: int, sh_chunk):
    """Rank of a lane among the lanes of the chunk that hold the same island: the position of this lane's item within
    the island, once added to the island's running total."""
    rank = 0
    for j in range(tid):
        if sh_chunk[j] == i_island:
            rank = rank + 1
    return rank


@qd.func
def func_group_constraints_by_island_coop(
    i_b: int, tid: int, sh_chunk, constraint_state: array_class.ConstraintState, rigid_config: qd.template()
):
    """Group the constraints of one env by island with the _K lanes of its block, in constraint index order.

    The grouping of func_group_constraints_by_island: the labels resolved lane by lane, the counts by atomics, the
    starts by a subgroup scan, and the fill per chunk of _K constraints, each ranked among the chunk's constraints of
    its island through sh_chunk (one slot per lane) and placed at its island's cursor, so the lists stay in constraint
    index order.
    """
    _K = qd.static(32)
    n_islands = constraint_state.island.n_islands[i_b]
    n_con = constraint_state.n_constraints[i_b]
    if n_islands == 1:
        if tid == 0:
            constraint_state.island_solve.constraint_slices.start[0, i_b] = 0
            constraint_state.island_solve.constraint_slices.n[0, i_b] = n_con
            constraint_state.island_solve.constraint_slices.curr[0, i_b] = n_con
        i_c = tid
        while i_c < n_con:
            constraint_state.island_solve.constraint_id[i_c, i_b] = i_c
            constraint_state.island_solve.constraint_island_idx[i_c, i_b] = 0
            i_c = i_c + _K
    else:
        i_island = tid
        while i_island < n_islands:
            constraint_state.island_solve.constraint_slices.n[i_island, i_b] = 0
            i_island = i_island + _K
        qd.simt.block.sync()
        i_c = tid
        while i_c < n_con:
            i_island = func_constraint_island(i_c, i_b, constraint_state)
            constraint_state.island_solve.constraint_island_idx[i_c, i_b] = i_island
            if i_island >= 0:
                qd.atomic_add(constraint_state.island_solve.constraint_slices.n[i_island, i_b], 1)
            i_c = i_c + _K
        qd.simt.block.sync()
        carry = 0
        for i_chunk in range((n_islands + _K - 1) // _K):
            i_island = i_chunk * _K + tid
            count = 0
            if i_island < n_islands:
                count = constraint_state.island_solve.constraint_slices.n[i_island, i_b]
            count_incl, count_total = su.qd_block_scan(count)
            if i_island < n_islands:
                constraint_state.island_solve.constraint_slices.start[i_island, i_b] = carry + count_incl - count
                constraint_state.island_solve.constraint_slices.curr[i_island, i_b] = carry + count_incl - count
            carry = carry + count_total
        qd.simt.block.sync()
        for i_chunk in range((n_con + _K - 1) // _K):
            i_c = i_chunk * _K + tid
            i_island = -1
            if i_c < n_con:
                i_island = constraint_state.island_solve.constraint_island_idx[i_c, i_b]
            sh_chunk[tid] = i_island
            qd.simt.block.sync()
            if i_island >= 0:
                i_pos = constraint_state.island_solve.constraint_slices.curr[i_island, i_b]
                i_pos = i_pos + func_chunk_island_rank(tid, i_island, sh_chunk)
                constraint_state.island_solve.constraint_id[i_pos, i_b] = i_c
            qd.simt.block.sync()
            if i_island >= 0:
                qd.atomic_add(constraint_state.island_solve.constraint_slices.curr[i_island, i_b], 1)
            qd.simt.block.sync()

    # Every awake island starts iterating, a constraint-free one included, see func_group_constraints_by_island
    i_island = tid
    while i_island < n_islands:
        is_awake = n_con > 0
        if qd.static(rigid_config.use_hibernation):
            is_awake = is_awake and constraint_state.island.is_hibernated[i_island, i_b] == 0
        constraint_state.island_solve.improved[i_island, i_b] = is_awake
        i_island = i_island + _K


@qd.func
def func_dof_range_start(i_island: int, i_b: int, island_state: array_class.IslandState):
    """First dof of an island whose ascending dof list holds consecutive dofs, -1 otherwise (see dof_range_start in
    array_class.py).

    The lists of the build ascend, so the first and last entries decide.
    """
    dof_lo = island_state.dof_slices.start[i_island, i_b]
    n_dofs_island = island_state.dof_slices.n[i_island, i_b]
    range_start = island_state.dof_id[dof_lo, i_b]
    if island_state.dof_id[dof_lo + n_dofs_island - 1, i_b] - range_start + 1 != n_dofs_island:
        range_start = -1
    return range_start


@qd.func
def func_finish_island_partition(
    i_b: int,
    has_sleepers: bool,
    dyn_state: array_class.DynState,
    island_state: array_class.IslandState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Label a union-find forest and materialize its shared island lists."""
    n_trees = rigid_info.trees_root_idx.shape[0]
    n_links = rigid_info.links_tree_idx.shape[0]
    n_islands = 0
    for i_t in range(n_trees):
        island_state.trees_island_idx[i_t, i_b] = -1
        if island_state.trees_parent_idx[i_t, i_b] == i_t and rigid_info.trees_n_dofs[i_t] > 0:
            island_state.trees_island_idx[i_t, i_b] = n_islands
            n_islands = n_islands + 1
    for i_t in range(n_trees):
        island_state.trees_island_idx[i_t, i_b] = island_state.trees_island_idx[
            func_find_tree_root(i_t, i_b, island_state), i_b
        ]
    island_state.n_islands[i_b] = n_islands

    for i_island in range(n_islands):
        island_state.dof_slices.n[i_island, i_b] = 0
        if qd.static(rigid_config.use_hibernation):
            island_state.link_slices.n[i_island, i_b] = 0
        island_state.inertia[i_island, i_b] = 0.0
        if qd.static(rigid_config.use_hibernation):
            island_state.is_hibernated[i_island, i_b] = 1
    for i_t in range(n_trees):
        i_island = island_state.trees_island_idx[i_t, i_b]
        if i_island >= 0:
            if qd.static(rigid_config.use_hibernation):
                island_state.link_slices.n[i_island, i_b] += rigid_info.trees_n_links[i_t]
            island_state.dof_slices.n[i_island, i_b] += rigid_info.trees_n_dofs[i_t]

    link_list_start = 0
    dof_list_start = 0
    for i_island in range(n_islands):
        if qd.static(rigid_config.use_hibernation):
            island_state.link_slices.start[i_island, i_b] = link_list_start
            island_state.link_slices.curr[i_island, i_b] = link_list_start
            link_list_start += island_state.link_slices.n[i_island, i_b]
        island_state.dof_slices.start[i_island, i_b] = dof_list_start
        island_state.dof_slices.curr[i_island, i_b] = dof_list_start
        dof_list_start += island_state.dof_slices.n[i_island, i_b]

    for i_t in range(n_trees):
        i_island = island_state.trees_island_idx[i_t, i_b]
        for i_l in range(rigid_info.trees_root_idx[i_t], rigid_info.trees_link_end[i_t]):
            if rigid_info.links_tree_idx[i_l] == i_t:
                island_state.links_island_idx[i_l, i_b] = i_island
                if qd.static(rigid_config.use_hibernation):
                    if i_island >= 0:
                        i_pos = island_state.link_slices.curr[i_island, i_b]
                        island_state.link_id[i_pos, i_b] = i_l
                        island_state.link_slices.curr[i_island, i_b] = i_pos + 1
        if i_island >= 0:
            dof_start_tree = rigid_info.trees_dof_start[i_t]
            for i_d in range(dof_start_tree, dof_start_tree + rigid_info.trees_n_dofs[i_t]):
                i_pos = island_state.dof_slices.curr[i_island, i_b]
                island_state.dof_id[i_pos, i_b] = i_d
                island_state.dof_local_pos[i_d, i_b] = i_pos - island_state.dof_slices.start[i_island, i_b]
                island_state.dofs_island_idx[i_d, i_b] = i_island
                island_state.dof_slices.curr[i_island, i_b] = i_pos + 1
                island_state.inertia[i_island, i_b] += rigid_info.mass_mat[i_d, i_d, i_b]
    for i_island in range(n_islands):
        island_state.dof_range_start[i_island, i_b] = func_dof_range_start(i_island, i_b, island_state)

    woke_sleepers = False
    if qd.static(rigid_config.use_hibernation):
        if has_sleepers:
            for i_l in range(n_links):
                i_island = island_state.links_island_idx[i_l, i_b]
                if i_island >= 0 and not dyn_state.links.is_hibernated[i_l, i_b]:
                    island_state.is_hibernated[i_island, i_b] = 0
            n_awake_dofs_before = rigid_info.n_awake_dofs[i_b]
            for i_l in range(n_links):
                i_island = island_state.links_island_idx[i_l, i_b]
                if i_island >= 0 and island_state.is_hibernated[i_island, i_b] == 0:
                    func_wakeup_link(i_l, i_b, dyn_state, island_state, dyn_info, rigid_info, rigid_config)
            woke_sleepers = rigid_info.n_awake_dofs[i_b] != n_awake_dofs_before
        else:
            for i_island in range(n_islands):
                island_state.is_hibernated[i_island, i_b] = 0
    return woke_sleepers


@qd.func
def func_build_islands(
    i_b: int,
    edges: qd.template(),
    dyn_state: array_class.DynState,
    island_state: array_class.IslandState,
    qd_n_equalities: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Build one env's island partition serially, returning whether it woke sleepers.

    The components of the trees come from a union-find over the edges: the ones the edge sources of the partition
    contribute (the live contacts, see RigidIsland.partition_edges), then the equality constraints and the
    hibernation chains (func_union_model_edges), labeled in ascending order of their smallest tree. Each island then
    lists its dofs (the trees in ascending order, the dofs of each in ascending order, so an island's dofs ascend) and,
    under hibernation, which alone reads them, its links, and holds its inertia (the trace of the mass matrix over its
    dofs, the scale of its convergence tests) and, under hibernation, its sleeping flag. The sleepers of an island an
    awake body reaches wake here; the contact ordering then promotes their kept contacts (func_promote_woken_contacts).
    A settled island falls asleep after the solve (func_hibernate_island_if_settled).
    """
    n_trees = rigid_info.trees_root_idx.shape[0]
    # An env with no sleeper (see n_awake_dofs in array_class.py) has no chain edge to union, no island flag to derive
    # and no sleeper to wake
    has_sleepers = False
    if qd.static(rigid_config.use_hibernation):
        has_sleepers = rigid_info.n_awake_dofs[i_b] < dyn_state.dofs.is_hibernated.shape[0]

    for i_t in range(n_trees):
        island_state.trees_parent_idx[i_t, i_b] = i_t
    edges.invoke((i_b, 0))
    func_union_model_edges(i_b, 0, has_sleepers, island_state, qd_n_equalities, dyn_info, rigid_info, rigid_config)

    return func_finish_island_partition(
        i_b,
        has_sleepers,
        dyn_state,
        island_state,
        dyn_info,
        rigid_info,
        rigid_config,
    )


@qd.func
def func_build_single_island(
    i_b: int,
    island_state: array_class.IslandState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Write the partition of one env of a single-island scene serially: one island holding every dof in order.

    The lists are the identity and the island's inertia the trace of the mass matrix. Reserved for scenes off the CPU
    skyline path, which alone reads the tree labels the full build resolves (see func_substep_partition_islands).
    Under hibernation the island is awake and its link list, written once at build (see
    RigidIsland.initialize_data), stands. An env holding a sleeper takes the full build.
    """
    n_dofs = island_state.dof_id.shape[0]
    island_state.n_islands[i_b] = 1
    island_state.dof_slices.start[0, i_b] = 0
    island_state.dof_slices.n[0, i_b] = n_dofs
    island_state.dof_slices.curr[0, i_b] = n_dofs
    island_state.dof_range_start[0, i_b] = 0
    inertia = gs.qd_float(0.0)
    for i_d in range(n_dofs):
        island_state.dof_id[i_d, i_b] = i_d
        island_state.dof_local_pos[i_d, i_b] = i_d
        island_state.dofs_island_idx[i_d, i_b] = 0
        inertia = inertia + rigid_info.mass_mat[i_d, i_d, i_b]
    island_state.inertia[0, i_b] = inertia
    if qd.static(rigid_config.use_hibernation):
        island_state.is_hibernated[0, i_b] = 0


@qd.func
def func_build_single_island_coop(
    i_b: int,
    tid: int,
    island_state: array_class.IslandState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Write the partition of one env of a single-island scene with the _K lanes of its block, see
    func_build_single_island."""
    _K = qd.static(32)
    n_dofs = island_state.dof_id.shape[0]
    if tid == 0:
        island_state.n_islands[i_b] = 1
        island_state.dof_slices.start[0, i_b] = 0
        island_state.dof_slices.n[0, i_b] = n_dofs
        island_state.dof_slices.curr[0, i_b] = n_dofs
        island_state.dof_range_start[0, i_b] = 0
        if qd.static(rigid_config.use_hibernation):
            island_state.is_hibernated[0, i_b] = 0
    inertia = gs.qd_float(0.0)
    i_d = tid
    while i_d < n_dofs:
        island_state.dof_id[i_d, i_b] = i_d
        island_state.dof_local_pos[i_d, i_b] = i_d
        island_state.dofs_island_idx[i_d, i_b] = 0
        inertia = inertia + rigid_info.mass_mat[i_d, i_d, i_b]
        i_d = i_d + _K
    inertia = su.qd_block_sum(inertia)
    if tid == 0:
        island_state.inertia[0, i_b] = inertia


@qd.func
def func_tree_component(i_t: int, i_b: int, island_state: array_class.IslandState):
    # Root of a tree's component in the union-find forest, reading only
    root = i_t
    while island_state.trees_parent_idx[root, i_b] != root:
        root = island_state.trees_parent_idx[root, i_b]
    return root


@qd.func
def func_build_islands_coop(
    i_b: int,
    tid: int,
    edges: qd.template(),
    dyn_state: array_class.DynState,
    island_state: array_class.IslandState,
    qd_n_equalities: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Build one env's island partition with the _K lanes of its block, recording whether it woke sleepers.

    The partition is the very one func_build_islands builds serially: the same labels and the same lists in the same
    order, so a per-island result depends on the partition alone.

    Every pass is lane-strided over the trees, the edges or the islands and fenced from the next by a block barrier,
    and n_islands holds the change flag of the hooking rounds until the label count takes its place. The components
    are found by min-label hooking: an edge between two components points the larger label at the smaller (every edge
    source hooking its own edges in each round, see func_union_link_trees), every tree then jumps to its root, until a
    pass meets no edge across two components. The roots are then ranked in tree order by a subgroup scan, the counts
    summed by atomics, the starts scanned, and each tree writes its own links and dofs after those of the earlier trees
    of its island, so an item's position depends only on the earlier items.
    """
    _K = qd.static(32)
    n_trees = rigid_info.trees_root_idx.shape[0]
    n_links = rigid_info.links_tree_idx.shape[0]
    n_dofs = island_state.dof_id.shape[0]
    # Every lane reads the env's sleeper gate, see func_build_islands
    has_sleepers = False
    if qd.static(rigid_config.use_hibernation):
        has_sleepers = rigid_info.n_awake_dofs[i_b] < dyn_state.dofs.is_hibernated.shape[0]

    i_t = tid
    while i_t < n_trees:
        island_state.trees_parent_idx[i_t, i_b] = i_t
        island_state.dof_slices.n[i_t, i_b] = 0
        if qd.static(rigid_config.use_hibernation):
            island_state.link_slices.n[i_t, i_b] = 0
        i_t = i_t + _K
    qd.simt.block.sync()

    # Hooking rounds over the edges, until none crosses two components. A hook may overwrite one made concurrently on
    # the same root: the edge behind the lost hook stays, and the next round meets it again.
    while True:
        if tid == 0:
            island_state.n_islands[i_b] = 0
        qd.simt.block.sync()
        edges.invoke((i_b, tid))
        func_union_model_edges(
            i_b, tid, has_sleepers, island_state, qd_n_equalities, dyn_info, rigid_info, rigid_config
        )
        qd.simt.block.sync()
        is_changed = island_state.n_islands[i_b]
        qd.simt.block.sync()
        if is_changed == 0:
            break
        # Every tree jumps to its root. A parent only ever moves to a smaller label, so a walk that meets another
        # lane's write still lands on an ancestor.
        i_t = tid
        while i_t < n_trees:
            island_state.trees_parent_idx[i_t, i_b] = func_tree_component(i_t, i_b, island_state)
            i_t = i_t + _K
        qd.simt.block.sync()

    # Label the components in ascending order of their root, a subgroup scan over the roots per chunk of _K trees. A
    # dof-less tree stays its own root, no edge reaching it, and labels no island.
    n_islands = 0
    for i_chunk in range((n_trees + _K - 1) // _K):
        i_t = i_chunk * _K + tid
        is_root = 0
        if i_t < n_trees:
            # The padding slot of a tree-less scene labels no island, see func_build_islands
            if island_state.trees_parent_idx[i_t, i_b] == i_t and rigid_info.trees_n_dofs[i_t] > 0:
                is_root = 1
        roots_incl, roots_total = su.qd_block_scan(is_root)
        if is_root == 1:
            island_state.trees_island_idx[i_t, i_b] = n_islands + roots_incl - 1
        n_islands = n_islands + roots_total
    qd.simt.block.sync()
    i_t = tid
    while i_t < n_trees:
        island_state.trees_island_idx[i_t, i_b] = island_state.trees_island_idx[
            island_state.trees_parent_idx[i_t, i_b], i_b
        ]
        i_t = i_t + _K
    if tid == 0:
        island_state.n_islands[i_b] = n_islands
    qd.simt.block.sync()

    # Per-island link and dof counts
    i_t = tid
    while i_t < n_trees:
        i_island = island_state.trees_island_idx[i_t, i_b]
        if i_island >= 0:
            qd.atomic_add(island_state.dof_slices.n[i_island, i_b], rigid_info.trees_n_dofs[i_t])
            if qd.static(rigid_config.use_hibernation):
                qd.atomic_add(island_state.link_slices.n[i_island, i_b], rigid_info.trees_n_links[i_t])
        i_t = i_t + _K
    qd.simt.block.sync()

    # The list starts by a scan over the islands, the cursors of the fill starting there, the inertia and the sleeping
    # flag of every island from their neutral values.
    links_carry = 0
    dofs_carry = 0
    for i_chunk in range((n_islands + _K - 1) // _K):
        i_island = i_chunk * _K + tid
        n_links_island = 0
        n_dofs_island = 0
        if i_island < n_islands:
            n_dofs_island = island_state.dof_slices.n[i_island, i_b]
            if qd.static(rigid_config.use_hibernation):
                n_links_island = island_state.link_slices.n[i_island, i_b]
        links_incl, links_total = su.qd_block_scan(n_links_island)
        dofs_incl, dofs_total = su.qd_block_scan(n_dofs_island)
        if i_island < n_islands:
            link_list_start = links_carry + links_incl - n_links_island
            dof_list_start = dofs_carry + dofs_incl - n_dofs_island
            if qd.static(rigid_config.use_hibernation):
                island_state.link_slices.start[i_island, i_b] = link_list_start
                island_state.link_slices.curr[i_island, i_b] = link_list_start
            island_state.dof_slices.start[i_island, i_b] = dof_list_start
            island_state.dof_slices.curr[i_island, i_b] = dof_list_start
            island_state.inertia[i_island, i_b] = 0.0
            if qd.static(rigid_config.use_hibernation):
                island_state.is_hibernated[i_island, i_b] = 1
        links_carry = links_carry + links_total
        dofs_carry = dofs_carry + dofs_total
    qd.simt.block.sync()

    # The fill, per chunk of _K trees: a tree's items follow those of the earlier trees of its island, the ones of this
    # chunk counted over the chunk's trees, the ones of the earlier chunks held by the cursors. The links of a dof-less
    # tree take the label of no island.
    for i_chunk in range((n_trees + _K - 1) // _K):
        i_t = i_chunk * _K + tid
        i_island = -1
        if i_t < n_trees:
            i_island = island_state.trees_island_idx[i_t, i_b]
            i_root = rigid_info.trees_root_idx[i_t]
            i_link_pos = 0
            i_dof_pos = 0
            if i_island >= 0:
                i_dof_pos = island_state.dof_slices.curr[i_island, i_b]
                if qd.static(rigid_config.use_hibernation):
                    i_link_pos = island_state.link_slices.curr[i_island, i_b]
                for j_t in range(i_chunk * _K, i_t):
                    if island_state.trees_island_idx[j_t, i_b] == i_island:
                        i_link_pos = i_link_pos + rigid_info.trees_n_links[j_t]
                        i_dof_pos = i_dof_pos + rigid_info.trees_n_dofs[j_t]
            for i_tl in range(i_root, rigid_info.trees_link_end[i_t]):
                if rigid_info.links_tree_idx[i_tl] == i_t:
                    island_state.links_island_idx[i_tl, i_b] = i_island
                    if qd.static(rigid_config.use_hibernation):
                        if i_island >= 0:
                            island_state.link_id[i_link_pos, i_b] = i_tl
                            i_link_pos = i_link_pos + 1
            if i_island >= 0:
                dof_start_tree = rigid_info.trees_dof_start[i_t]
                dof_list_start = island_state.dof_slices.start[i_island, i_b]
                for i_d in range(dof_start_tree, dof_start_tree + rigid_info.trees_n_dofs[i_t]):
                    island_state.dof_id[i_dof_pos, i_b] = i_d
                    island_state.dof_local_pos[i_d, i_b] = i_dof_pos - dof_list_start
                    island_state.dofs_island_idx[i_d, i_b] = i_island
                    i_dof_pos = i_dof_pos + 1
        qd.simt.block.sync()
        if i_island >= 0:
            qd.atomic_add(island_state.dof_slices.curr[i_island, i_b], rigid_info.trees_n_dofs[i_t])
            if qd.static(rigid_config.use_hibernation):
                qd.atomic_add(island_state.link_slices.curr[i_island, i_b], rigid_info.trees_n_links[i_t])
        qd.simt.block.sync()
    i_island = tid
    while i_island < n_islands:
        island_state.dof_range_start[i_island, i_b] = func_dof_range_start(i_island, i_b, island_state)
        i_island = i_island + _K

    # An island is hibernated unless one of its links is awake, the sleeping links of an awake island then waking, see
    # func_build_islands
    if qd.static(rigid_config.use_hibernation):
        if has_sleepers:
            i_l = tid
            while i_l < n_links:
                i_island = island_state.links_island_idx[i_l, i_b]
                if i_island >= 0 and not dyn_state.links.is_hibernated[i_l, i_b]:
                    island_state.is_hibernated[i_island, i_b] = 0
                i_l = i_l + _K
            qd.simt.block.sync()
            n_awake_dofs_before = rigid_info.n_awake_dofs[i_b]
            i_l = tid
            while i_l < n_links:
                i_island = island_state.links_island_idx[i_l, i_b]
                if i_island >= 0 and island_state.is_hibernated[i_island, i_b] == 0:
                    func_wakeup_link(i_l, i_b, dyn_state, island_state, dyn_info, rigid_info, rigid_config)
                i_l = i_l + _K
            qd.simt.block.sync()
            if tid == 0:
                island_state.woke_sleepers[i_b] = rigid_info.n_awake_dofs[i_b] != n_awake_dofs_before
        else:
            i_island = tid
            while i_island < n_islands:
                island_state.is_hibernated[i_island, i_b] = 0
                i_island = i_island + _K

    # The inertia of every island, the mass diagonal summed over the dof list per chunk of _K dofs: the lanes of one
    # island reduce as one segment, whose last lane adds it into the island's total. The segments of an island come one
    # chunk after the other, so its total adds up in list order.
    for i_chunk in range((n_dofs + _K - 1) // _K):
        i_pos = i_chunk * _K + tid
        i_island = -1
        mass = gs.qd_float(0.0)
        if i_pos < n_dofs:
            i_d = island_state.dof_id[i_pos, i_b]
            i_island = island_state.dofs_island_idx[i_d, i_b]
            mass = rigid_info.mass_mat[i_d, i_d, i_b]
        i_island_prev, i_island_next = su.qd_slot_neighbors(tid, i_island)
        total, is_tail = su.qd_segmented_sum(tid, i_island, i_island_prev, i_island_next, mass)
        if is_tail:
            island_state.inertia[i_island, i_b] = island_state.inertia[i_island, i_b] + total
        qd.simt.block.sync()


@qd.func
def func_contact_island(
    i_col: int, i_b: int, collider_state: array_class.ColliderState, island_state: array_class.IslandState
):
    # A contact belongs to the island of its dof-carrying endpoint: both endpoints share an island when both carry dofs,
    # since the contact unioned them, otherwise one side is a fixed body.
    i_la = collider_state.contact_data.link_a[i_col, i_b]
    i_island = island_state.links_island_idx[i_la, i_b]
    if i_island < 0:
        i_lb = collider_state.contact_data.link_b[i_col, i_b]
        i_island = island_state.links_island_idx[i_lb, i_b]
    return i_island


@qd.func
def func_contact_tree_slots(
    i_col: int,
    i_b: int,
    collider_state: array_class.ColliderState,
    island_state: array_class.IslandState,
    reorder_state: array_class.IslandReorderState,
    rigid_info: array_class.RigidInfo,
):
    """Island-local tree slots (rcm_tree_pos) of a contact's two endpoints, -1 for a fixed / dof-less side."""
    i_la = collider_state.contact_data.link_a[i_col, i_b]
    i_lb = collider_state.contact_data.link_b[i_col, i_b]
    i_ta = rigid_info.links_tree_idx[i_la]
    i_tb = rigid_info.links_tree_idx[i_lb]
    if i_ta >= 0 and island_state.trees_island_idx[i_ta, i_b] >= 0:
        i_ta = reorder_state.rcm_tree_pos[i_ta, i_b]
    else:
        i_ta = -1
    if i_tb >= 0 and island_state.trees_island_idx[i_tb, i_b] >= 0:
        i_tb = reorder_state.rcm_tree_pos[i_tb, i_b]
    else:
        i_tb = -1
    return i_ta, i_tb


@qd.func
def func_reorder_island_dofs(
    i_b: int,
    collider_state: array_class.ColliderState,
    island_state: array_class.IslandState,
    reorder_state: array_class.IslandReorderState,
    rigid_info: array_class.RigidInfo,
):
    """Fill-reducing DOF reordering for the CPU per-island skyline path: rebuild each island's dof_id in reverse
    Cuthill-McKee order of its kinematic trees over the contact adjacency, instead of ascending tree order.

    The build-time DOF order says nothing about which bodies end up in contact, so a settled pile otherwise produces a
    near-dense skyline; ordering trees by contact adjacency shrinks the envelope the factor, rank-1 updates and
    triangular solves sweep. Trees stay contiguous with their DOFs in original relative order, preserving the local
    contiguity of the mass blocks. Every per-island consumer addresses nt_H through (dof_id, dof_local_pos) pairs with
    island-local triangle orientation, so a non-monotonic dof_id only changes where blocks are stored.
    """
    n_trees = rigid_info.trees_root_idx.shape[0]
    n_islands = island_state.n_islands[i_b]
    n_first_contact = collider_state.n_contacts_hibernated[i_b]
    n_contacts = collider_state.n_contacts[i_b]

    # Per-island contact lists (island -> contact ranges in contact_id) over the live contacts, in contact_sort_idx
    # order
    for i_island in range(n_islands):
        reorder_state.contact_slices.n[i_island, i_b] = 0
    for i_c in range(n_first_contact, n_contacts):
        i_island = func_contact_island(collider_state.contact_sort_idx[i_c, i_b], i_b, collider_state, island_state)
        if i_island >= 0:
            reorder_state.contact_slices.n[i_island, i_b] = reorder_state.contact_slices.n[i_island, i_b] + 1
    contact_list_start = 0
    for i_island in range(n_islands):
        reorder_state.contact_slices.start[i_island, i_b] = contact_list_start
        reorder_state.contact_slices.curr[i_island, i_b] = contact_list_start
        contact_list_start = contact_list_start + reorder_state.contact_slices.n[i_island, i_b]
    for i_c in range(n_first_contact, n_contacts):
        i_col = collider_state.contact_sort_idx[i_c, i_b]
        i_island = func_contact_island(i_col, i_b, collider_state, island_state)
        if i_island >= 0:
            reorder_state.contact_id[reorder_state.contact_slices.curr[i_island, i_b], i_b] = i_col
            reorder_state.contact_slices.curr[i_island, i_b] = reorder_state.contact_slices.curr[i_island, i_b] + 1

    # The per-island scratch rows live side by side: island i_island's slots start where the earlier islands' end
    tree_base = 0
    for i_island in range(n_islands):
        con_base = reorder_state.contact_slices.start[i_island, i_b]
        n_island_contacts = reorder_state.contact_slices.n[i_island, i_b]

        # Island-local tree slots, in ascending tree order
        n_island_trees = 0
        for i_t in range(n_trees):
            if island_state.trees_island_idx[i_t, i_b] == i_island:
                reorder_state.rcm_tree_pos[i_t, i_b] = n_island_trees
                n_island_trees = n_island_trees + 1

        # Reordering cannot shrink the envelope of at most two trees; keep the ascending build order there so small
        # islands stay bit-identical to the unordered path (and the common tiny-island case costs nothing).
        if n_island_trees <= 2:
            tree_base = tree_base + n_island_trees
            continue

        # Contact degree per tree (duplicate contacts inflate degrees, which only biases the tie-break)
        for i_slot in range(n_island_trees):
            reorder_state.rcm_tree_degree[tree_base + i_slot, i_b] = 0
            reorder_state.rcm_tree_is_ordered[tree_base + i_slot, i_b] = False
        for i_c_ in range(n_island_contacts):
            i_col = reorder_state.contact_id[con_base + i_c_, i_b]
            i_ta, i_tb = func_contact_tree_slots(i_col, i_b, collider_state, island_state, reorder_state, rigid_info)
            if i_ta >= 0 and i_tb >= 0 and i_ta != i_tb:
                reorder_state.rcm_tree_degree[tree_base + i_ta, i_b] = (
                    reorder_state.rcm_tree_degree[tree_base + i_ta, i_b] + 1
                )
                reorder_state.rcm_tree_degree[tree_base + i_tb, i_b] = (
                    reorder_state.rcm_tree_degree[tree_base + i_tb, i_b] + 1
                )

        # Cuthill-McKee: BFS from the lowest-degree unordered tree, appending each frontier sorted by degree
        n_ordered = 0
        i_head = 0
        while n_ordered < n_island_trees:
            i_slot_start = -1
            for i_slot in range(n_island_trees):
                if not reorder_state.rcm_tree_is_ordered[tree_base + i_slot, i_b] and (
                    i_slot_start == -1
                    or reorder_state.rcm_tree_degree[tree_base + i_slot, i_b]
                    < reorder_state.rcm_tree_degree[tree_base + i_slot_start, i_b]
                ):
                    i_slot_start = i_slot
            reorder_state.rcm_tree_order[tree_base + n_ordered, i_b] = i_slot_start
            reorder_state.rcm_tree_is_ordered[tree_base + i_slot_start, i_b] = True
            n_ordered = n_ordered + 1
            while i_head < n_ordered:
                i_slot_head = reorder_state.rcm_tree_order[tree_base + i_head, i_b]
                n_frontier_start = n_ordered
                for i_c_ in range(n_island_contacts):
                    i_col = reorder_state.contact_id[con_base + i_c_, i_b]
                    i_ta, i_tb = func_contact_tree_slots(
                        i_col, i_b, collider_state, island_state, reorder_state, rigid_info
                    )
                    i_slot_next = -1
                    if i_ta == i_slot_head and i_tb >= 0:
                        i_slot_next = i_tb
                    elif i_tb == i_slot_head and i_ta >= 0:
                        i_slot_next = i_ta
                    if i_slot_next >= 0 and not reorder_state.rcm_tree_is_ordered[tree_base + i_slot_next, i_b]:
                        # Insert into the current frontier keeping it sorted by ascending degree
                        i_ins = n_ordered
                        while i_ins > n_frontier_start and (
                            reorder_state.rcm_tree_degree[
                                tree_base + reorder_state.rcm_tree_order[tree_base + i_ins - 1, i_b], i_b
                            ]
                            > reorder_state.rcm_tree_degree[tree_base + i_slot_next, i_b]
                        ):
                            reorder_state.rcm_tree_order[tree_base + i_ins, i_b] = reorder_state.rcm_tree_order[
                                tree_base + i_ins - 1, i_b
                            ]
                            i_ins = i_ins - 1
                        reorder_state.rcm_tree_order[tree_base + i_ins, i_b] = i_slot_next
                        reorder_state.rcm_tree_is_ordered[tree_base + i_slot_next, i_b] = True
                        n_ordered = n_ordered + 1
                i_head = i_head + 1

        # Rebuild dof_id with trees in REVERSE Cuthill-McKee order, each tree's dofs in ascending order. The per-slot
        # tree scan is O(n_island_trees * n_trees), dominated by the BFS neighbor sweep above (O(n_island_trees *
        # n_island_contacts)), so it is not worth a per-slot tree buffer. The list leaves build order, so the sweeps
        # read it.
        island_state.dof_range_start[i_island, i_b] = -1
        i_dof_curr = island_state.dof_slices.start[i_island, i_b]
        for i_slot_ in range(n_island_trees):
            i_slot = reorder_state.rcm_tree_order[tree_base + (n_island_trees - 1 - i_slot_), i_b]
            for i_t in range(n_trees):
                if (
                    island_state.trees_island_idx[i_t, i_b] == i_island
                    and reorder_state.rcm_tree_pos[i_t, i_b] == i_slot
                ):
                    dof_start_tree = rigid_info.trees_dof_start[i_t]
                    for i_d in range(dof_start_tree, dof_start_tree + rigid_info.trees_n_dofs[i_t]):
                        island_state.dof_id[i_dof_curr, i_b] = i_d
                        island_state.dof_local_pos[i_d, i_b] = i_dof_curr - island_state.dof_slices.start[i_island, i_b]
                        i_dof_curr = i_dof_curr + 1
        tree_base = tree_base + n_island_trees


@qd.func
def func_sort_contacts_coop(
    i_b: int,
    tid: int,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
):
    """Sort the contacts of one env in the order of func_sort_contacts with the _K lanes of its block.

    Every contact takes the rank of its key among all of them, equal keys ranking by input position, which is the
    stable order the serial insertion sort yields.

    The input order moves to contact_sort_input and the position keys to contact_sort_key first, so the ranks scatter
    straight into contact_sort_idx.
    """
    _K = qd.static(32)
    n = collider_state.n_contacts[i_b]
    n_first = collider_state.n_contacts_hibernated[i_b]
    i_c = n_first + tid
    while i_c < n:
        i_col = collider_state.contact_sort_idx[i_c, i_b]
        geom_b = collider_state.contact_data.geom_b[i_col, i_b]
        pos = gu.qd_inv_transform_by_quat(
            collider_state.contact_data.pos[i_col, i_b] - dyn_state.geoms.pos[geom_b, i_b],
            dyn_state.geoms.quat[geom_b, i_b],
        )
        collider_state.contact_sort_input[i_c, i_b] = i_col
        collider_state.contact_sort_key[i_c, i_b] = func_contact_order_key(pos)
        i_c = i_c + _K
    qd.simt.block.sync()
    i_c = n_first + tid
    while i_c < n:
        i_col = collider_state.contact_sort_input[i_c, i_b]
        geom_a = collider_state.contact_data.geom_a[i_col, i_b]
        geom_b = collider_state.contact_data.geom_b[i_col, i_b]
        key = collider_state.contact_sort_key[i_c, i_b]
        rank = n_first
        for j_c in range(n_first, n):
            j_col = collider_state.contact_sort_input[j_c, i_b]
            geom_a_j = collider_state.contact_data.geom_a[j_col, i_b]
            precedes = geom_a_j < geom_a
            if not precedes and geom_a_j == geom_a:
                geom_b_j = collider_state.contact_data.geom_b[j_col, i_b]
                if geom_b_j < geom_b:
                    precedes = True
                elif geom_b_j == geom_b:
                    key_j = collider_state.contact_sort_key[j_c, i_b]
                    precedes = key_j < key or (key_j == key and j_c < i_c)
            if precedes:
                rank = rank + 1
        collider_state.contact_sort_idx[rank, i_b] = i_col
        i_c = i_c + _K


@qd.func
def func_sort_contacts(
    i_b: int,
    i_first: int,
    n: int,
    contact_idx: qd.Tensor,
    contacts_pos: qd.Tensor,
    contacts_geom_a: qd.Tensor,
    contacts_geom_b: qd.Tensor,
    geoms_pos: qd.Tensor,
    geoms_quat: qd.Tensor,
):
    """Insertion-sort the contact indices contact_idx[i_first : n] of one env by a deterministic total order.

    The order is (geom_a, geom_b, then the contact position along one direction in geom_b's own frame), a pure function
    of contact data, so it is independent of the racy atomic_add narrowphase layout. The position is taken in that frame
    because a world coordinate carries the orientation of the whole scene, which makes the order of a scene and of a
    rigidly rotated copy of it differ. geom_b carries the frame since the pair is canonically ordered by geom type and a
    plane, whose frame stays put while the scene turns, sorts first. Leading with the geom pair keeps each pair's
    contacts contiguous, so the frame is fixed across every position comparison the sort reaches; the position reduces
    to one scalar there, see func_contact_order_key.

    contact_idx is collider_state.contact_sort_idx; the contact-data tensors are passed as leaves rather than the whole
    collider_state struct so that it can be sorted in place without the struct-expansion aliasing its own field.
    """
    for i_s in range(i_first + 1, n):
        i_p = contact_idx[i_s, i_b]
        geom_a_p = contacts_geom_a[i_p, i_b]
        geom_b_p = contacts_geom_b[i_p, i_b]
        pos_p = gu.qd_inv_transform_by_quat(
            contacts_pos[i_p, i_b] - geoms_pos[geom_b_p, i_b], geoms_quat[geom_b_p, i_b]
        )
        j_s = i_s - 1
        while j_s >= i_first:
            i_q = contact_idx[j_s, i_b]
            geom_a_q = contacts_geom_a[i_q, i_b]
            geom_b_q = contacts_geom_b[i_q, i_b]
            precedes = geom_a_q < geom_a_p
            if not precedes and geom_a_q == geom_a_p:
                if geom_b_q < geom_b_p:
                    precedes = True
                elif geom_b_q == geom_b_p:
                    pos_q = gu.qd_inv_transform_by_quat(
                        contacts_pos[i_q, i_b] - geoms_pos[geom_b_q, i_b], geoms_quat[geom_b_q, i_b]
                    )
                    precedes = func_contact_order_key(pos_q) <= func_contact_order_key(pos_p)
            if precedes:
                break
            contact_idx[j_s + 1, i_b] = i_q
            j_s = j_s - 1
        contact_idx[j_s + 1, i_b] = i_p


@qd.func
def func_partition_islands(
    edges: qd.template(),
    dyn_state: array_class.DynState,
    island_state: array_class.IslandState,
    qd_n_equalities: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Build the island partition of every env and record whether it woke sleepers.

    Where the cooperative kernels run, a block serves each env (func_build_islands_coop); elsewhere one thread per env
    builds it (func_build_islands). Both ways give the same partition. The edges of the contacts come from the edge
    sources of the partition (see RigidIsland.partition_edges). A single-island scene writes its partition
    outright (func_build_single_island), off the CPU skyline path and in every env where nothing sleeps.
    """
    _B = dyn_state.dofs.ctrl_mode.shape[1]
    # Under hibernation the trivial partition serves the envs where nothing sleeps, the full build the others
    has_trivial_partition = qd.static(rigid_config.is_single_island and not rigid_config.sparse_solve)
    if qd.static(rigid_config.enable_cooperative_constraint_kernels):
        _K = qd.static(32)
        qd.loop_config(name="build_islands", block_dim=_K)
        for i_flat in range(_B * _K):
            tid = i_flat % _K
            i_b = i_flat // _K
            if qd.static(rigid_config.use_hibernation):
                if tid == 0:
                    island_state.woke_sleepers[i_b] = False
            if qd.static(has_trivial_partition):
                is_partition_trivial = True
                if qd.static(rigid_config.use_hibernation):
                    is_partition_trivial = rigid_info.n_awake_dofs[i_b] >= dyn_state.dofs.is_hibernated.shape[0]
                if is_partition_trivial:
                    func_build_single_island_coop(i_b, tid, island_state, rigid_info, rigid_config)
                else:
                    if qd.static(rigid_config.use_hibernation):
                        func_build_islands_coop(
                            i_b,
                            tid,
                            edges,
                            dyn_state,
                            island_state,
                            qd_n_equalities,
                            dyn_info,
                            rigid_info,
                            rigid_config,
                        )
            else:
                func_build_islands_coop(
                    i_b, tid, edges, dyn_state, island_state, qd_n_equalities, dyn_info, rigid_info, rigid_config
                )
    else:
        qd.loop_config(name="build_islands", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_b in range(_B):
            woke_sleepers = False
            if qd.static(has_trivial_partition):
                is_partition_trivial = True
                if qd.static(rigid_config.use_hibernation):
                    is_partition_trivial = rigid_info.n_awake_dofs[i_b] >= dyn_state.dofs.is_hibernated.shape[0]
                if is_partition_trivial:
                    func_build_single_island(i_b, island_state, rigid_info, rigid_config)
                else:
                    if qd.static(rigid_config.use_hibernation):
                        woke_sleepers = func_build_islands(
                            i_b, edges, dyn_state, island_state, qd_n_equalities, dyn_info, rigid_info, rigid_config
                        )
            else:
                woke_sleepers = func_build_islands(
                    i_b, edges, dyn_state, island_state, qd_n_equalities, dyn_info, rigid_info, rigid_config
                )
            if qd.static(rigid_config.use_hibernation):
                island_state.woke_sleepers[i_b] = woke_sleepers


@qd.func(requires_top_level=True)
def func_substep_partition_islands(
    island_data: qd.template(),
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    edges: qd.template(),
):
    func_partition_islands(
        edges,
        rigid_data.dyn_state,
        island_data.island,
        island_data.qd_n_equalities,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        rigid_config,
    )


@qd.func
def func_order_contacts(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    island_state: array_class.IslandState,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """Move the kept contacts of the sleepers the partition woke among the live contacts, then sort the live contacts
    in the order of func_sort_contacts, by the lanes of a block per env where the cooperative kernels run."""
    _B = dyn_state.dofs.ctrl_mode.shape[1]
    if qd.static(rigid_config.enable_cooperative_constraint_kernels):
        _K = qd.static(32)
        qd.loop_config(name="order_contacts", block_dim=_K)
        for i_flat in range(_B * _K):
            tid = i_flat % _K
            i_b = i_flat // _K
            if qd.static(rigid_config.use_hibernation):
                if tid == 0 and island_state.woke_sleepers[i_b]:
                    func_promote_woken_contacts(i_b, dyn_state, collider_state)
            if qd.static(collider_static_config.has_non_box_plane_convex_convex):
                qd.simt.block.sync()
                func_sort_contacts_coop(i_b, tid, dyn_state, collider_state)
    else:
        qd.loop_config(name="order_contacts", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_b in range(_B):
            if qd.static(rigid_config.use_hibernation):
                if island_state.woke_sleepers[i_b]:
                    func_promote_woken_contacts(i_b, dyn_state, collider_state)
            if qd.static(collider_static_config.has_non_box_plane_convex_convex):
                func_sort_contacts(
                    i_b,
                    collider_state.n_contacts_hibernated[i_b],
                    collider_state.n_contacts[i_b],
                    collider_state.contact_sort_idx,
                    collider_state.contact_data.pos,
                    collider_state.contact_data.geom_a,
                    collider_state.contact_data.geom_b,
                    dyn_state.geoms.pos,
                    dyn_state.geoms.quat,
                )


@qd.func(requires_top_level=True)
def func_substep_order_contacts(
    contact_data: qd.template(),
    island_data: qd.template(),
    rigid_data: qd.template(),
    rigid_config: qd.template(),
):
    func_order_contacts(
        rigid_data.dyn_state,
        contact_data.collider_state,
        island_data.island,
        rigid_config,
        contact_data.collider_config,
    )
