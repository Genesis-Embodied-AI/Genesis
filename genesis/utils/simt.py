import quadrants as qd


@qd.func
def qd_block_sum(value, log2_size: qd.template() = 5):
    """Sum a value over the lanes of a block, every lane receiving the bits the first lane of its block holds.

    A block is 2**log2_size consecutive lanes of the subgroup, so a subgroup wider than the block (64-lane AMDGPU
    wavefronts) holds several blocks, each summed on its own.
    """
    # The butterfly adds the same operands on the two lanes of every pair, but the compiler contracts the multiply that
    # produced a lane's own operand into that add, so the two lanes round differently, and a decision every lane takes
    # on the sum then diverges. The source lane differs between the blocks of a subgroup, which SPIR-V allows for a
    # shuffle alone, its broadcast requiring the same source lane on every lane of the subgroup.
    value_sum = qd.simt.subgroup.reduce_all_add_tiled(value, log2_size)
    return qd.simt.subgroup.shuffle(value_sum, qd.u32(qd.simt.subgroup.invocation_id() >> log2_size << log2_size))


@qd.func
def qd_block_scan(value):
    """Inclusive prefix sum of a value over the 32 lanes of a block, returning this lane's prefix and the block total.

    Every lane receives as total the bits the last lane of its block holds, so every lane carries the same running
    count across chunks. See qd_block_sum for the blocks and the source lane.
    """
    value_incl = qd.simt.subgroup.inclusive_add_tiled(value, 5)
    return value_incl, qd.simt.subgroup.shuffle(value_incl, qd.u32(qd.simt.subgroup.invocation_id() | 31))


@qd.func
def qd_block_min(value):
    """Minimum of a value over the 32 lanes of a block, every lane receiving the bits the first lane of its block holds.

    See qd_block_sum for the blocks and the source lane.
    """
    value_min = qd.simt.subgroup.reduce_all_min_tiled(value, 5)
    return qd.simt.subgroup.shuffle(value_min, qd.u32(qd.simt.subgroup.invocation_id() >> 5 << 5))


@qd.func
def qd_block_max(value):
    """Maximum of a value over the 32 lanes of a block, every lane receiving the bits the first lane of its block holds.

    See qd_block_sum for the blocks and the source lane.
    """
    value_max = qd.simt.subgroup.reduce_all_max_tiled(value, 5)
    return qd.simt.subgroup.shuffle(value_max, qd.u32(qd.simt.subgroup.invocation_id() >> 5 << 5))


@qd.func
def qd_lane_vector_get(v, i, block_dim: qd.template()):
    """Read entry i of a vector spread over the lanes of a block, every lane receiving it.

    Lane tid holds entries tid + k * block_dim of the vector in v[k], so the entry is broadcast from lane
    i % block_dim, which reads it from its register v[i // block_dim].
    """
    k_owner = i // block_dim
    v_owner = v[0]
    for k in qd.static(range(1, v.n)):
        if k == k_owner:
            v_owner = v[k]
    return qd.simt.subgroup.broadcast(v_owner, qd.u32(i - k_owner * block_dim))


@qd.func
def qd_slot_neighbors(tid, i_slot):
    """Slots of the two neighboring lanes of a 32-lane chunk, the inputs of qd_segment_bounds."""
    i_slot_prev = qd.simt.subgroup.shuffle_up(i_slot, qd.u32(1))
    i_slot_next = qd.simt.subgroup.shuffle_down(i_slot, qd.u32(1))
    return i_slot_prev, i_slot_next


@qd.func
def qd_segment_bounds(tid, i_slot, i_slot_prev, i_slot_next):
    """Whether lane tid of a 32-lane chunk opens and whether it closes the run of lanes sharing slot i_slot.

    i_slot_prev and i_slot_next are the slots of the neighboring lanes, a lane holding no item carrying slot -1. The
    chunk boundary closes every run, and the tail lane alone writes a segment out so segments accumulate in chunk order.
    """
    WARP_SIZE = qd.static(32)
    is_head = 1
    if tid > 0 and i_slot_prev == i_slot:
        is_head = 0
    is_tail = i_slot >= 0 and (tid == WARP_SIZE - 1 or i_slot_next != i_slot)
    return is_head, is_tail


@qd.func
def qd_segmented_sum(tid, i_slot, i_slot_prev, i_slot_next, value):
    """Sum value over the lanes of a 32-lane chunk that share slot i_slot.

    Returns the total and whether this lane is the segment's tail (see qd_segment_bounds).
    """
    is_head, is_tail = qd_segment_bounds(tid, i_slot, i_slot_prev, i_slot_next)
    total = qd.simt.subgroup.segmented_reduce_add_tiled(value, is_head, 5)
    return total, is_tail


@qd.func
def qd_segmented_min(tid, i_slot, i_slot_prev, i_slot_next, value):
    """Minimum of value over the lanes of a 32-lane chunk that share slot i_slot.

    Returns the minimum and whether this lane is the segment's tail, as qd_segmented_sum does.
    """
    is_head, is_tail = qd_segment_bounds(tid, i_slot, i_slot_prev, i_slot_next)
    total = qd.simt.subgroup.segmented_reduce_min_tiled(value, is_head, 5)
    return total, is_tail


@qd.func
def qd_segment_add(tid, i_slot, i_slot_prev, i_slot_next, i_row, value, sh_acc):
    """Add the segmented sum of value into entry i_row + i_slot of the shared array, from the tail lane of the segment.

    See qd_segmented_sum for the segments.
    """
    total, is_tail = qd_segmented_sum(tid, i_slot, i_slot_prev, i_slot_next, value)
    if is_tail:
        sh_acc[i_row + i_slot] = sh_acc[i_row + i_slot] + total


@qd.func
def qd_segment_min(tid, i_slot, i_slot_prev, i_slot_next, i_row, value, sh_min):
    """Fold the segmented minimum of value into entry i_row + i_slot of the shared array, from the tail lane of the
    segment.

    See qd_segmented_min for the segments.
    """
    total, is_tail = qd_segmented_min(tid, i_slot, i_slot_prev, i_slot_next, value)
    if is_tail:
        sh_min[i_row + i_slot] = qd.min(sh_min[i_row + i_slot], total)
