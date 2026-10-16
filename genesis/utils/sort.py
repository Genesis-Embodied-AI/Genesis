"""Sort of 64-bit keys per segment, with an optional value moved along each key.

The keys of segment i_s sit at keys[segments_start[i_s]:segments_start[i_s + 1]], and the keys of a segment exceed the
keys of every earlier segment, the segment index leading their bits. Sorting each segment is then the same as sorting
every key at once, which lets the device-wide sort of quadrants serve a set of many keys on GPU, while the sorts per
segment serve small segments and the host backend.
"""

import math
import os
from typing import NamedTuple

import quadrants as qd
from quadrants.algorithms import sort, sort_scratch_slots

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.linalg as lu
from genesis.utils.misc import get_gpu_core_count

# The rank sort, quadratic per segment, beats the radix sort up to a key count per segment. On the CPU the segments sort
# serially, so the bound holds at every segment count. On the GPU the passes of every segment run together and the
# crossover sits near one key per core for one segment, shrinking with the cube root of the segment count:
# n_segments * n_segment_keys**3 stays under the cube of this fraction of the core count, set under the crossover of
# every device measured.
RANK_SORT_MAX_KEYS_CPU = 256
RANK_SORT_KEYS_PER_CORE = 0.75
# Most chunks of the keys of one segment the host radix sort counts and scatters in parallel (see func_sort_segments).
# The chunks bring parallelism to a set of few segments. A set of many segments is parallel over its segments already,
# and every chunk costs a histogram of 256 bins per pass.
SORT_MAX_CHUNKS = 64


class SegmentSortData(NamedTuple):
    """A sort of keys per segment: its state and the static config of its kernels."""

    state: array_class.SegmentSortBuffer
    config: array_class.SegmentSortShape


def get_segment_sort_data(
    n_segments: int, n_keys: int, n_segment_keys_max: int, key_bits: int, has_values: bool, is_active: bool = True
) -> SegmentSortData:
    """Return the buffers and the static config of a sort of keys per segment.

    The sort orders n_keys keys over n_segments segments. A segment holds n_segment_keys_max keys at most, and every
    key, its segment index included, fits in key_bits bits.
    The sort takes the rank sort for small segments, a radix sort per segment on the host backend, and the device-wide
    radix sort of quadrants otherwise. A sort switched off takes the empty shape on every tensor (see maybe_shape in
    array_class.py).
    """
    if key_bits > 64:
        gs.raise_exception(f"A segment sort orders 64-bit keys, its keys need {key_bits} bits.")

    # The radix sort orders a positive even multiple of 8 bits
    end_bit = min(64, 16 * math.ceil(key_bits / 16))
    log256_max_n = 1
    while 256**log256_max_n < n_keys:
        log256_max_n += 1
    if gs.backend == gs.cpu:
        is_rank_sort = n_segment_keys_max <= RANK_SORT_MAX_KEYS_CPU
    else:
        max_keys = RANK_SORT_KEYS_PER_CORE * get_gpu_core_count()
        is_rank_sort = n_segments * n_segment_keys_max**3 <= max_keys**3
    # Outside performance mode the radix sort serves every set, with the widest keys and a scan depth reaching any key
    # count a device holds. The choice of the rank sort stays per set, its single pass being much faster on small sets.
    if gs.use_ndarray:
        end_bit = 64
        log256_max_n = max(log256_max_n, 4)
    if is_rank_sort:
        sort_kind = array_class.SEGMENT_SORT_KIND.RANK
    elif gs.backend == gs.cpu:
        sort_kind = array_class.SEGMENT_SORT_KIND.SEGMENT_RADIX
    else:
        sort_kind = array_class.SEGMENT_SORT_KIND.DEVICE_RADIX
    sort_config = array_class.SegmentSortShape(
        end_bit=end_bit, log256_max_n=log256_max_n, sort_kind=sort_kind, has_values=has_values
    )

    n_sort_scratch = (
        sort_scratch_slots(n_keys, log256_max_n) if sort_kind == array_class.SEGMENT_SORT_KIND.DEVICE_RADIX else 0
    )
    n_sort_chunks = max(1, min(SORT_MAX_CHUNKS, (os.cpu_count() or 1) // n_segments))
    sort_state = array_class.get_segment_sort_buffer(
        n_segments, n_keys, n_sort_scratch, n_sort_chunks, sort_config, is_active
    )
    return SegmentSortData(sort_state, sort_config)


@qd.func
def func_sort_segments(sort_state: array_class.SegmentSortBuffer, sort_config: qd.template()):
    """Sort the keys of every segment in increasing order, with their values, at the top level of a kernel.

    The caller writes the keys, their values, the segment bounds and the key count first. Equal keys keep their input
    order, so the result is a pure function of the input.
    """
    if qd.static(sort_config.sort_kind == array_class.SEGMENT_SORT_KIND.DEVICE_RADIX):
        # Without values, the keys stand in for the placeholder values the sort leaves untouched
        sort(
            sort_state.keys,
            sort_state.keys_scratch,
            sort_state.values if qd.static(sort_config.has_values) else sort_state.keys,
            sort_state.values_scratch if qd.static(sort_config.has_values) else sort_state.keys_scratch,
            sort_state.sort_scratch,
            sort_state.n_keys,
            key_dtype=qd.u64,
            has_values=qd.static(sort_config.has_values),
            end_bit=qd.static(sort_config.end_bit),
            log256_max_n=qd.static(sort_config.log256_max_n),
        )
    elif qd.static(sort_config.sort_kind == array_class.SEGMENT_SORT_KIND.RANK):
        # Each key ranks itself among the keys of its segment, found by a binary search over the segment bounds
        n_segments = sort_state.segments_start.shape[0] - 1
        n_search_steps = 32 - qd.math.clz(qd.u32(n_segments))
        for i_k in range(sort_state.n_keys[()]):
            i_seg_lo = 0
            i_seg_hi = n_segments
            for _ in range(n_search_steps):
                if i_seg_hi - i_seg_lo > 1:
                    i_seg_mid = (i_seg_lo + i_seg_hi) // 2
                    if sort_state.segments_start[i_seg_mid] <= i_k:
                        i_seg_lo = i_seg_mid
                    else:
                        i_seg_hi = i_seg_mid
            i_k_start = sort_state.segments_start[i_seg_lo]
            key = sort_state.keys[i_k]
            rank = 0
            for i_o in range(i_k_start, sort_state.segments_start[i_seg_lo + 1]):
                key_o = sort_state.keys[i_o]
                if key_o < key or (key_o == key and i_o < i_k):
                    rank += 1
            sort_state.keys_scratch[i_k_start + rank] = key
            if qd.static(sort_config.has_values):
                sort_state.values_scratch[i_k_start + rank] = sort_state.values[i_k]
        for i_k in range(sort_state.n_keys[()]):
            sort_state.keys[i_k] = sort_state.keys_scratch[i_k]
            if qd.static(sort_config.has_values):
                sort_state.values[i_k] = sort_state.values_scratch[i_k]
    else:
        # A least-significant-digit radix sort of eight bits per pass on the host backend, the keys of each segment
        # split over chunks: the chunks count their digits and scatter their keys in parallel, one thread per segment
        # turning the counts into offsets in between. The passes ping-pong between the keys and the scratch and end in
        # the keys. The passes unroll statically: a loop of the kernel is parallel at its top level alone, so a runtime
        # loop over the passes would serialize the chunks.
        n_segments = sort_state.segments_start.shape[0] - 1
        n_sort_chunks = sort_state.sort_hist.shape[1]
        for i_pass in qd.static(range(qd.static(sort_config.end_bit) // 8)):
            shift = 8 * i_pass
            for i_seg, i_c in qd.ndrange(n_segments, n_sort_chunks):
                for i_d in range(256):
                    sort_state.sort_hist[i_seg, i_c, i_d] = 0
                i_k_start = sort_state.segments_start[i_seg]
                n_segment_keys = sort_state.segments_start[i_seg + 1] - i_k_start
                i_k_start, i_k_end = lu.func_chunk_range(i_k_start, i_c, n_segment_keys, n_sort_chunks)
                for i_k in range(i_k_start, i_k_end):
                    key = qd.u64(0)
                    if qd.static(i_pass % 2 == 0):
                        key = sort_state.keys[i_k]
                    else:
                        key = sort_state.keys_scratch[i_k]
                    digit = qd.i32((key >> shift) & qd.u64(0xFF))
                    sort_state.sort_hist[i_seg, i_c, digit] += 1
            for i_seg in range(n_segments):
                total = sort_state.segments_start[i_seg]
                for i_d in range(256):
                    for i_c in range(n_sort_chunks):
                        count = sort_state.sort_hist[i_seg, i_c, i_d]
                        sort_state.sort_hist[i_seg, i_c, i_d] = total
                        total += count
            # The keys scattered in order within each chunk, so that the pass is stable
            for i_seg, i_c in qd.ndrange(n_segments, n_sort_chunks):
                i_k_start = sort_state.segments_start[i_seg]
                n_segment_keys = sort_state.segments_start[i_seg + 1] - i_k_start
                i_k_start, i_k_end = lu.func_chunk_range(i_k_start, i_c, n_segment_keys, n_sort_chunks)
                for i_k in range(i_k_start, i_k_end):
                    key = qd.u64(0)
                    if qd.static(i_pass % 2 == 0):
                        key = sort_state.keys[i_k]
                    else:
                        key = sort_state.keys_scratch[i_k]
                    digit = qd.i32((key >> shift) & qd.u64(0xFF))
                    i_dst = sort_state.sort_hist[i_seg, i_c, digit]
                    sort_state.sort_hist[i_seg, i_c, digit] = i_dst + 1
                    if qd.static(i_pass % 2 == 0):
                        sort_state.keys_scratch[i_dst] = key
                        if qd.static(sort_config.has_values):
                            sort_state.values_scratch[i_dst] = sort_state.values[i_k]
                    else:
                        sort_state.keys[i_dst] = key
                        if qd.static(sort_config.has_values):
                            sort_state.values[i_dst] = sort_state.values_scratch[i_k]
