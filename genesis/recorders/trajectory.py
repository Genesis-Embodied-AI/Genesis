"""Log the state of a scene as it steps, one frame per step, and read the log back to seek and replay it.

A trajectory file holds the scene as 'Scene.export' writes it, then its frames in compressed chunks. Every part carries
its own length and checksum, so a run that crashes leaves every chunk written before readable, and a reader stops at the
first part it cannot trust. A checkpoint file (see 'Scene.save_checkpoint') is a trajectory file of one frame holding
every array, scratch included.
"""

import bisect
import dataclasses
import enum
import io
import json
import logging
import os
import queue
import struct
import threading
import time
import zlib
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import numba as nb
import zstandard

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.base_solver import StateChange, Subscriber
from genesis.engine.states.solvers import SimulatorCheckpoint, SolverCheckpoint
from genesis.options.recorders import TrajectoryFile
from genesis.utils import serialization
from genesis.utils.array_class import ArrayItem, ConfigItem, DataKind, ScalarItem
from genesis.utils.misc import data_to_array, tensor_to_array

from .file_writers import BaseFileWriter
from .recorder_manager import register_recording

if TYPE_CHECKING:
    from genesis.engine.scene import Scene, TrajectorySource
    from genesis.engine.simulator import Simulator
    from genesis.options.renderers import RendererOptions
    from genesis.options.vis import ViewerOptions, VisOptions

LOGGER = logging.getLogger(__name__)

TRAJECTORY_FORMAT = ".gstraj"
MAGIC = b"GSTRAJ4\n"
# The magic of any version of the format, which a file from another version shares up to its version number
MAGIC_PREFIX = b"GSTRAJ"
# The fastest level of zstd, so that the compression thread keeps pace with the fastest stepping scene. The levels
# above shrink frame differences by a few percent only.
ZSTD_LEVEL = 1
# The frames a writer holds at once: the one being filled, the one the compression thread takes the next difference
# against, the one it holds back for the stop to drop (see 'cleanup'), and one more queued so that a compression slower
# than a step for a while does not hold the step back. The file is written by a thread of its own, so a slow disk delays
# the records only, never the release of a frame.
N_FRAME_SLOTS = 4
# The bytes of the completed chunks waiting for the I/O thread at most, one chunk being let through whatever its size. A
# chunk holds many frames compressed, so this absorbs a slow disk far more cheaply than frame slots would, and the
# compression thread waits once it is reached. Bytes rather than chunks, since a chunk grows with the environments.
MAX_PENDING_BYTES = 2**28
CHUNK_MAGIC = b"CHNK"
# A chunk record: frame count, payload length, payload checksum.
CHUNK_HEADER = struct.Struct("<IQI")
INFO_MAGIC = b"INFO"
# A record of the model parameters the frames from a given one on hold: that frame, payload length, payload checksum.
INFO_HEADER = struct.Struct("<QQI")
LAST_MAGIC = b"LAST"
# The record a trajectory ends with: payload length, payload checksum.
LAST_HEADER = struct.Struct("<QI")
LENGTH = struct.Struct("<Q")
# What a frame holds in each mode (see TrajectoryFile). The model parameters change through setters alone, so they go in
# a record of their own whenever one changes them (see INFO_MAGIC). A checkpoint file and the record a trajectory ends
# with hold every array but the static configs and constants, the scratch of the solvers included. They serve the
# analysis of a failure, and a failure may depend on the platform and never reproduce once shared. What the last step
# computed is therefore kept whole.
MODES_KINDS = {
    "exact": frozenset((DataKind.STATE, DataKind.DERIVED)),
    "minimal": frozenset((DataKind.STATE,)),
}
INFO_KINDS = frozenset((DataKind.INFO,))
CHECKPOINT_FILE_KINDS = frozenset(DataKind) - {DataKind.CONSTANT}
# The largest frame a mode is resolved to (see TrajectoryFile). Every recorded step copies a whole frame, and the
# compression thread goes through it, so a larger frame holds the step back once the compression falls behind.
MAX_FRAME_BYTES = 2**22
# A chunk stores each frame as its difference with the previous frame of the chunk, block-sparse: a bitmap of the blocks
# of BLOCK_SIZE bytes holding a non-zero byte, then these blocks alone. A step changes a small part of a frame, so the
# compressor goes through that part alone.
BLOCK_SIZE = 256
# The host-side end of a frame starts on a whole byte of the bitmap, so that the device encodes the rest on its own
FRAME_ALIGNMENT = 8 * BLOCK_SIZE
# The name a frame holds the step count of each environment under (see 'Simulator.data')
STEPS_FIELD = "Simulator.steps"
# The wall time of each phase of the step a frame was reached by, which closes a frame (see 'StepTimer.timings')
TIMINGS_PREFIX = "timings/"


class FrameField(NamedTuple):
    """Where one array stands in a frame: its name, kind, canonical shape, numpy dtype string, byte offset, byte length
    and memory order.

    The memory order lists the canonical axes from the slowest-varying to the fastest-varying one. A frame stores each
    array in the memory order of the scene that wrote it, so that a layout-tagged array is copied as it lies.
    """

    name: str
    kind: DataKind
    shape: tuple[int, ...]
    dtype: str
    offset: int
    nbytes: int
    axes: tuple[int, ...]


class InfoRecord(NamedTuple):
    """The model parameters of the frames from a given one on, as a flat uint8 buffer.

    The payload is laid out as 'info_fields' states it in the manifest (see '_arrays_fields').
    """

    i_frame: int
    payload: np.ndarray


def _json_value(value):
    """Return a value as JSON holds it: the name of an enum member, a native for a numpy scalar, a list for a tuple."""
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _fields_to_json(fields: list[FrameField]) -> list:
    """Return the frame layout as the JSON rows the file header holds."""
    return [
        [field.name, field.kind.name, list(field.shape), field.dtype, field.offset, field.nbytes, list(field.axes)]
        for field in fields
    ]


def _fields_from_json(rows: list) -> list[FrameField]:
    """Return the frame layout the JSON rows of a file header hold."""
    return [
        FrameField(name, DataKind[kind], tuple(shape), dtype, offset, nbytes, tuple(axes))
        for name, kind, shape, dtype, offset, nbytes, axes in rows
    ]


def _frame_arrays(frame: np.ndarray, fields: list[FrameField], kinds: frozenset[DataKind]) -> dict[str, np.ndarray]:
    """Return the arrays of the given kinds that a frame holds, by name, shaped as the scene holds them."""
    arrays = {}
    for field in fields:
        if field.kind in kinds:
            array = frame[field.offset : field.offset + field.nbytes].view(field.dtype)
            arrays[field.name] = array.reshape([field.shape[i] for i in field.axes]).transpose(np.argsort(field.axes))
    return arrays


def _arrays_fields(sim: "Simulator", kinds: frozenset[DataKind], drop_unused: bool) -> list[FrameField]:
    """Lay out the arrays of the given kinds back to back, from the start of a frame.

    With 'drop_unused', an unused array is listed empty. Every unused array of the kinds a step frame holds is the
    placeholder of a feature switched off (see 'maybe_shape'), which a restore leaves as built (see 'check_data').

    Where zero-copy views exist, an array keeps the memory order of its view, a permutation of its canonical axes, so
    that a frame is copied from the views as they lie (see 'TrajectoryFileWriter.build'). Otherwise it is canonical.
    """
    fields = []
    offset = 0
    for item in sim.data(kinds):
        if isinstance(item, ArrayItem):
            array = data_to_array(item.array)
            if drop_unused and not item.array.shape and not isinstance(item.array, torch.Tensor):
                fields.append(FrameField(item.name, item.kind, (0,), array.dtype.str, offset, 0, (0,)))
                continue
            axes = tuple(range(array.ndim))
            if gs.use_zerocopy and item.array.shape:
                view = data_to_array(item.array, to_torch=True, copy=False)
                # Sorting is stable, so the axes of equal strides, which only a size-1 axis can share, keep their order
                axes = tuple(sorted(axes, key=view.stride, reverse=True))
                assert view.permute(axes).is_contiguous()
            fields.append(FrameField(item.name, item.kind, array.shape, array.dtype.str, offset, array.nbytes, axes))
            offset += array.nbytes
    return fields


def _scalars_fields(sim: "Simulator", kinds: frozenset[DataKind], offset: int) -> list[FrameField]:
    """Lay out the scalars of the given kinds back to back, from the given byte offset (see 'ScalarItem')."""
    fields = []
    for item in sim.data(kinds):
        if isinstance(item, ScalarItem):
            dtype = np.dtype(type(getattr(item.owner, item.attr)))
            fields.append(FrameField(item.name, item.kind, (), dtype.str, offset, dtype.itemsize, ()))
            offset += dtype.itemsize
    return fields


def _read_scalars(scalars: list[ScalarItem], scalars_fields: list[FrameField]) -> np.ndarray:
    """Return the given scalars as a flat uint8 buffer, laid out as their fields state.

    A frame is laid out once, so a scalar whose type changed since raises.
    """
    parts = [np.empty((0,), dtype=np.uint8)]
    for item, field in zip(scalars, scalars_fields):
        scalar = np.array(getattr(item.owner, item.attr))
        if scalar.dtype.str != field.dtype:
            gs.raise_exception(
                f"'{field.name}' holds a {scalar.dtype} value, where a {np.dtype(field.dtype)} one was laid out."
            )
        parts.append(scalar.reshape(-1).view(np.uint8))
    return np.concatenate(parts)


def _frame_fields(sim: "Simulator", kinds: frozenset[DataKind], drop_unused: bool) -> tuple[list[FrameField], int]:
    """Lay out one frame of the given kinds (see '_read_frame'), and return its fields and its size.

    A frame holds every array (see '_arrays_fields'), the step count of each environment included, then the scalars
    (see 'ScalarItem') and the wall time of every phase of a step. The scalars and the timings live on the host, so
    they start on a multiple of FRAME_ALIGNMENT, and so does the size of the frame. The bytes in between are zeros.
    """
    fields = _arrays_fields(sim, kinds, drop_unused)
    offset = sum(field.nbytes for field in fields)
    offset = -(-offset // FRAME_ALIGNMENT) * FRAME_ALIGNMENT
    scalars_fields = _scalars_fields(sim, kinds, offset)
    fields += scalars_fields
    offset += sum(field.nbytes for field in scalars_fields)
    for phase in sim.step_timer.phases:
        fields.append(FrameField(f"{TIMINGS_PREFIX}{phase}", DataKind.STATE, (), "<f8", offset, 8, ()))
        offset += 8
    return fields, -(-offset // FRAME_ALIGNMENT) * FRAME_ALIGNMENT


def _frame_tail(sim: "Simulator", scalars: list[ScalarItem], scalars_fields: list[FrameField]) -> np.ndarray:
    """Return the end of a frame that lives on the host, as a flat uint8 buffer.

    It holds the given scalars, laid out as their fields state, then the wall time of every phase of the step the state
    was reached by, NaN for a phase that step did not time.
    """
    phases = sim.step_timer.phases
    step_timings = sim.step_timer.timings
    # A phase timed without being declared would be lost, as the frame is laid out once (see 'StepTimer.add_phases')
    assert step_timings.keys() <= set(phases)
    timings = np.array([step_timings.get(phase, np.nan) for phase in phases])
    return np.concatenate((_read_scalars(scalars, scalars_fields), timings.view(np.uint8)))


def _read_arrays(arrays: dict[str, ArrayItem], fields: list[FrameField]) -> np.ndarray:
    """Return the given arrays as a flat uint8 buffer, laid out as their fields state (see '_arrays_fields').

    Where zero-copy views exist the arrays are concatenated on the device and cross to the host once. Otherwise each
    array crosses on its own, to be concatenated on the host.
    """
    parts = [(arrays[field.name].array, field.axes) for field in fields if field.name in arrays and field.nbytes]
    if not parts:
        return np.empty((0,), dtype=np.uint8)
    if gs.use_zerocopy:
        views = [
            data_to_array(array, to_torch=True).permute(axes).reshape(-1).view(torch.uint8) for array, axes in parts
        ]
        return tensor_to_array(torch.cat(views))
    return np.concatenate([data_to_array(array).transpose(axes).reshape(-1).view(np.uint8) for array, axes in parts])


def _read_frame(sim: "Simulator", kinds: frozenset[DataKind], fields: list[FrameField], frame_size: int) -> np.ndarray:
    """Return one frame of the given kinds as a flat uint8 buffer laid out as 'fields' states it.

    See '_frame_fields'.
    """
    items = list(sim.data(kinds))
    arrays = {item.name: item for item in items if isinstance(item, ArrayItem)}
    scalars = [item for item in items if isinstance(item, ScalarItem)]
    # Every array of the given kinds has a field, and these fields come first, followed by the ones of the scalars in
    # the order of their items
    device_part = _read_arrays(arrays, fields[: len(arrays)])
    tail = _frame_tail(sim, scalars, fields[len(arrays) : len(arrays) + len(scalars)])
    tail_offset = fields[len(arrays)].offset
    frame = np.zeros((frame_size,), dtype=np.uint8)
    frame[: device_part.size] = device_part
    frame[tail_offset : tail_offset + tail.size] = tail
    return frame


@nb.jit(nopython=True, cache=True, nogil=True)
def _encode_blocks(words: np.ndarray, words_prev: np.ndarray, is_delta: bool, bitmap: np.ndarray, blocks: np.ndarray):
    """Encode a frame block-sparse (see BLOCK_SIZE), or its difference with the previous one.

    The difference is encoded if 'is_delta' is set. 'words' and 'words_prev' are the two frames viewed as uint64. Write
    into 'bitmap' one bit per block, set if it holds a non-zero word, the lowest bit first, and into 'blocks' the words
    of these blocks back to back. Return their count.
    """
    n_words = BLOCK_SIZE // 8
    n_blocks = 0
    for i_byte in range(bitmap.size):
        byte = 0
        for i_bit in range(8):
            i_word_start = (8 * i_byte + i_bit) * n_words
            is_nonzero = False
            for i_word in range(n_words):
                word = words[i_word_start + i_word]
                if is_delta:
                    word ^= words_prev[i_word_start + i_word]
                blocks[n_blocks * n_words + i_word] = word
                is_nonzero |= word != 0
            if is_nonzero:
                byte |= 1 << i_bit
                n_blocks += 1
        bitmap[i_byte] = byte
    return n_blocks


def _header(
    source: "TrajectorySource",
    mode: str,
    chunk_size: int,
    fields: list[FrameField],
    frame_size: int,
    info_fields: list[FrameField],
    last_fields: list[FrameField],
    last_frame_size: int,
) -> bytes:
    """Return the magic, the JSON manifest and the scene description that open a trajectory file.

    'fields' lays out its frames of 'frame_size' bytes, 'info_fields' its records of model parameters and 'last_fields'
    the record of 'last_frame_size' bytes it ends with.
    """
    sim, desc, layout = source
    description = io.BytesIO()
    serialization.export(description, {"scene": desc})
    manifest = {
        "genesis": gs.__version__,
        "source": serialization.source_digest(),
        "backend": gs.backend.name,
        "precision": np.dtype(gs.np_float).name,
        "mode": mode,
        "chunk_size": chunk_size,
        "dt": sim.dt,
        "layout": {key: _json_value(value) for key, value in dataclasses.asdict(layout).items()},
        "configs": {
            item.name: {key: _json_value(value) for key, value in vars(item.config).items() if not key.startswith("_")}
            for solver in sim.active_solvers
            for item in solver.data
            if isinstance(item, ConfigItem)
        },
        "fields": _fields_to_json(fields),
        "frame_size": frame_size,
        "info_fields": _fields_to_json(info_fields),
        "last_fields": _fields_to_json(last_fields),
        "last_frame_size": last_frame_size,
    }
    text = json.dumps(manifest).encode()
    return MAGIC + LENGTH.pack(len(text)) + text + LENGTH.pack(description.getbuffer().nbytes) + description.getvalue()


def save_checkpoint(path: str | os.PathLike, source: "TrajectorySource") -> None:
    """Write the whole state of a built scene as a trajectory file of one frame, the record a trajectory ends with (see
    'Scene.save_checkpoint').
    """
    fields, frame_size = _frame_fields(source.sim, CHECKPOINT_FILE_KINDS, drop_unused=False)
    with open(path, "wb") as file:
        file.write(
            _header(
                source,
                mode="exact",
                chunk_size=1,
                fields=fields,
                frame_size=frame_size,
                info_fields=[],
                last_fields=fields,
                last_frame_size=frame_size,
            )
        )
        payload = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(
            _read_frame(source.sim, CHECKPOINT_FILE_KINDS, fields, frame_size)
        )
        file.write(LAST_MAGIC + LAST_HEADER.pack(len(payload), zlib.crc32(payload)) + payload)


@register_recording(TrajectoryFile)
class TrajectoryFileWriter(BaseFileWriter):
    """Write the frames of a scene to a trajectory file (see TrajectoryFile), one chunk at a time.

    Each frame captures the state a step starts from, control inputs included, since it is read before the step. The
    stepping thread copies each frame into one of a few frame slots allocated at build and hands it to a compression
    thread, which streams it through the compressor as its XOR with the previous frame of its chunk. Unchanged bytes
    thus become zeros the compressor removes. On a GPU backend the device takes the XOR before the copy, which spares
    the compression thread a pass over the frame. A completed chunk goes to a third thread, which appends it to the
    file. A step costs one copy of the frame, issued without waiting for a GPU to finish the step: the compression
    thread waits for the copy to land instead. The step waits only when every slot is taken, which happens when the
    frames are compressed slower than the scene steps. The model parameters change through setters alone, so they are
    recorded ahead of the first frame and of the first frame after a setter changed them, in a record of their own.
    """

    def build(self):
        self._source: TrajectorySource = self._data_func()
        sim = self._source.sim
        # The frames of the steps leave the unused arrays out, as they are read once per step. The last record holds
        # them, with the scratch.
        # The exact frame holds the kinds of every mode, so it sizes them all.
        exact_fields, _ = _frame_fields(sim, MODES_KINDS["exact"], drop_unused=True)
        modes_frame_size = {
            mode: sum(field.nbytes for field in exact_fields if field.kind in kinds)
            for mode, kinds in MODES_KINDS.items()
        }
        self._mode = self._options.mode
        if self._mode is None:
            self._mode = "exact" if modes_frame_size["exact"] <= MAX_FRAME_BYTES else "minimal"
        elif self._mode == "exact" and modes_frame_size["exact"] > MAX_FRAME_BYTES:
            gs.logger.warning(
                f"[TrajectoryFileWriter] Recording in exact mode copies {modes_frame_size['exact'] / 1e6:.1f} MB at "
                f"every recorded step, which slows the simulation down a lot. A minimal frame holds "
                f"{modes_frame_size['minimal'] / 1e6:.1f} MB (see TrajectoryFile)."
            )
        self._kinds = MODES_KINDS[self._mode]
        self._fields, self._frame_size = _frame_fields(sim, self._kinds, drop_unused=True)
        self._last_fields, self._last_frame_size = _frame_fields(sim, CHECKPOINT_FILE_KINDS, drop_unused=False)
        # The model parameters are recorded with the first frame, then with the first frame after a setter changes them
        self._info_fields = _arrays_fields(sim, INFO_KINDS, drop_unused=True)
        self._info_scalars_fields = _scalars_fields(sim, INFO_KINDS, sum(field.nbytes for field in self._info_fields))
        self._info_fields += self._info_scalars_fields
        # The data of the solvers is walked once, as walking it costs more than a step of a small scene
        info_items = list(sim.data(INFO_KINDS))
        self._info_arrays = {item.name: item for item in info_items if isinstance(item, ArrayItem)}
        self._info_scalars = [item for item in info_items if isinstance(item, ScalarItem)]
        self._has_info = any(field.nbytes for field in self._info_fields)
        self._info_subscriber = Subscriber(frozenset((StateChange.MODEL,)))
        for solver in sim.active_solvers:
            solver.subscribe(self._info_subscriber)
        self._i_frame = 0
        self._steps_field = next(field for field in self._fields if field.name == STEPS_FIELD)
        # The scalars and the step timings close the frame (see '_frame_fields'). They live on the host, so a frame copy
        # stops short of them and they are written beside it.
        items = list(sim.data(self._kinds))
        arrays = {item.name: item.array for item in items if isinstance(item, ArrayItem)}
        self._device_end = sum(field.nbytes for field in self._fields if field.name in arrays)
        self._tail_offset = self._fields[len(arrays)].offset
        self._scalars = [item for item in items if isinstance(item, ScalarItem)]
        self._scalars_fields = self._fields[len(arrays) : len(arrays) + len(self._scalars)]

        # The arrays of a built scene keep their memory, so the byte view each array is copied from is taken once, in
        # the memory order the frame stores it in (see '_frame_fields'). Without zero-copy views each frame is read
        # whole (see '_read_frame').
        # FIXME: quadrants#970 - a 0-d field has no zero-copy view, and reading one per frame runs a copy kernel. The
        # step frames leave out the 0-d arrays (see '_frame_fields'), whose content the step frames therefore lack.
        self._frame_views: list[np.ndarray | torch.Tensor] | None = None
        self._staging: torch.Tensor | None = None
        # On CUDA the device takes the difference of a frame with the previous one and gathers the blocks it changed
        # (see '_get_frame'), so that the frame copy carries them alone
        self._is_device_encoded = gs.use_zerocopy and gs.backend == gs.cuda
        self._device_blocks: torch.Tensor | None = None
        self._slots_mask_tc: torch.Tensor | None = None
        self._slots_mask: np.ndarray | None = None
        self._copy_stream: torch.cuda.Stream | None = None
        if gs.use_zerocopy:
            self._frame_views = []
            for field in self._fields[: len(arrays)]:
                if field.nbytes:
                    array = data_to_array(arrays[field.name], to_torch=True, copy=False)
                    view = array.permute(field.axes).view(-1).view(torch.uint8)
                    # The views of the CPU backend are host memory already, and numpy concatenates them on the calling
                    # thread. Torch would run the copy on its thread pool, whose wake-up costs more than the copy.
                    self._frame_views.append(view.numpy() if gs.backend == gs.cpu else view)
            if gs.backend != gs.cpu:
                # The two latest frames in turn, the one before the latest taking its difference with it (see
                # '_get_frame'). The bytes between the arrays and the scalars stay zeros.
                self._staging = torch.zeros((2, self._tail_offset), dtype=torch.uint8, device=gs.device)
        if self._is_device_encoded:
            # The changed blocks of each frame, gathered on the device, and which blocks they are on the host
            n_device_blocks = self._tail_offset // BLOCK_SIZE
            self._device_blocks = torch.empty(
                (N_FRAME_SLOTS, n_device_blocks, BLOCK_SIZE // 8), dtype=torch.int64, device=gs.device
            )
            slots_mask = torch.empty((N_FRAME_SLOTS, n_device_blocks), dtype=torch.bool, pin_memory=True)
            self._slots_mask_tc = slots_mask
            self._slots_mask = slots_mask.numpy()
            self._copy_stream = torch.cuda.Stream()

        # Page-locked on CUDA, so that the copy of a frame from the device runs without blocking the stepping thread.
        # The bytes between the arrays and the scalars stay zeros.
        slots = torch.zeros((N_FRAME_SLOTS, self._frame_size), dtype=torch.uint8, pin_memory=gs.backend == gs.cuda)
        self._slots_tc = slots
        self._slots = slots.numpy()
        self._slots_event = [torch.cuda.Event() if gs.backend == gs.cuda else None for _ in range(N_FRAME_SLOTS)]
        # The block-sparse encoding of a frame (see '_encode_blocks'), of its host-side end alone when the device
        # encodes the rest
        encoded_size = self._frame_size - (self._tail_offset if self._is_device_encoded else 0)
        self._bitmap = np.empty((encoded_size // FRAME_ALIGNMENT,), dtype=np.uint8)
        self._blocks = np.empty((encoded_size // 8,), dtype=np.uint64)
        self._free_slots: queue.SimpleQueue[int] = queue.SimpleQueue()
        for i_slot in range(N_FRAME_SLOTS):
            self._free_slots.put(i_slot)
        # The frames handed to the compression thread in step order, each preceded by the model parameters it holds if
        # they changed, then None at the stop
        self._frames: queue.SimpleQueue[int | InfoRecord | None] = queue.SimpleQueue()
        self._i_last_slot: int | None = None
        self._is_last_dropped = False

        # The chunk the compression thread is filling: its compressor, its compressed pieces and its frame count
        self._compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
        self._info_compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
        self._chunk_stream = self._compressor.compressobj()
        self._chunk_pieces: list[bytes] = []
        # The records of model parameters for the frames of the open chunk, which go to the file ahead of it
        self._info_records: list[bytes] = []
        self._n_chunk_frames = 0
        self._n_frames = 0
        self._written = 0
        self._fault: str | None = None
        self._error: Exception | None = None
        # The completed chunks handed to the I/O thread, in order, then None at the stop
        self._records: queue.SimpleQueue[tuple[bytes, int] | None] = queue.SimpleQueue()
        self._pending_bytes = 0
        self._pending_bytes_changed = threading.Condition()
        self._compression_thread: threading.Thread | None = None
        self._io_thread: threading.Thread | None = None
        self._file: io.BufferedWriter | None = None

        super().build()

    def _initialize_writer(self):
        # The header goes out with the first record. A run that ends before any leaves an empty file, which no reader
        # takes for a trajectory.
        self._file = open(self._get_filename(), "wb")
        self._header = _header(
            self._source,
            self._mode,
            self._options.chunk_size,
            self._fields,
            self._frame_size,
            self._info_fields,
            self._last_fields,
            self._last_frame_size,
        )
        self._written = 0
        self._compression_thread = threading.Thread(target=self._compress_frames, daemon=True)
        self._compression_thread.start()
        self._io_thread = threading.Thread(target=self._write_records, daemon=True)
        self._io_thread.start()

    def _get_frame(self):
        """Copy the frame of the current state into a free slot and return the slot.

        The model parameters go to the compression thread ahead of the frame, when it is the first or a setter changed
        them since the previous one.
        """
        sim = self._source.sim
        i_slot = self._free_slots.get()
        slot = self._slots[i_slot]
        # On Metal, the arrays are written by a queue torch does not wait for
        if self._frame_views is not None and gs.backend == gs.metal:
            qd.sync()
        if self._has_info and (self._i_frame == 0 or self._info_subscriber.pending):
            self._info_subscriber.clear()
            info_arrays = _read_arrays(self._info_arrays, self._info_fields)
            info_scalars = _read_scalars(self._info_scalars, self._info_scalars_fields)
            self._frames.put(InfoRecord(self._i_frame, np.concatenate((info_arrays, info_scalars))))
        if self._frame_views is not None and gs.backend == gs.cpu:
            np.concatenate(self._frame_views, out=slot[: self._device_end])
        elif self._is_device_encoded:
            # The first frame of a chunk is stored whole, the others as their difference with the previous frame (see
            # '_compress_frames'). The buffer of the previous frame takes the difference, and the next frame is
            # concatenated into it once the work queued before has read it.
            frame = self._staging[self._i_frame % 2]
            torch.cat(self._frame_views, out=frame[: self._device_end])
            if self._i_frame % self._options.chunk_size != 0:
                frame_prev = self._staging[(self._i_frame + 1) % 2]
                frame = torch.bitwise_xor(frame_prev, frame, out=frame_prev)
            # The changed blocks are gathered at the front of the buffer of the slot, in order, with a fixed number of
            # indices so that nothing waits on the device. The work is queued behind the step on the stream both run
            # on, and the compression thread waits for it to land (see '_compress_frames').
            blocks = frame.view(torch.int64).view((-1, BLOCK_SIZE // 8))
            is_block_changed = (blocks != 0).any(dim=1)
            blocks_idx = torch.nonzero_static(is_block_changed, size=blocks.shape[0], fill_value=0)[:, 0]
            torch.index_select(blocks, 0, blocks_idx, out=self._device_blocks[i_slot])
            self._slots_mask_tc[i_slot].copy_(is_block_changed, non_blocking=True)
            self._slots_event[i_slot].record()
        elif self._frame_views is not None:
            frame = self._staging[0]
            torch.cat(self._frame_views, out=frame[: self._device_end])
            self._slots_tc[i_slot, : self._tail_offset].copy_(frame)
        else:
            frame = _read_frame(sim, self._kinds, self._fields, self._frame_size)
            slot[: self._tail_offset] = frame[: self._tail_offset]
        tail = _frame_tail(sim, self._scalars, self._scalars_fields)
        slot[self._tail_offset : self._tail_offset + tail.size] = tail
        self._i_frame += 1
        return i_slot

    def step(self, global_step: int):
        # A failed write ends the recorder, as a failed sample does (see 'Recorder.record')
        if self._fault is not None:
            self._is_alive = False
            fault, self._fault = self._fault, None
            gs.raise_exception(fault)
        if self._error is not None:
            self._is_alive = False
            error, self._error = self._error, None
            gs.raise_exception_from(f"[TrajectoryFileWriter] Writing '{self._get_filename()}' failed.", error)
        super().step(global_step)

    def process(self, data, cur_time):
        self._i_last_slot = data
        self._frames.put(data)

    def _compress_frames(self):
        """Take the frames off the queue in step order and append them to the open chunk, on the compression thread.

        Each frame is held back until the next one arrives, so that the stop can drop the frame it recorded (see
        'cleanup'). A failure is raised on the stepping thread (see 'step'), and the frames that follow are released
        unwritten so the stepping thread never waits on a slot.
        """
        i_slot_prev: int | None = None
        i_slot_pending: int | None = None
        while True:
            i_slot = self._frames.get()
            if isinstance(i_slot, InfoRecord):
                if self._error is None and self._fault is None:
                    try:
                        payload = self._info_compressor.compress(i_slot.payload)
                        record = INFO_MAGIC + INFO_HEADER.pack(i_slot.i_frame, len(payload), zlib.crc32(payload))
                        self._info_records.append(record + payload)
                    except Exception as error:
                        self._error = error
                continue
            # A failure of the copy surfaces here, and is reported like a failure of the compression
            if i_slot is not None and self._slots_event[i_slot] is not None and self._error is None:
                try:
                    self._slots_event[i_slot].synchronize()
                except Exception as error:
                    self._error = error
            if i_slot_pending is not None and (i_slot is not None or not self._is_last_dropped):
                if self._error is None and self._fault is None:
                    try:
                        # The first frame of a chunk is stored whole, the others as their difference with the
                        # previous frame, block-sparse either way (see BLOCK_SIZE)
                        is_delta = self._n_chunk_frames > 0 and i_slot_prev is not None
                        frame = self._slots[i_slot_pending]
                        frame_prev = self._slots[i_slot_prev] if is_delta else frame
                        if self._is_device_encoded:
                            # The device took the difference of the part before the scalars and gathered its changed
                            # blocks (see '_get_frame'), so only these cross to the host
                            is_block_changed = self._slots_mask[i_slot_pending]
                            n_device_bytes = np.count_nonzero(is_block_changed) * BLOCK_SIZE
                            with torch.cuda.stream(self._copy_stream):
                                blocks = self._device_blocks[i_slot_pending].view(-1).view(torch.uint8)
                                self._slots_tc[i_slot_pending, :n_device_bytes].copy_(blocks[:n_device_bytes])
                            self._copy_stream.synchronize()
                            n_blocks = _encode_blocks(
                                frame[self._tail_offset :].view(np.uint64),
                                frame_prev[self._tail_offset :].view(np.uint64),
                                is_delta,
                                self._bitmap,
                                self._blocks,
                            )
                            bitmap = np.packbits(is_block_changed, bitorder="little")
                            self._chunk_pieces.append(self._chunk_stream.compress(bitmap))
                            self._chunk_pieces.append(self._chunk_stream.compress(self._bitmap))
                            self._chunk_pieces.append(self._chunk_stream.compress(frame[:n_device_bytes]))
                        else:
                            n_blocks = _encode_blocks(
                                frame.view(np.uint64), frame_prev.view(np.uint64), is_delta, self._bitmap, self._blocks
                            )
                            self._chunk_pieces.append(self._chunk_stream.compress(self._bitmap))
                        blocks = self._blocks[: n_blocks * (BLOCK_SIZE // 8)]
                        self._chunk_pieces.append(self._chunk_stream.compress(blocks))
                        self._n_chunk_frames += 1
                        if self._n_chunk_frames == self._options.chunk_size:
                            self._close_chunk()
                    except Exception as error:
                        self._error = error
                if i_slot_prev is not None:
                    self._free_slots.put(i_slot_prev)
                i_slot_prev, i_slot_pending = i_slot_pending, None
            if i_slot is None:
                break
            i_slot_pending = i_slot
        self._close_chunk()
        for i_slot in (i_slot_prev, i_slot_pending):
            if i_slot is not None:
                self._free_slots.put(i_slot)

    def _close_chunk(self):
        """Close the open chunk and hand it to the I/O thread, on the compression thread."""
        if self._n_chunk_frames == 0 or self._error is not None or self._fault is not None:
            return
        try:
            self._chunk_pieces.append(self._chunk_stream.flush())
            payload = b"".join(self._chunk_pieces)
            chunk_header = CHUNK_MAGIC + CHUNK_HEADER.pack(self._n_chunk_frames, len(payload), zlib.crc32(payload))
            # The records of model parameters go ahead of the chunk whose frames hold them, in one write, so that a file
            # closed at its max_size never ends on parameters without their frames
            record = b"".join((*self._info_records, chunk_header, payload))
            self._info_records.clear()
            with self._pending_bytes_changed:
                self._pending_bytes_changed.wait_for(
                    lambda: self._pending_bytes == 0 or self._pending_bytes + len(record) <= MAX_PENDING_BYTES
                )
                self._pending_bytes += len(record)
            self._records.put((record, self._n_chunk_frames))
        except Exception as error:
            self._error = error
        self._chunk_pieces.clear()
        self._chunk_stream = self._compressor.compressobj()
        self._n_chunk_frames = 0

    def _write_records(self):
        """Append the completed chunks to the file in order, on the I/O thread.

        A failure is raised on the stepping thread (see 'step'), and the chunks that follow are dropped.
        """
        while True:
            item = self._records.get()
            if item is None:
                break
            record, n_record_frames = item
            if self._error is None and self._file is not None:
                try:
                    self._write(record, n_record_frames)
                except Exception as error:
                    self._error = error
            with self._pending_bytes_changed:
                self._pending_bytes -= len(record)
                self._pending_bytes_changed.notify()

    def _write(self, record: bytes, n_record_frames: int):
        """Append one record to the file, with the header ahead of the first, if the file may still grow by that much.

        'n_record_frames' is the number of frames the record holds. A record the file cannot hold closes it and records
        the fault.
        """
        n_bytes = len(self._header) + len(record)
        if self._options.max_size is not None and self._written + n_bytes > self._options.max_size:
            self._file.close()
            self._file = None
            if self._written == 0:
                # A limit below the header and the first record leaves nothing to read, so the empty file goes too
                os.remove(self._get_filename())
                self._fault = (
                    f"The max_size of {self._options.max_size} bytes cannot hold the header and the first record of "
                    f"'{self._get_filename()}'."
                )
            else:
                self._fault = (
                    f"'{self._get_filename()}' reached its max_size of {self._options.max_size} bytes after "
                    f"{self._n_frames} frames, and was closed."
                )
            return
        # Joining the header and the record would copy the description of the scene, megabytes with its meshes, while
        # holding the interpreter lock the stepping thread waits on
        self._file.write(self._header)
        self._file.write(record)
        self._file.flush()
        self._header = b""
        self._written += n_bytes
        self._n_frames += n_record_frames

    def cleanup(self):
        sim = self._source.sim
        if self._compression_thread is not None:
            # The frame the stop recorded carries the step counts of the live state. The whole state ends the file in
            # its place (see CHECKPOINT_FILE_KINDS), so it is dropped. The compression thread holds that frame back
            # until told, so its slot is still intact.
            if self._i_last_slot is not None:
                if self._slots_event[self._i_last_slot] is not None:
                    self._slots_event[self._i_last_slot].synchronize()
                field = self._steps_field
                frame = self._slots[self._i_last_slot]
                # The slot holds the changed blocks when the device encodes the frame, which keeps the last frame
                if self._is_device_encoded:
                    frame = tensor_to_array(self._staging[(self._i_frame - 1) % 2])
                steps = frame[field.offset : field.offset + field.nbytes].view(field.dtype).reshape(field.shape)
                self._is_last_dropped = np.array_equal(steps, tensor_to_array(sim.steps))
            self._frames.put(None)
            self._compression_thread.join()
            self._compression_thread = None
            self._records.put(None)
            self._io_thread.join()
            self._io_thread = None
        try:
            if self._error is None and self._file is not None:
                payload = self._compressor.compress(
                    _read_frame(sim, CHECKPOINT_FILE_KINDS, self._last_fields, self._last_frame_size)
                )
                self._write(LAST_MAGIC + LAST_HEADER.pack(len(payload), zlib.crc32(payload)) + payload, 1)
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None
        # No step follows to raise what the last writes left, so it is raised here, the file closed
        if self._error is not None:
            gs.raise_exception_from(f"[TrajectoryFileWriter] Writing '{self._get_filename()}' failed.", self._error)
        if self._fault is not None:
            gs.raise_exception(self._fault)
        (gs.logger or LOGGER).info(
            f'[TrajectoryFileWriter] Saved {self._n_frames} frames to ~<"{self._get_filename()}">~.'
        )

    @property
    def run_in_thread(self) -> bool:
        return False


class Trajectory:
    """A recorded trajectory and the built scene it plays in (see 'Scene.load_trajectory').

    Frame i is the state the scene stood in after i steps, and the last frame is the state the recording stopped at,
    whole: every array but the static configs and constants (see CHECKPOINT_FILE_KINDS).

    Parameters
    ----------
    path : str or os.PathLike
        The file to read, as written by recording with 'TrajectoryFile' or by 'Scene.save_checkpoint'.
    scene : Scene, optional
        A built scene to play the trajectory in, from the same description and environment layout as the recorded
        one. If None, the recorded scene is created and built. Defaults to None.
    show_viewer : bool, optional
        Whether the created scene opens an interactive viewer. Defaults to False.
    viewer_options : ViewerOptions, optional
        Viewer options overriding the recorded ones in the created scene, for the fields they set.
    vis_options : VisOptions, optional
        Visualizer options overriding the recorded ones in the created scene, for the fields they set.
    renderer : RendererOptions, optional
        Renderer replacing the recorded one in the created scene. If None, the recorded one stands.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        scene: "Scene | None" = None,
        show_viewer: bool = False,
        viewer_options: "ViewerOptions | None" = None,
        vis_options: "VisOptions | None" = None,
        renderer: "RendererOptions | None" = None,
    ):
        # The scene module imports the recorders, so its classes are reached here rather than at import.
        from genesis.engine.scene import EnvironmentLayout, SceneCheckpoint, SceneDescription, description_digest

        self._path = path
        with open(path, "rb") as file:
            magic = file.read(len(MAGIC))
            if magic != MAGIC:
                if magic.startswith(MAGIC_PREFIX):
                    gs.raise_exception(
                        f"'{path}' was written in version {magic[len(MAGIC_PREFIX) : -1].decode()} of the trajectory "
                        f"format, and this Genesis reads version {MAGIC[len(MAGIC_PREFIX) : -1].decode()} only."
                    )
                gs.raise_exception(f"'{path}' is not a Genesis trajectory.")
            (length,) = LENGTH.unpack(file.read(LENGTH.size))
            self._manifest = json.loads(file.read(length))
            (length,) = LENGTH.unpack(file.read(LENGTH.size))
            description = file.read(length)
            self._fields = _fields_from_json(self._manifest["fields"])
            self._info_fields = _fields_from_json(self._manifest["info_fields"])
            self._last_fields = _fields_from_json(self._manifest["last_fields"])
            self._frame_size = self._manifest["frame_size"]
            self._last_frame_size = self._manifest["last_frame_size"]
            # Each record is trusted by its checksum, so the offsets stop at the first one a crash left incomplete.
            self._chunk_offsets: list[int] = []
            self._chunk_ends: list[int] = []
            self._last_offset: int | None = None
            # The records of model parameters, by the first frame each holds for, in the order they were written
            self._info_frames: list[int] = []
            self._info_offsets: list[int] = []
            n_frames = 0
            while True:
                offset = file.tell()
                magic = file.read(len(CHUNK_MAGIC))
                if magic == CHUNK_MAGIC:
                    header = file.read(CHUNK_HEADER.size)
                    if len(header) < CHUNK_HEADER.size:
                        break
                    n_chunk_frames, length, crc = CHUNK_HEADER.unpack(header)
                    payload = file.read(length)
                    if len(payload) < length or zlib.crc32(payload) != crc:
                        break
                    n_frames += n_chunk_frames
                    self._chunk_offsets.append(offset)
                    self._chunk_ends.append(n_frames)
                elif magic == INFO_MAGIC:
                    header = file.read(INFO_HEADER.size)
                    if len(header) < INFO_HEADER.size:
                        break
                    i_frame, length, crc = INFO_HEADER.unpack(header)
                    payload = file.read(length)
                    if len(payload) < length or zlib.crc32(payload) != crc:
                        break
                    self._info_frames.append(i_frame)
                    self._info_offsets.append(offset)
                elif magic == LAST_MAGIC:
                    header = file.read(LAST_HEADER.size)
                    if len(header) < LAST_HEADER.size:
                        break
                    length, crc = LAST_HEADER.unpack(header)
                    payload = file.read(length)
                    if len(payload) < length or zlib.crc32(payload) != crc:
                        break
                    self._last_offset = offset
                else:
                    break
        self._n_frames = n_frames + (self._last_offset is not None)
        if self._n_frames == 0:
            gs.raise_exception(f"'{path}' holds no frame.")
        self._i_cached_chunk: int | None = None
        self._cached_frames: np.ndarray | None = None
        self._i_cached_info: int | None = None
        self._cached_info: np.ndarray | None = None
        self._decompressor = zstandard.ZstdDecompressor()

        layout = self._manifest["layout"]
        self._layout = EnvironmentLayout(
            layout["n_envs"], tuple(layout["env_spacing"]), layout["n_envs_per_row"], layout["center_envs_at_origin"]
        )
        self._checkpoint_class = SceneCheckpoint
        if scene is None:
            scene = gs.Scene.load(io.BytesIO(description), show_viewer, viewer_options, vis_options, renderer)
            scene.build(**dataclasses.asdict(self._layout))
            description = None
        self._scene = scene
        self._digest = description_digest(scene.desc)
        if description is not None:
            recorded = serialization.load(io.BytesIO(description), {"scene": SceneDescription})["scene"]
            if description_digest(recorded) != self._digest:
                gs.raise_exception(f"'{path}' was recorded from another scene: its entities or physics options differ.")

    def __len__(self) -> int:
        return self._n_frames

    @property
    def scene(self) -> "Scene":
        """The built scene the trajectory plays in."""
        return self._scene

    @property
    def mode(self) -> str:
        """What the frames hold: 'exact' or 'minimal' (see TrajectoryFile)."""
        return self._manifest["mode"]

    @property
    def n_envs(self) -> int:
        """The number of environments the scene was recorded with."""
        return self._layout.n_envs

    def frame(self, index: int, kinds: frozenset[DataKind] = frozenset(DataKind)) -> dict[str, np.ndarray]:
        """Return the arrays of frame 'index' of the given kinds by name, shaped as the scene holds them. A negative
        'index' counts from the end.
        """
        try:
            index = range(len(self))[index]
        except IndexError:
            gs.raise_exception(f"Frame {index} is out of the {len(self)} frames of '{self._path}'.")
        if self._last_offset is not None and index == len(self) - 1:
            with open(self._path, "rb") as file:
                file.seek(self._last_offset + len(LAST_MAGIC))
                length, _ = LAST_HEADER.unpack(file.read(LAST_HEADER.size))
                payload = self._decompressor.decompress(file.read(length), max_output_size=self._last_frame_size)
                # A writable buffer, as the restore wraps the arrays as tensors
                frame = np.frombuffer(bytearray(payload), np.uint8)
            return _frame_arrays(frame, self._last_fields, kinds)
        # The chunk holding the frame is decoded whole, undoing its block-sparse encoding then its chain of
        # differences (see BLOCK_SIZE), and kept for the frames of the same chunk read next.
        i_chunk = bisect.bisect_right(self._chunk_ends, index)
        if i_chunk != self._i_cached_chunk:
            n_blocks = self._frame_size // BLOCK_SIZE
            n_bitmap_bytes = n_blocks // 8
            with open(self._path, "rb") as file:
                file.seek(self._chunk_offsets[i_chunk] + len(CHUNK_MAGIC))
                n_chunk_frames, length, _ = CHUNK_HEADER.unpack(file.read(CHUNK_HEADER.size))
                # A chunk is compressed as a stream, whose header states no size, so the decompression is bounded by
                # the size the chunk holds at most
                payload = self._decompressor.decompress(
                    file.read(length), max_output_size=n_chunk_frames * (n_bitmap_bytes + self._frame_size)
                )
            encoded = np.frombuffer(payload, np.uint8)
            frames = np.zeros((n_chunk_frames, n_blocks, BLOCK_SIZE), dtype=np.uint8)
            offset = 0
            for frame in frames:
                is_block_changed = np.unpackbits(encoded[offset : offset + n_bitmap_bytes], bitorder="little")
                offset += n_bitmap_bytes
                n_frame_bytes = np.count_nonzero(is_block_changed) * BLOCK_SIZE
                blocks = encoded[offset : offset + n_frame_bytes]
                frame[is_block_changed.view(np.bool_)] = blocks.reshape((-1, BLOCK_SIZE))
                offset += n_frame_bytes
            frames = frames.reshape((n_chunk_frames, self._frame_size))
            np.bitwise_xor.accumulate(frames, axis=0, out=frames)
            self._i_cached_chunk = i_chunk
            self._cached_frames = frames
        i_frame_start = self._chunk_ends[i_chunk - 1] if i_chunk > 0 else 0
        arrays = _frame_arrays(self._cached_frames[index - i_frame_start], self._fields, kinds)
        if DataKind.INFO in kinds and self._info_frames:
            # The model parameters of a frame are the ones of the latest record written for it or a frame before it,
            # which the writer puts ahead of every frame they hold for
            i_info = bisect.bisect_right(self._info_frames, index) - 1
            if i_info != self._i_cached_info:
                with open(self._path, "rb") as file:
                    file.seek(self._info_offsets[i_info] + len(INFO_MAGIC))
                    _, length, _ = INFO_HEADER.unpack(file.read(INFO_HEADER.size))
                    payload = self._decompressor.decompress(
                        file.read(length), max_output_size=sum(field.nbytes for field in self._info_fields)
                    )
                self._i_cached_info = i_info
                # A writable buffer, as the restore wraps the arrays as tensors
                self._cached_info = np.frombuffer(bytearray(payload), np.uint8)
            arrays.update(_frame_arrays(self._cached_info, self._info_fields, kinds))
        return arrays

    def time(self, index: int) -> np.ndarray:
        """The simulated time of each environment at frame 'index', in seconds, as 'Scene.get_time' reports it."""
        # Multiplied as the simulator multiplies its own clock, so the two agree to the bit.
        steps = torch.as_tensor(self.frame(index)[STEPS_FIELD], dtype=gs.tc_float, device="cpu")
        return tensor_to_array(steps * self._manifest["dt"])

    def timings(self, index: int) -> dict[str, float]:
        """The wall time in seconds of every phase of the step a frame was reached by, by phase.

        'index' is the frame, counted from the end when negative. The phases are named as 'Scene.timings' names them. A
        phase the step did not time is NaN, and so is every phase of a frame no step led to, such as the first.
        """
        return {
            name[len(TIMINGS_PREFIX) :]: value[()]
            for name, value in self.frame(index, frozenset((DataKind.STATE,))).items()
            if name.startswith(TIMINGS_PREFIX)
        }

    def seek(self, index: int) -> None:
        """Put the scene in the state of frame 'index', counted from the end when negative.

        An exact frame puts back everything a step reads or writes, so the scene steps on from there bit-for-bit. A
        minimal frame puts back the state and recomputes the link and geom poses by forward kinematics, and the next
        step finds the contacts and constraint forces anew. Either puts back the model parameters the frame was recorded
        with. The scratch that a checkpoint file also holds is for inspection, shaped by the backend of the recording
        that wrote it.
        """
        kinds = MODES_KINDS[self.mode] | INFO_KINDS
        values = self.frame(index, kinds)
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for name, value in values.items():
            if name != STEPS_FIELD and not name.startswith(TIMINGS_PREFIX):
                solver_name, array_name = name.split(".", 1)
                arrays.setdefault(solver_name, {})[array_name] = value
        solvers = {
            solver_name: SolverCheckpoint(solver_arrays, {}, kinds) for solver_name, solver_arrays in arrays.items()
        }
        sim = SimulatorCheckpoint(steps=values[STEPS_FIELD], solvers=solvers)
        self._scene.__setstate__(
            self._checkpoint_class(scene=self._scene.desc, digest=self._digest, layout=self._layout, sim=sim)
        )

    def play(self, loop: bool = False) -> None:
        """Seek every frame in turn, which redraws the scene at each and plays the run in the viewer at its pace.

        With 'loop', the run starts over until the interactive viewer is closed, so the scene must have one.
        """
        viewer = self._scene.viewer
        if loop and viewer is None:
            gs.raise_exception("Looping a trajectory needs an interactive viewer.")
        # The viewer holds each redraw to one step of wall time (see ViewerOptions.realtime_factor). A frame spanning
        # more steps than one, as frames recorded every few steps do, waits for the steps beyond that one as well.
        is_paced = viewer is not None and viewer.realtime_factor is not None
        while True:
            steps_prev = None
            for index in range(len(self)):
                # A closed viewer ends the replay before a redraw reaches it
                if viewer is not None and not viewer.is_alive():
                    return
                steps = self.frame(index)[STEPS_FIELD]
                if is_paced and steps_prev is not None:
                    time.sleep(max((steps - steps_prev).max() - 1, 0) * self._manifest["dt"] / viewer.realtime_factor)
                self.seek(index)
                steps_prev = steps
            if not loop:
                return
