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

import zstandard

import quadrants as qd

import genesis as gs
from genesis.engine.solvers import KinematicSolver
from genesis.engine.states.solvers import KinematicSolverCheckpoint, SimulatorCheckpoint, SolverCheckpoint
from genesis.options.recorders import TrajectoryFile
from genesis.utils import serialization
from genesis.utils.array_class import CHECKPOINT_KINDS, DataKind
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array

from .file_writers import BaseFileWriter
from .recorder_manager import register_recording

if TYPE_CHECKING:
    from genesis.engine.scene import Scene, TrajectorySource
    from genesis.engine.simulator import Simulator
    from genesis.options.renderers import RendererOptions
    from genesis.options.vis import ViewerOptions, VisOptions

LOGGER = logging.getLogger(__name__)

TRAJECTORY_FORMAT = ".gstraj"
MAGIC = b"GSTRAJ2\n"
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
LAST_MAGIC = b"LAST"
# The record a trajectory ends with: payload length, payload checksum.
LAST_HEADER = struct.Struct("<QI")
LENGTH = struct.Struct("<Q")
# What a frame holds in each mode (see TrajectoryFile). A checkpoint file and the record a trajectory ends with hold
# every array but the static configs and constants, the scratch of the solvers included. They serve the analysis of a
# failure, and a failure may depend on the platform and never reproduce once shared. What the last step computed is
# therefore kept whole.
COMPRESSED_KINDS = frozenset((DataKind.INFO, DataKind.STATE))
EXACT_KINDS = CHECKPOINT_KINDS
CHECKPOINT_FILE_KINDS = frozenset(DataKind) - {DataKind.CONFIG, DataKind.CONSTANT}
# The per-environment step counts and the forward-kinematics flags of a solver, which follow the arrays in a frame.
STEPS_FIELD = "steps"
FLAGS_SUFFIX = ".forward_flags"
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


def _frame_fields(sim: "Simulator", kinds: frozenset[DataKind], drop_unused: bool) -> list[FrameField]:
    """Lay out one frame of the given kinds (see '_read_frame'): every array, then the step count of each environment,
    the forward-kinematics flags of every kinematic or rigid solver, and the wall time of every phase of a step.

    With 'drop_unused', a 0-d array is listed empty. Every 0-d array of the kinds a step frame holds is the placeholder
    of a feature switched off (see 'maybe_shape'), which a restore leaves as built (see 'check_data').

    Where zero-copy views exist, an array keeps the memory order of its view, a permutation of its canonical axes, so
    that a frame is copied from the views as they lie (see 'TrajectoryFileWriter.build'). Otherwise it is canonical.
    """
    fields = []
    offset = 0
    for name, value, kind in sim.data(kinds):
        array = qd_to_numpy(value)
        if drop_unused and not value.shape:
            fields.append(FrameField(name, kind, (0,), array.dtype.str, offset, 0, (0,)))
            continue
        axes = tuple(range(array.ndim))
        if gs.use_zerocopy and value.shape:
            view = qd_to_torch(value, copy=False)
            # Sorting is stable, so the axes of equal strides, which only a size-1 axis can share, keep their order
            axes = tuple(sorted(axes, key=view.stride, reverse=True))
            assert view.permute(axes).is_contiguous()
        fields.append(FrameField(name, kind, array.shape, array.dtype.str, offset, array.nbytes, axes))
        offset += array.nbytes
    steps = tensor_to_array(sim.steps)
    fields.append(FrameField(STEPS_FIELD, DataKind.STATE, steps.shape, steps.dtype.str, offset, steps.nbytes, (0,)))
    offset += steps.nbytes
    for solver in sim.active_solvers:
        if isinstance(solver, KinematicSolver):
            name = f"{type(solver).__name__}{FLAGS_SUFFIX}"
            fields.append(FrameField(name, DataKind.STATE, (2,), "|u1", offset, 2, (0,)))
            offset += 2
    for phase in sim.step_timer.phases:
        fields.append(FrameField(f"{TIMINGS_PREFIX}{phase}", DataKind.STATE, (), "<f8", offset, 8, ()))
        offset += 8
    return fields


def _frame_tail(sim: "Simulator") -> np.ndarray:
    """Return the end of a frame that lives on the host, as a flat uint8 buffer.

    It holds the forward-kinematics flags, then the wall time of every phase of the step the state was reached by, NaN
    for a phase that step did not time.
    """
    flags = np.array(
        [
            flag
            for solver in sim.active_solvers
            if isinstance(solver, KinematicSolver)
            for flag in (solver.is_forward_pos_updated, solver.is_forward_vel_updated)
        ],
        dtype=np.uint8,
    )
    phases = sim.step_timer.phases
    step_timings = sim.step_timer.timings
    # A phase timed without being declared would be lost, as the frame is laid out once (see 'StepTimer.add_phases')
    assert step_timings.keys() <= set(phases)
    timings = np.array([step_timings.get(phase, np.nan) for phase in phases])
    return np.concatenate((flags, timings.view(np.uint8)))


def _read_frame(sim: "Simulator", kinds: frozenset[DataKind], fields: list[FrameField]) -> np.ndarray:
    """Return one frame of the given kinds as a flat uint8 buffer laid out as 'fields' states it (see '_frame_fields').

    Where zero-copy views exist the arrays are concatenated on the device and cross to the host once. Otherwise each
    array crosses on its own, to be concatenated on the host.
    """
    values = [(value, field.axes) for (_, value, _), field in zip(sim.data(kinds), fields) if field.nbytes]
    if gs.use_zerocopy:
        parts = [qd_to_torch(value).permute(axes).reshape(-1).view(torch.uint8) for value, axes in values]
        parts.append(sim.steps.reshape(-1).view(torch.uint8))
        return np.concatenate((tensor_to_array(torch.cat(parts)), _frame_tail(sim)))
    parts = [qd_to_numpy(value).transpose(axes).reshape(-1).view(np.uint8) for value, axes in values]
    parts.append(tensor_to_array(sim.steps).reshape(-1).view(np.uint8))
    parts.append(_frame_tail(sim))
    return np.concatenate(parts)


def _header(
    source: "TrajectorySource", exact: bool, chunk_size: int, fields: list[FrameField], last_fields: list[FrameField]
) -> bytes:
    """Return the magic, the JSON manifest and the scene description that open a trajectory file, 'fields' laying out
    its frames and 'last_fields' the record it ends with.
    """
    sim, desc, layout = source
    description = io.BytesIO()
    serialization.export(description, {"scene": desc})
    manifest = {
        "genesis": gs.__version__,
        "source": serialization.source_digest(),
        "backend": gs.backend.name,
        "precision": np.dtype(gs.np_float).name,
        "exact": exact,
        "chunk_size": chunk_size,
        "dt": sim.dt,
        "layout": {key: _json_value(value) for key, value in dataclasses.asdict(layout).items()},
        "configs": {
            name: {key: _json_value(value) for key, value in vars(config).items() if not key.startswith("_")}
            for solver in sim.active_solvers
            for name, config, kind in solver.data
            if kind == DataKind.CONFIG
        },
        "fields": _fields_to_json(fields),
        "last_fields": _fields_to_json(last_fields),
    }
    text = json.dumps(manifest).encode()
    return MAGIC + LENGTH.pack(len(text)) + text + LENGTH.pack(description.getbuffer().nbytes) + description.getvalue()


def save_checkpoint(path: str | os.PathLike, source: "TrajectorySource") -> None:
    """Write the whole state of a built scene as a trajectory file of one frame, the record a trajectory ends with (see
    'Scene.save_checkpoint').
    """
    fields = _frame_fields(source.sim, CHECKPOINT_FILE_KINDS, drop_unused=False)
    with open(path, "wb") as file:
        file.write(_header(source, exact=True, chunk_size=1, fields=fields, last_fields=fields))
        payload = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(
            _read_frame(source.sim, CHECKPOINT_FILE_KINDS, fields)
        )
        file.write(LAST_MAGIC + LAST_HEADER.pack(len(payload), zlib.crc32(payload)) + payload)


@register_recording(TrajectoryFile)
class TrajectoryFileWriter(BaseFileWriter):
    """Write the frames of a scene to a trajectory file (see TrajectoryFile), one chunk at a time.

    Each frame captures the state a step starts from, control inputs included, since it is read before the step. The
    stepping thread copies each frame into one of a few frame slots allocated at build and hands it to a compression
    thread, which stores it as its XOR with the previous frame of its chunk and streams it through the compressor.
    Unchanged bytes thus become zeros the compressor removes, and an unchanged info array costs nothing. A completed
    chunk goes to a third thread, which appends it to the file. A step costs one copy of the frame, issued without
    waiting for a GPU to finish the step: the compression thread waits for the copy to land instead. The step waits only
    when every slot is taken, which happens when the frames are compressed slower than the scene steps.
    """

    def build(self):
        self._source: TrajectorySource = self._data_func()
        sim = self._source.sim
        self._is_exact = self._options.exact if self._options.exact is not None else sim.n_envs <= 1
        if not self._is_exact:
            gs.logger.warning(
                "[TrajectoryFileWriter] Recording the state alone: link and geom poses are recomputed at load and may "
                "differ at the last bits, and the log cannot resume the simulation bit-for-bit. Set 'exact=True' to "
                "log everything a step reads or writes."
            )
        self._kinds = EXACT_KINDS if self._is_exact else COMPRESSED_KINDS
        # The frames of the steps leave the unused arrays out, as they are read once per step. The last record holds
        # them, with the scratch.
        self._fields = _frame_fields(sim, self._kinds, drop_unused=True)
        self._last_fields = _frame_fields(sim, CHECKPOINT_FILE_KINDS, drop_unused=False)
        self._frame_size = sum(field.nbytes for field in self._fields)
        self._steps_field = next(field for field in self._fields if field.name == STEPS_FIELD)
        # The forward-kinematics flags and the step timings close the frame (see '_frame_fields'). They live on the
        # host, so a frame copy stops short of them and they are written beside it.
        self._flags_offset = self._steps_field.offset + self._steps_field.nbytes

        # The arrays of a built scene keep their memory, so the byte view each array is copied from is taken once, in
        # the memory order the frame stores it in (see '_frame_fields'). Without zero-copy views each frame is read
        # whole (see '_read_frame').
        # FIXME: quadrants#970 - a 0-d field has no zero-copy view, and reading one per frame runs a copy kernel. The
        # step frames leave out the 0-d arrays (see '_frame_fields'), whose content the step frames therefore lack.
        self._frame_views: list[np.ndarray | torch.Tensor] | None = None
        self._staging: torch.Tensor | None = None
        if gs.use_zerocopy:
            self._frame_views = []
            for (_, value, _), field in zip(sim.data(self._kinds), self._fields):
                if field.nbytes:
                    view = qd_to_torch(value, copy=False).permute(field.axes).view(-1).view(torch.uint8)
                    # The views of the CPU backend are host memory already, and numpy concatenates them on the calling
                    # thread. Torch would run the copy on its thread pool, whose wake-up costs more than the copy.
                    self._frame_views.append(view.numpy() if gs.backend == gs.cpu else view)
            if gs.backend == gs.cpu:
                self._frame_views.append(sim.steps.numpy().view(np.uint8))
            else:
                self._frame_views.append(sim.steps.view(torch.uint8))
                self._staging = torch.empty((self._flags_offset,), dtype=torch.uint8, device=gs.device)

        # Page-locked on CUDA, so that the copy of a frame from the device runs without blocking the stepping thread
        slots = torch.empty((N_FRAME_SLOTS, self._frame_size), dtype=torch.uint8, pin_memory=gs.backend == gs.cuda)
        self._slots_tc = slots
        self._slots = slots.numpy()
        self._slots_event = [torch.cuda.Event() if gs.backend == gs.cuda else None for _ in range(N_FRAME_SLOTS)]
        self._delta = np.empty((self._frame_size,), dtype=np.uint8)
        self._free_slots: queue.SimpleQueue[int] = queue.SimpleQueue()
        for i_slot in range(N_FRAME_SLOTS):
            self._free_slots.put(i_slot)
        # The frames handed to the compression thread, in step order, then None at the stop
        self._frames: queue.SimpleQueue[int | None] = queue.SimpleQueue()
        self._i_last_slot: int | None = None
        self._is_last_dropped = False

        # The chunk the compression thread is filling: its compressor, its compressed pieces and its frame count
        self._compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
        self._chunk_stream = self._compressor.compressobj()
        self._chunk_pieces: list[bytes] = []
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
        self._header = _header(self._source, self._is_exact, self._options.chunk_size, self._fields, self._last_fields)
        self._written = 0
        self._compression_thread = threading.Thread(target=self._compress_frames, daemon=True)
        self._compression_thread.start()
        self._io_thread = threading.Thread(target=self._write_records, daemon=True)
        self._io_thread.start()

    def _get_frame(self):
        """Copy the frame of the current state into a free slot and return the slot."""
        sim = self._source.sim
        i_slot = self._free_slots.get()
        slot = self._slots[i_slot]
        if self._frame_views is not None and gs.backend == gs.cpu:
            np.concatenate(self._frame_views, out=slot[: self._flags_offset])
        elif self._frame_views is not None:
            # On Metal, the arrays are written by a queue torch does not wait for
            if gs.backend == gs.metal:
                qd.sync()
            torch.cat(self._frame_views, out=self._staging)
            # The copy is queued behind the step on the stream both run on, and the compression thread waits for it to
            # land (see '_compress_frames'). On another GPU backend the copy returns once done.
            self._slots_tc[i_slot, : self._flags_offset].copy_(self._staging, non_blocking=gs.backend == gs.cuda)
            if self._slots_event[i_slot] is not None:
                self._slots_event[i_slot].record()
        else:
            slot[: self._flags_offset] = _read_frame(sim, self._kinds, self._fields)[: self._flags_offset]
        slot[self._flags_offset :] = _frame_tail(sim)
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
            # A failure of the copy surfaces here, and is reported like a failure of the compression
            if i_slot is not None and self._slots_event[i_slot] is not None and self._error is None:
                try:
                    self._slots_event[i_slot].synchronize()
                except Exception as error:
                    self._error = error
            if i_slot_pending is not None and (i_slot is not None or not self._is_last_dropped):
                if self._error is None and self._fault is None:
                    try:
                        if self._n_chunk_frames == 0 or i_slot_prev is None:
                            delta = self._slots[i_slot_pending]
                        else:
                            delta = np.bitwise_xor(
                                self._slots[i_slot_pending], self._slots[i_slot_prev], out=self._delta
                            )
                        self._chunk_pieces.append(self._chunk_stream.compress(delta))
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
            record = chunk_header + payload
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
                payload = self._compressor.compress(_read_frame(sim, CHECKPOINT_FILE_KINDS, self._last_fields))
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
            self._last_fields = _fields_from_json(self._manifest["last_fields"])
            self._frame_size = sum(field.nbytes for field in self._fields)
            # Each record is trusted by its checksum, so the offsets stop at the first one a crash left incomplete.
            self._chunk_offsets: list[int] = []
            self._chunk_ends: list[int] = []
            self._last_offset: int | None = None
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
    def is_exact(self) -> bool:
        """Whether the frames hold everything a step reads or writes, or the state alone (see TrajectoryFile)."""
        return self._manifest["exact"]

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
                payload = self._decompressor.decompress(
                    file.read(length), max_output_size=sum(field.nbytes for field in self._last_fields)
                )
                # A writable buffer, as the restore wraps the arrays as tensors
                frame = np.frombuffer(bytearray(payload), np.uint8)
            return _frame_arrays(frame, self._last_fields, kinds)
        # The chunk holding the frame is decoded whole, undoing its chain of differences, and kept for the frames of
        # the same chunk read next.
        i_chunk = bisect.bisect_right(self._chunk_ends, index)
        if i_chunk != self._i_cached_chunk:
            with open(self._path, "rb") as file:
                file.seek(self._chunk_offsets[i_chunk] + len(CHUNK_MAGIC))
                n_chunk_frames, length, _ = CHUNK_HEADER.unpack(file.read(CHUNK_HEADER.size))
                # A chunk is compressed as a stream, whose header states no size, so the decompression is bounded by
                # the size the chunk holds
                payload = self._decompressor.decompress(
                    file.read(length), max_output_size=n_chunk_frames * self._frame_size
                )
                deltas = np.frombuffer(payload, np.uint8)
            frames = deltas.reshape((n_chunk_frames, self._frame_size)).copy()
            np.bitwise_xor.accumulate(frames, axis=0, out=frames)
            self._i_cached_chunk = i_chunk
            self._cached_frames = frames
        i_frame_start = self._chunk_ends[i_chunk - 1] if i_chunk > 0 else 0
        return _frame_arrays(self._cached_frames[index - i_frame_start], self._fields, kinds)

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
        compressed frame puts back the state and recomputes the link and geom poses by forward kinematics. The scratch a
        checkpoint file also holds is for inspection: its shape follows the backend of the recording.
        """
        kinds = EXACT_KINDS if self.is_exact else COMPRESSED_KINDS
        values = self.frame(index, kinds)
        arrays: dict[str, dict[str, np.ndarray]] = {}
        flags: dict[str, tuple[bool, bool]] = {}
        for name, value in values.items():
            if name == STEPS_FIELD or name.startswith(TIMINGS_PREFIX):
                continue
            solver_name, array_name = name.split(".", 1)
            if name.endswith(FLAGS_SUFFIX):
                # Native booleans: the step hands these to kernels as template arguments.
                flags[solver_name] = tuple(map(bool, value))
            else:
                arrays.setdefault(solver_name, {})[array_name] = value
        solvers = {}
        for solver_name, solver_arrays in arrays.items():
            if solver_name in flags:
                is_pos_updated, is_vel_updated = flags[solver_name]
                solvers[solver_name] = KinematicSolverCheckpoint(
                    solver_arrays, {}, kinds, is_pos_updated, is_vel_updated
                )
            else:
                solvers[solver_name] = SolverCheckpoint(solver_arrays, {}, kinds)
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
