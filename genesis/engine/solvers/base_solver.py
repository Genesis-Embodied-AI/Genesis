import dataclasses
import enum
import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable

import quadrants as qd
import numpy as np
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import qd_to_torch
from genesis.engine.entities.base_entity import Entity
from genesis.engine.states import QueriedStates
from genesis.repr_base import RBC


if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class StateChange(enum.Enum):
    """Category of solver scene-state mutation broadcast to subscribers (see `Solver.subscribe`).

    Solver-agnostic: it names what kind of state changed, never an index space (links, dofs, particles, ...), which
    differs from one solver to the next. GEOMETRY is the kinematic configuration that places or deforms the world
    surface (link poses, qpos, vertices); DYNAMICS is the velocity state. Model parameters (mass, inertia, friction,
    gains, limits) are not scene state and are never broadcast.
    """

    GEOMETRY = enum.auto()
    DYNAMICS = enum.auto()


def mutates(*changes: StateChange) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tag a solver state-mutating method with the StateChange categories it produces (one method may produce several,
    e.g. set_state changes both GEOMETRY and DYNAMICS).

    Notification is deferred to the outermost tagged call: a setter that internally calls other tagged setters fires a
    single notification per StateChange that occurred, when the outermost call returns, rather than one per nested
    call. The outermost call's `envs_idx` (None meaning all envs) is forwarded. Untagged methods, reads included, never
    notify, so a subscriber only ever wakes on a genuine mutation.
    """
    triggered = frozenset(changes)

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(method)

        @functools.wraps(method)
        def wrapper(self: "Solver", *args: Any, **kwargs: Any) -> Any:
            if not self._subscribers:
                return method(self, *args, **kwargs)
            if self._is_mutating:
                # Nested tagged call: record the changes, let the outermost call do the single notification.
                self._mutation_changes |= triggered
                return method(self, *args, **kwargs)
            self._is_mutating = True
            self._mutation_changes = set(triggered)
            try:
                result = method(self, *args, **kwargs)
            finally:
                self._is_mutating = False
            envs_idx = signature.bind(self, *args, **kwargs).arguments.get("envs_idx")
            for subscriber in self._subscribers:
                for changed in self._mutation_changes & subscriber.to:
                    if subscriber.callback is None:
                        subscriber._pending.add(changed)
                    else:
                        subscriber.callback(changed, envs_idx)
            return result

        return wrapper

    return decorator


class Subscriber:
    """A unique handle for the solver state changes whose category is in `to`.

    A consumer constructs a Subscriber and registers it with a solver via Solver.subscribe. The mode is fixed at
    construction by whether a callback is given:
      - eager (callback given): each matching change immediately calls callback(change, envs_idx);
      - lazy (no callback): matching changes accumulate into `pending` until the owner calls clear() - e.g. a sensor
        that rebuilds a cache on its next update rather than on every set_pos.
    """

    def __init__(self, to: frozenset[StateChange], callback: Callable[[StateChange, object], None] | None = None):
        self.to = to
        self.callback = callback
        self._pending: set[StateChange] = set()

    @property
    def pending(self) -> frozenset[StateChange]:
        """Categories accumulated since the last clear() (always empty in eager mode)."""
        return frozenset(self._pending)

    def clear(self):
        """Drop the accumulated changes, once they have been handled."""
        self._pending.clear()


class Solver(RBC):
    def __init__(self, scene: "Scene", sim: "Simulator", options):
        self._uid = gs.UID()
        self._sim = sim
        self._scene = scene
        self._dt: float = options.dt
        self._substep_dt: float = options.dt / sim.substeps
        self._init_gravity = getattr(options, "gravity", None)
        self._gravity = None
        self._entities: list[Entity] = gs.List()

        # Queue of solver-level states queried during the current backward window. Solvers that surface solver-state
        # (kinematic, rigid) push into it from `get_state`; others leave it empty. `Simulator.get_state` calls `discard`
        # here to lift entries owned by a `SimState`, preventing `collect_output_grads` from accumulating adjoints twice
        # through both the simulator-level and the per-solver loop.
        self._queried_states = QueriedStates()

        self.data_manager = None

        # force fields
        self._ffs = list()

        # Registered Subscribers, notified after @mutates-tagged methods run; see subscribe(). The re-entrancy guard
        # below defers notification to the outermost tagged call, accumulating every change that occurred in between,
        # so a setter calling other tagged setters notifies once rather than per nested call.
        self._subscribers: set[Subscriber] = set()
        self._is_mutating = False
        self._mutation_changes: set[StateChange] = set()

    def _add_force_field(self, force_field):
        self._ffs.append(force_field)

    def subscribe(self, subscriber: Subscriber):
        """Register a Subscriber to be notified after any @mutates-tagged method whose change is in its filter."""
        self._subscribers.add(subscriber)

    def build(self):
        self._B = self._sim._B
        if self._init_gravity is not None:
            gravity = np.tile(np.asarray(self._init_gravity, dtype=gs.np_float), (self._B, 1))
            self._gravity = array_class.V(gs.qd_vec3, (self._B,))
            self._gravity.from_numpy(gravity)

    @gs.assert_built
    def set_gravity(self, gravity, envs_idx=None):
        if self._gravity is None:
            gs.logger.debug("Gravity is not defined, skipping `set_gravity`.")
            return

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        gravity = torch.as_tensor(gravity, dtype=gs.tc_float, device=gs.device).expand((len(envs_idx), 3)).contiguous()
        assert gravity.shape == (len(envs_idx), 3), "Input gravity array should match (n_envs, 3)"
        gravity_arg = self._gravity if type(self._gravity) is qd.VectorTensor else qd.wrap(self._gravity)
        _kernel_set_gravity(gravity, envs_idx, gravity_arg)

    def get_gravity(self, envs_idx=None):
        tensor = qd_to_torch(self._gravity, envs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def _iter_data_manager_tensors(self):
        """Yield (store name, tensor) for every tensor reachable from the data manager, descending through the
        nested per-component dataclasses."""

        def walk(prefix, struct):
            if dataclasses.is_dataclass(struct):
                for field in dataclasses.fields(struct):
                    yield from walk(f"{prefix}.{field.name}", getattr(struct, field.name))
            elif isinstance(struct, (qd.Tensor, qd.Field, qd.Ndarray)):
                yield prefix, struct

        for attr_name, struct in self.data_manager.__dict__.items():
            yield from walk(f"{self.__class__.__name__}.data_manager.{attr_name}", struct)

    def dump_ckpt_to_numpy(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}

        for attr_name, value in self.__dict__.items():
            if not isinstance(value, (qd.Tensor, qd.Field, qd.Ndarray)):
                continue

            key_base = ".".join((self.__class__.__name__, attr_name))
            data = value.to_numpy()

            # StructField -> data is a dict: flatten each member
            if isinstance(data, dict):
                for sub_name, sub_arr in data.items():
                    arrays[f"{key_base}.{sub_name}"] = sub_arr
            else:
                arrays[key_base] = data

        if self.data_manager is not None:
            for store_name, sub_arr in self._iter_data_manager_tensors():
                arrays[store_name] = sub_arr.to_numpy()

        return arrays

    def load_ckpt_from_numpy(self, arr_dict: dict[str, np.ndarray]) -> None:
        for attr_name, value in self.__dict__.items():
            if not isinstance(value, (qd.Tensor, qd.Field, qd.Ndarray)):
                continue

            key_base = ".".join((self.__class__.__name__, attr_name))
            member_prefix = key_base + "."

            # ---- StructField: gather its members -----------------------------
            member_items = {}
            for saved_key, saved_arr in arr_dict.items():
                if saved_key.startswith(member_prefix):
                    sub_name = saved_key[len(member_prefix) :]
                    member_items[sub_name] = saved_arr

            if member_items:  # we found at least one sub-member
                value.from_numpy(member_items)
                continue

            # ---- Ordinary field ---------------------------------------------
            if key_base not in arr_dict:
                continue  # nothing saved for this attribute

            arr = arr_dict[key_base]
            value.from_numpy(arr)

        # if it has data_manager, add it to the arrays
        if self.data_manager is not None:
            for store_name, sub_arr in self._iter_data_manager_tensors():
                if store_name in arr_dict:
                    sub_arr.from_numpy(arr_dict[store_name])
                else:
                    gs.logger.warning(f"Failed to load {store_name}. Not found in stored arrays.")

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        return self._uid

    @property
    def scene(self):
        return self._scene

    @property
    def sim(self):
        return self._sim

    @property
    def dt(self):
        return self._dt

    @property
    def is_built(self):
        return self._scene._is_built

    @property
    def substep_dt(self):
        return self._substep_dt

    @property
    def gravity(self):
        return self._gravity.to_numpy() if self._gravity is not None else None

    @property
    def entities(self) -> list[Entity]:
        return self._entities

    @property
    def n_entities(self):
        return len(self._entities)

    def _repr_brief(self):
        repr_str = f"{self.__repr_name__()}: {self._uid}, n_entities: {self.n_entities}"
        return repr_str


@qd.kernel
def _kernel_set_gravity(tensor: qd.types.ndarray(), envs_idx: qd.types.ndarray(), gravity: qd.Tensor):
    # qd.Tensor annotation accepts qd.Tensor wrappers, raw qd.field(), and raw qd.ndarray(). Subclass solvers store
    # _gravity as raw qd.field(); base_solver stores it as qd.Tensor.
    for i_b_ in range(envs_idx.shape[0]):
        for j in qd.static(range(3)):
            gravity[envs_idx[i_b_]][j] = tensor[i_b_, j]
