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
from genesis.utils.misc import qd_to_torch, sanitize_index, tensor_to_array
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


class MutatedLinks(enum.Enum):
    """Which links a @mutates-tagged method can affect, consumed by link-filtered subscribers (see Subscriber).

    ALL: any link (the conservative default). ARTICULATED: only links whose pose is a function of the configuration
    - a configuration-space setter (qpos, dofs) can never displace a fixed link, which carries no degree of freedom
    (fixed links move only through explicit base-pose setters, whose reach is their link selection). A method whose
    reach is an explicit link selection instead declares the name of its links-index argument.
    """

    ALL = enum.auto()
    ARTICULATED = enum.auto()


def _expand_links_subtree(links_idx: np.ndarray, links_parent_idx: np.ndarray) -> np.ndarray:
    """Close a set of global link indices over their kinematic subtrees: every descendant of a member is a member.

    Iterates one tree level per pass (a link joins when its parent is a member), so it terminates after at most the
    tree depth."""
    is_affected = np.zeros(links_parent_idx.shape[0], dtype=bool)
    is_affected[links_idx] = True
    has_parent = links_parent_idx >= 0
    while True:
        is_expanded = is_affected.copy()
        is_expanded[has_parent] |= is_affected[links_parent_idx[has_parent]]
        if (is_expanded == is_affected).all():
            return np.flatnonzero(is_affected)
        is_affected = is_expanded


def mutates(
    *changes: StateChange, links: str | MutatedLinks = MutatedLinks.ALL
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tag a solver state-mutating method with the StateChange categories it produces (one method may produce several,
    e.g. set_state changes both GEOMETRY and DYNAMICS).

    Notification is deferred to the outermost tagged call: a setter that internally calls other tagged setters fires a
    single notification per StateChange that occurred, when the outermost call returns, rather than one per nested
    call. The outermost call's `envs_idx` (None meaning all envs) is forwarded. Untagged methods, reads included, never
    notify, so a subscriber only ever wakes on a genuine mutation.

    `links` declares which links the method can affect, so link-filtered subscribers can drop notifications that
    cannot concern them: the name of the argument holding the targeted global link indices (e.g. "links_idx") - the
    reach then closes over their kinematic subtrees, since forward kinematics carries a new pose to every descendant,
    fixed children included - MutatedLinks.ARTICULATED for configuration-space setters, or MutatedLinks.ALL when any
    link may change. Nested tagged calls union their mutated links, an unbounded reach absorbing the rest.
    """
    triggered = frozenset(changes)

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(method)

        def resolve_mutated_links_idx(self: "Solver", bound_arguments) -> np.ndarray | None:
            if links is MutatedLinks.ARTICULATED:
                return self._articulated_links_idx
            if links is MutatedLinks.ALL:
                return None
            links_idx = bound_arguments.get(links)
            if links_idx is None or self._links_parent_idx is None:
                # The setter substitutes its own default selection (potentially reaching any link), or the solver
                # exposes no kinematic structure to close the reach over: stay unbounded.
                return None
            # sanitize_index covers every index form the setters accept (int, sequence, slice, bool mask, tensor);
            # the reach set lives host-side, where the subscriber filters compare.
            links_idx = tensor_to_array(sanitize_index(links_idx, -1, self.n_links, 0, links))
            return _expand_links_subtree(links_idx, self._links_parent_idx)

        @functools.wraps(method)
        def wrapper(self: "Solver", *args: Any, **kwargs: Any) -> Any:
            if not self._subscribers:
                return method(self, *args, **kwargs)
            bound_arguments = signature.bind(self, *args, **kwargs).arguments
            mutated_links_idx = resolve_mutated_links_idx(self, bound_arguments)
            if self._is_mutating:
                # Nested tagged call: record the changes and mutated links, let the outermost call do the single
                # notification.
                self._mutation_changes |= triggered
                if self._mutation_links_idx_list is not None:
                    if mutated_links_idx is None:
                        self._mutation_links_idx_list = None
                    else:
                        self._mutation_links_idx_list.append(mutated_links_idx)
                return method(self, *args, **kwargs)
            self._is_mutating = True
            self._mutation_changes = set(triggered)
            self._mutation_links_idx_list = None if mutated_links_idx is None else [mutated_links_idx]
            try:
                result = method(self, *args, **kwargs)
            finally:
                self._is_mutating = False
            envs_idx = bound_arguments.get("envs_idx")
            links_idx = None
            if self._mutation_links_idx_list is not None:
                links_idx = np.concatenate(self._mutation_links_idx_list)
            for subscriber in self._subscribers:
                if not subscriber.is_watching(links_idx):
                    continue
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

    `links_filter` (global link indices, normalized by Solver.subscribe) restricts the subscription to changes that
    can affect those links: a notification whose mutated links (see mutates) are disjoint from the filter is dropped;
    a notification with an unbounded reach always passes.
    """

    def __init__(
        self,
        to: frozenset[StateChange],
        callback: Callable[[StateChange, object], None] | None = None,
        links_filter: "np.typing.ArrayLike | None" = None,
    ):
        self.to = to
        self.callback = callback
        self.links_filter = links_filter
        self._pending: set[StateChange] = set()

    def is_watching(self, links_idx: np.ndarray | None) -> bool:
        """Whether a change affecting `links_idx` (None meaning possibly any link) concerns this subscriber."""
        if self.links_filter is None or links_idx is None:
            return True
        return bool(np.isin(links_idx, self.links_filter).any())

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
        self._mutation_links_idx_list: list[np.ndarray] | None = None
        # Global indices of the links whose pose is a function of the configuration, resolving
        # MutatedLinks.ARTICULATED notifications, and the per-link parent indices (-1 for roots) closing named-links
        # reaches over kinematic subtrees; None keeps the corresponding notifications unbounded. Set at build by
        # solvers that expose a kinematic structure (see KinematicSolver).
        self._articulated_links_idx: np.ndarray | None = None
        self._links_parent_idx: np.ndarray | None = None

    def _add_force_field(self, force_field):
        self._ffs.append(force_field)

    def subscribe(self, subscriber: Subscriber):
        """Register a Subscriber to be notified after any @mutates-tagged method whose change is in its filter."""
        if subscriber.links_filter is not None:
            subscriber.links_filter = tensor_to_array(
                sanitize_index(subscriber.links_filter, -1, self.n_links, 0, "links_filter")
            )
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
