"""Recursive compilation of frozen Action control graphs.

An ActionSchedule is the compile-time control counterpart of GlobalDataContext:

- GlobalDataContext is the only device-data discovery root.
- ActionSchedule contains only which frozen Actions execute and how nested Actions are connected.

Context/schedule-based graph Pipelines take the Schedule instead of their manager System. Passing the manager would
make Quadrants traverse its dependency graph as another data root, duplicate GlobalDataContext's role and hide child
Action edges behind arbitrary System attributes. Legacy or subsystem kernels may still take a System directly. The
compiler snapshots only explicit edges carried in Action bound arguments, so the graph topology is immutable,
inspectable and cycle-checked before Quadrants traces a kernel.
"""

from collections.abc import Mapping

import genesis as gs
from genesis.engine.core import Action, ActionCollection


class ActionSchedule:
    """Frozen recursive Action control graph with no device-data ownership.

    Root collection names live under ``collections``. Nested Action/ActionCollection edges are discovered only through
    explicit Action bound arguments. ``topology_key`` records the structural graph for diagnostics and cache
    integration; ``nested`` and ``actions`` retain the exact frozen objects QD statically unrolls.
    """


class _ScheduleCollections:
    """Namespace separating user-declared collection names from schedule metadata."""


def compile_action_schedule(collections: Mapping[str, ActionCollection]) -> ActionSchedule:
    """Snapshot frozen roots and every explicit child edge into one validated, cycle-free control schedule."""
    allowed_children = {
        "host": {"host"},
        "inline": {"inline"},
        "stage": {"stage", "inline"},
    }
    nested: list[tuple[Action, ...]] = []
    collection_indices: dict[int, int] = {}
    direct_actions: list[Action] = []
    direct_action_indices: dict[int, int] = {}
    topology: list[tuple] = []
    active_collections: list[ActionCollection] = []
    active_actions: list[Action] = []

    def function_key(action: Action) -> tuple[str, str]:
        function = getattr(action.function, "fn", action.function)
        return function.__module__, function.__qualname__

    def check_kind(parent: Action, child_kind: str) -> None:
        if child_kind not in allowed_children[parent.kind]:
            gs.raise_exception(f"{parent.kind} Action cannot invoke {child_kind} Action")

    def walk_action(action: Action) -> int:
        action_id = id(action)
        if any(action is parent for parent in active_actions):
            cycle = " -> ".join((*[function_key(parent)[1] for parent in active_actions], function_key(action)[1]))
            gs.raise_exception(f"recursive Action cycle: {cycle}")
        if action_id in direct_action_indices:
            return direct_action_indices[action_id]

        index = len(direct_actions)
        direct_actions.append(action)
        direct_action_indices[action_id] = index
        active_actions.append(action)
        child_edges: list[tuple[str, int]] = []
        for bound_arg in action.bound_args:
            if isinstance(bound_arg, Action):
                check_kind(action, bound_arg.kind)
                child_edges.append(("action", walk_action(bound_arg)))
            elif isinstance(bound_arg, ActionCollection):
                check_kind(action, bound_arg._kind)
                child_edges.append(("collection", walk_collection(bound_arg)))
        active_actions.pop()
        topology.append(("action", index, action.kind, *function_key(action), tuple(child_edges)))
        return index

    def walk_collection(collection: ActionCollection) -> int:
        collection_id = id(collection)
        if any(collection is parent for parent in active_collections):
            cycle = " -> ".join(item._name for item in (*active_collections, collection))
            gs.raise_exception(f"recursive ActionCollection cycle: {cycle}")
        if collection_id in collection_indices:
            return collection_indices[collection_id]

        actions = collection.actions  # Rejects an unfrozen collection.
        index = len(nested)
        nested.append(actions)
        collection_indices[collection_id] = index
        active_collections.append(collection)
        action_indices = tuple(walk_action(action) for action in actions)
        active_collections.pop()
        topology.append(
            ("collection", index, collection._name, collection._kind, collection._call_args, action_indices)
        )
        return index

    schedule = ActionSchedule()
    schedule.collections = _ScheduleCollections()
    root_indices = []
    for name, collection in collections.items():
        if not isinstance(collection, ActionCollection):
            gs.raise_exception(f"schedule root {name!r} must be an ActionCollection")
        root_indices.append((name, walk_collection(collection)))
        setattr(schedule.collections, name, collection.actions)
    schedule.nested = tuple(nested)
    schedule.actions = tuple(direct_actions)
    schedule.topology_key = (tuple(root_indices), tuple(topology))
    return schedule
