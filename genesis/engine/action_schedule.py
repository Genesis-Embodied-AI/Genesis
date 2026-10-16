"""Recursive compilation of frozen Action control graphs.

An ActionSchedule is the compile-time control counterpart of GlobalDataContext:

- GlobalDataContext is the only device-data discovery root.
- ActionSchedule contains only which frozen Actions execute and how nested Actions are connected.

Graph kernels take the Schedule instead of their manager System. Passing the manager would make Quadrants traverse its
dependency graph as another data root, duplicate GlobalDataContext's role and hide child Action edges behind arbitrary
System attributes. The compiler snapshots only explicit edges carried in Action bound arguments, so the graph topology
is immutable, inspectable and cycle-checked before Quadrants traces a kernel.
"""

from collections.abc import Mapping

from genesis.engine.core import Action, ActionCollection


class ActionSchedule:
    """Frozen recursive Action control graph with no device-data ownership.

    Root collection names are exposed as attributes for graph kernels. Nested Action/ActionCollection edges are
    discovered only through explicit Action bound arguments. ``topology_key`` records the structural graph for
    diagnostics and cache integration; root/nested attributes retain the exact frozen Action tuples QD statically
    unrolls.
    """


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
            raise TypeError(f"{parent.kind} Action cannot invoke {child_kind} Action")

    def walk_action(action: Action) -> int:
        action_id = id(action)
        if any(action is parent for parent in active_actions):
            cycle = " -> ".join((*[function_key(parent)[1] for parent in active_actions], function_key(action)[1]))
            raise ValueError(f"recursive Action cycle: {cycle}")
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
            raise ValueError(f"recursive ActionCollection cycle: {cycle}")
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
    root_indices = []
    for name, collection in collections.items():
        if not isinstance(collection, ActionCollection):
            raise TypeError(f"schedule root {name!r} must be an ActionCollection")
        root_indices.append((name, walk_collection(collection)))
        setattr(schedule, name, collection.actions)
    for index, actions in enumerate(nested):
        setattr(schedule, f"nested_{index}", actions)
    for index, action in enumerate(direct_actions):
        setattr(schedule, f"action_{index}", action)
    schedule.topology_key = (tuple(root_indices), tuple(topology))
    return schedule
