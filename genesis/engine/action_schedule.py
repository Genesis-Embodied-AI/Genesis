"""Expansion of frozen Action control graphs.

An ActionSchedule is the compile-time control counterpart of GlobalDataContext:

- GlobalDataContext is the only device-data discovery root.
- ActionSchedule contains only which Actions execute and how nested Actions are connected.

Context/schedule-based graph Pipelines take the Schedule instead of their manager System. Passing the manager would
make Quadrants traverse its dependency graph as another data root, duplicate GlobalDataContext's role and hide child
Action edges behind arbitrary System attributes. Legacy or subsystem kernels may still take a System directly. The
first Pipeline call expands the explicit roots and snapshots their immutable topology immediately before Quadrants
traces the kernel.
"""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import genesis as gs
from genesis.engine.core import (
    Action,
    ActionCollection,
    _ActionBinder,
    _BoundAction,
    _BoundActionCollection,
)


class _ScheduleRoots:
    """Immutable private namespace of Schedule-local aliases to normalized Action tuples."""

    def __init__(self, roots: Mapping[str, tuple[_BoundAction, ...]]) -> None:
        for name, actions in roots.items():
            object.__setattr__(self, name, actions)

    def __setattr__(self, name: str, value: object) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")


class _ExpandedActionSchedule:
    """Bound recursive Action control graph passed to a graph kernel."""

    def __init__(
        self,
        roots: Mapping[str, tuple[_BoundAction, ...]],
        nested: tuple[tuple[_BoundAction, ...], ...],
        actions: tuple[_BoundAction, ...],
        topology_key: tuple,
    ) -> None:
        object.__setattr__(self, "_roots", _ScheduleRoots(roots))
        object.__setattr__(self, "_nested", nested)
        object.__setattr__(self, "_actions", actions)
        object.__setattr__(self, "_topology_key", topology_key)
        for name, root_actions in roots.items():
            object.__setattr__(self, name, root_actions)

    def __setattr__(self, name: str, value: object) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")


class ActionSchedule:
    """Frozen Action-root mapping expanded and cached by Pipeline immediately before its first call.

    Aliases become direct attributes only on the private expanded object that the graph kernel receives. Recursive
    Action/ActionCollection topology remains private compiler state.
    """

    def __init__(self, roots: Mapping[str, Action | ActionCollection | None]) -> None:
        object.__setattr__(self, "_roots", MappingProxyType(dict(roots)))
        object.__setattr__(self, "_expanded", None)

    def __setattr__(self, name: str, value: object) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def expand(self) -> _ExpandedActionSchedule:
        """Bind and recursively snapshot the roots once, immediately before the first graph call."""
        if self._expanded is None:
            object.__setattr__(self, "_expanded", _expand_action_schedule(self._roots))
        return self._expanded


def _expand_action_schedule(
    roots: Mapping[str, Action | ActionCollection | None],
) -> _ExpandedActionSchedule:
    """Resolve the roots and snapshot every explicit child edge into a cycle-free QD control adapter."""
    binder = _ActionBinder()
    normalized_roots: dict[str, tuple[_BoundAction, ...]] = {}
    for name, root in roots.items():
        if root is None:
            normalized_roots[name] = ()
        elif isinstance(root, Action):
            normalized_roots[name] = (binder.bind_action(root),)
        else:
            normalized_roots[name] = binder.bind_collection(root)._actions

    nested: list[tuple[_BoundAction, ...]] = []
    collection_indices: dict[int, int] = {}
    direct_actions: list[_BoundAction] = []
    direct_action_indices: dict[int, int] = {}
    topology: list[tuple] = []

    def function_key(action: _BoundAction) -> tuple[str, str]:
        function = getattr(action._function, "fn", action._function)
        return function.__module__, function.__qualname__

    def walk_action(action: _BoundAction) -> int:
        action_id = id(action)
        if action_id in direct_action_indices:
            return direct_action_indices[action_id]

        index = len(direct_actions)
        direct_actions.append(action)
        direct_action_indices[action_id] = index
        child_edges: list[tuple[str, int]] = []
        for bound_arg in action._bound_args:
            if isinstance(bound_arg, _BoundAction):
                child_edges.append(("action", walk_action(bound_arg)))
            elif isinstance(bound_arg, _BoundActionCollection):
                child_edges.append(("collection", walk_collection(bound_arg)))
        topology.append(("action", index, action.kind, *function_key(action), tuple(child_edges)))
        return index

    def walk_collection(collection: _BoundActionCollection) -> int:
        collection_id = id(collection)
        if collection_id in collection_indices:
            return collection_indices[collection_id]

        actions = collection._actions
        index = len(nested)
        nested.append(actions)
        collection_indices[collection_id] = index
        action_indices = tuple(walk_action(action) for action in actions)
        topology.append(("collection", index, action_indices))
        return index

    root_indices = []
    for name, actions in normalized_roots.items():
        root_indices.append((name, tuple(walk_action(action) for action in actions)))
    return _ExpandedActionSchedule(
        roots=normalized_roots,
        nested=tuple(nested),
        actions=tuple(direct_actions),
        topology_key=(tuple(root_indices), tuple(topology)),
    )


def compile_action_schedule(roots: Mapping[str, Action | ActionCollection | None]) -> ActionSchedule:
    """Validate explicit root aliases and create their QD control adapter."""
    seen_roots: set[int] = set()
    for name, root in roots.items():
        if name.startswith("_"):
            gs.raise_exception(f"schedule root alias {name!r} is reserved for private compiler state")
        if root is None:
            continue
        if not isinstance(root, (Action, ActionCollection)):
            gs.raise_exception(f"schedule root {name!r} must be an Action, ActionCollection or None")
        if id(root) in seen_roots:
            gs.raise_exception("the same Action/ActionCollection cannot be bound to two Schedule root aliases")
        seen_roots.add(id(root))
        if isinstance(root, ActionCollection):
            root._frozen_actions()
    return ActionSchedule(roots)
