"""Composition core of the simulation engine: Systems own Data, Actions bind functions to it, pipelines run them.

A Data subclass follows the conventions of genesis.utils.array_class: a `@qd.data_oriented` class with `qd.Tensor`
fields allocated by `V()`, named with a `State` / `Info` suffix, created empty with its System and filled by its init
Action.

A System is a backend unit the user never names: it declares the Systems it depends on (Require / Find), the Actions it
hands out (@System.action) and the ActionCollections it receives (@System.action_collection, filled through a
@System.protocol). A concrete Engine owns product context; scene-agnostic base Engine build invokes the factories
registered for that exact concrete class, resolves dependencies, runs every build() and freezes collections. A Pipeline
runs one entry point: a host function or a kernel.
"""

from collections import deque
from collections.abc import Iterable
import functools
import inspect
import keyword
import types
import typing
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Literal, TypeVar, cast, overload

import quadrants as qd

import genesis as gs
from genesis.utils.misc import check_inheritance_depth

if TYPE_CHECKING:
    from genesis.engine.action_schedule import ActionSchedule
    from genesis.utils.qd_ext import GlobalDataContext

# ------------------------------------------------------------------------------------
# --------------------------------------- Data ---------------------------------------
# ------------------------------------------------------------------------------------


@qd.data_oriented
class Data:
    """A System-owned device-data root, automatically exposed through GlobalDataContext after initialization."""


class DataDeclaration:
    """One custom device-data parameter exposed by a System through ``@System.data``."""

    def __init__(self, function: Callable[[object], object]) -> None:
        self.__doc__ = function.__doc__
        self.function = function

    def __set_name__(self, owner: type[object], name: str) -> None:
        self._name = name


# ------------------------------------------------------------------------------------
# ------------------------------------- Actions --------------------------------------
# ------------------------------------------------------------------------------------

# Where an Action may be invoked, derived from the wrapped function:
#   "host"   - Python function, bound method or @qd.kernel, invoked from host code
#   "inline" - @qd.func, invoked inside the device loop of the manager
#   "stage"  - @qd.func(requires_top_level=True), invoked at the top level of a kernel or graph body
ActionKind = Literal["host", "inline", "stage"]
_ActionSpec = Callable[..., object] | tuple[object, ...]


def kind_of(function: Callable[..., Any]) -> ActionKind:
    """Derive the kind of an Action from the markers Quadrants sets on its callables."""
    if getattr(function, "_is_wrapped_kernel", False):
        # A kernel is launched from Python, so it is a valid HostAction implementation. StageAction continues to bind
        # the canonical top-level @qd.func itself for graph composition.
        return "host"
    if not getattr(function, "_is_quadrants_function", False):
        return "host"
    return "stage" if getattr(function, "_qd_requires_top_level", False) else "inline"


# Maps each ActionKind -> the one Action class of that kind, filled when the class is defined
ACTION_TYPES_MAP: dict[ActionKind, type["Action"]] = {}


class Action:
    """A callable closure over one function and its persistent arguments.

    ``invoke(call_args)`` executes ``function(*bound_args, *call_args)`` at the host, inline or graph-stage call site
    identified by ``kind``.
    """

    kind: ClassVar[ActionKind]
    _function: Callable[..., Any] | None
    _bound_args: tuple[object, ...]

    # Each ActionKind has exactly one Action class, HostAction, InlineAction or StageAction, and each of them is final,
    # so a protocol annotation names the kind it requires without ambiguity
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        check_inheritance_depth(cls, Action, max_depth=1)
        kind = vars(cls).get("kind")
        action_kinds = typing.get_args(ActionKind)
        if kind not in action_kinds:
            gs.raise_exception(f"{cls.__name__} must set `kind` to one of {action_kinds}, got {kind!r}.")
        if kind in ACTION_TYPES_MAP:
            gs.raise_exception(
                f"{cls.__name__} declares kind {kind!r}, which {ACTION_TYPES_MAP[kind].__name__} already holds."
            )
        ACTION_TYPES_MAP[kind] = cls

    def __init__(
        self,
        function: Callable[..., Any],
        bound_args: tuple[object, ...],
    ) -> None:
        self._function = function
        self._bound_args = bound_args
        self._owner: System | None = None
        self._declaration: ActionDeclaration | None = None
        self._bound: _BoundAction | None = None
        self._collections: list[ActionCollection] = []

    @classmethod
    def _from_declaration(cls, owner: "System", declaration: "ActionDeclaration") -> "Action":
        action = cls.__new__(cls)
        action._function = None
        action._bound_args = ()
        action._owner = owner
        action._declaration = declaration
        action._bound = None
        action._collections = []
        return action

    def _bind(self, binder: "_ActionBinder | None" = None) -> "_BoundAction":
        return (binder or _ActionBinder()).bind_action(self)

    def invoke(self, call_args: qd.template() = ()) -> None:
        return self._bind().invoke(call_args)


class HostAction(Action):
    kind = "host"


class InlineAction(Action):
    kind = "inline"


class StageAction(Action):
    kind = "stage"


class _BoundAction:
    """Private immutable function partial used by QD adapters."""

    def __init__(
        self,
        function: Callable[..., Any],
        bound_args: tuple[object, ...],
        kind: ActionKind,
    ) -> None:
        self._function = function
        self._bound_args = bound_args
        self.kind = kind

    @qd.pyfunc
    def invoke(self, call_args: qd.template() = ()):
        return self._function(*(self._bound_args + call_args))


class _BoundActionCollection:
    """Frozen bound counterpart of one ActionCollection, safe to pass into a QD function."""

    def __init__(self, actions: tuple[_BoundAction, ...]) -> None:
        self._actions = actions

    @qd.pyfunc
    def invoke(self, call_args: qd.template() = ()):
        for action in qd.static(self._actions):
            action.invoke(call_args)


class ActionCollection:
    """Protocol storage owned by a manager System, filled during build and frozen in invocation order.

    `collection.add(action, rank=r)` registers with an explicit position, and `collection += action` registers
    unranked. The frozen order puts ranked entries first, sorted by rank, then unranked entries in registration order.
    The rank belongs to this registration, so one System can take different positions in different managers.

    A rank is only for deterministic arrangement of otherwise independent entries, such as data-layout or stable
    numbering order. It must never encode a producer/consumer, read/write or barrier dependency. Dependent behavior
    belongs in distinct, explicitly sequenced pipeline phases.
    """

    def __init__(self, owner: "System", name: str, kind: ActionKind, call_args: tuple[str, ...]) -> None:
        self._owner = owner
        self._name = name
        self._kind: ActionKind = kind
        self._call_args = call_args
        self._pending: list[tuple[int | None, Action]] = []
        self._frozen: tuple[Action, ...] | None = None
        self._bound: _BoundActionCollection | None = None

    def _frozen_actions(self) -> tuple[Action, ...]:
        """Return compiler-only frozen storage after Engine build."""
        if self._frozen is None:
            gs.raise_exception(f"{type(self._owner).__name__}.{self._name} is frozen only after build().")
        return self._frozen

    def _bind(self, binder: "_ActionBinder | None" = None) -> _BoundActionCollection:
        return (binder or _ActionBinder()).bind_collection(self)

    def invoke(self, call_args: qd.template() = ()) -> None:
        """Invoke without exposing the inner Action tuple."""
        return self._bind().invoke(call_args)

    def __iadd__(self, action: Action) -> "ActionCollection":
        self.add(action)
        return self

    def add(self, action: Action, *, rank: int | None = None) -> None:
        """Register one Action, optionally at a fixed arrangement rank, from inside a protocol of the owner System.

        `rank` is valid only when entries are dependency-free. Use explicit pipeline phases for dependent Actions.
        """
        where = f"{type(self._owner).__name__}.{self._name}"
        if self._frozen is not None:
            gs.raise_exception(f"{where} is already frozen.")
        if not isinstance(action, Action):
            gs.raise_exception(f"{where}: expected an Action, got {type(action).__name__}.")
        # The manager invokes every Action of this collection from one kind of call site. A host function fails in
        # tracing, a stage inside a loop breaks top-level ordering, and an inline function at top level runs serially.
        if action.kind != self._kind:
            gs.raise_exception(f"{where}: expects {self._kind!r} Actions, got {type(action).__name__}.")
        # The declaration has not run yet. Function-kind and signature validation happen atomically when this frozen
        # collection is first bound, after initialization has made every Data/config argument available.
        # None keeps registration order, and an int >= 0 pins a position that one entry at most can hold
        if rank is not None:
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
                gs.raise_exception(f"{where}: rank must be None or an int >= 0, got {rank!r}.")
            if any(rank_taken == rank for rank_taken, _ in self._pending):
                gs.raise_exception(f"{where}: rank {rank} is already taken.")
        action._collections.append(self)
        self._pending.append((rank, action))

    def freeze(self) -> None:
        """Fix the schedule once Engine has finished build(), since graphs unroll it at compile time."""
        ranked_entries = sorted(
            ((rank, action) for rank, action in self._pending if rank is not None), key=lambda entry: entry[0]
        )
        unranked_entries = [(rank, action) for rank, action in self._pending if rank is None]
        self._frozen = tuple(action for _, action in (*ranked_entries, *unranked_entries))


class _ActionBinder:
    """Private recursive closure resolver shared by every root of a Schedule expansion."""

    _allowed_children: ClassVar[dict[ActionKind, set[ActionKind]]] = {
        "host": {"host"},
        "inline": {"inline"},
        "stage": {"stage", "inline"},
    }

    def __init__(self) -> None:
        self._active_actions: list[Action] = []
        self._active_collections: list[ActionCollection] = []

    @staticmethod
    def _action_name(action: Action) -> str:
        if action._declaration is not None:
            return action._declaration.func.__qualname__
        assert action._function is not None
        function = getattr(action._function, "fn", action._function)
        return function.__qualname__

    def _check_child_kind(self, parent: Action, child_kind: ActionKind) -> None:
        if child_kind not in self._allowed_children[parent.kind]:
            gs.raise_exception(f"{parent.kind} Action cannot invoke {child_kind} Action")

    def bind_action(self, action: Action) -> _BoundAction:
        if action._bound is not None:
            return action._bound
        if any(action is parent for parent in self._active_actions):
            cycle = " -> ".join((*map(self._action_name, self._active_actions), self._action_name(action)))
            gs.raise_exception(f"recursive Action cycle: {cycle}")

        self._active_actions.append(action)
        try:
            if action._declaration is None:
                function = action._function
                bound_args = list(action._bound_args)
            else:
                assert action._owner is not None
                declared_result: Any = action._declaration.func(action._owner)
                function, *bound_args = declared_result if isinstance(declared_result, tuple) else (declared_result,)

            if isinstance(function, types.MethodType):
                function, bound_args = function.__func__, [function.__self__, *bound_args]
            if not isinstance(function, Callable):
                gs.raise_exception(f"{self._action_name(action)} must return a callable first.")
            function_kind = kind_of(function)
            if function_kind != action.kind:
                gs.raise_exception(
                    f"{self._action_name(action)} declared {action.kind!r}, but the function is {function_kind!r}."
                )

            resolved_args: list[object] = []
            for bound_arg in bound_args:
                if isinstance(bound_arg, Action):
                    self._check_child_kind(action, bound_arg.kind)
                    resolved_args.append(self.bind_action(bound_arg))
                elif isinstance(bound_arg, ActionCollection):
                    self._check_child_kind(action, bound_arg._kind)
                    resolved_args.append(self.bind_collection(bound_arg))
                else:
                    resolved_args.append(bound_arg)

            bound = _BoundAction(function, tuple(resolved_args), action.kind)
            for collection in action._collections:
                if collection._frozen is not None:
                    self._validate_collection_action(collection, bound)
            action._bound = bound
            return action._bound
        finally:
            self._active_actions.pop()

    @staticmethod
    def _validate_collection_action(collection: ActionCollection, action: _BoundAction) -> None:
        where = f"{type(collection._owner).__name__}.{collection._name}"
        function = getattr(action._function, "fn", action._function)
        parameters = list(inspect.signature(function).parameters.values())
        positional_kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        if any(parameter.kind not in positional_kinds for parameter in parameters):
            gs.raise_exception(f"{where}: {function.__qualname__} must take positional parameters only.")
        n_bound = len(action._bound_args)
        if n_bound > len(parameters):
            gs.raise_exception(
                f"{where}: {function.__qualname__} takes {len(parameters)} parameter(s), "
                f"but the Action binds {n_bound}."
            )
        call_args = tuple(parameter.name for parameter in parameters[n_bound:])
        if call_args != collection._call_args:
            gs.raise_exception(
                f"{where}: {function.__qualname__} binds {n_bound} argument(s) and then takes {call_args}, but the "
                f"protocol calls it with {collection._call_args}."
            )

    def bind_collection(self, collection: ActionCollection) -> _BoundActionCollection:
        if collection._bound is not None:
            return collection._bound
        if any(collection is parent for parent in self._active_collections):
            cycle = " -> ".join(item._name for item in (*self._active_collections, collection))
            gs.raise_exception(f"recursive ActionCollection cycle: {cycle}")

        self._active_collections.append(collection)
        try:
            actions = tuple(self.bind_action(action) for action in collection._frozen_actions())
            for action in actions:
                self._validate_collection_action(collection, action)
            collection._bound = _BoundActionCollection(actions)
            return collection._bound
        finally:
            self._active_collections.pop()


# ------------------------------------------------------------------------------------
# ----------------------------------- Declarations -----------------------------------
# ------------------------------------------------------------------------------------

T = TypeVar("T")
A = TypeVar("A", bound=Action)
F = TypeVar("F", bound=Callable[..., Any])

# "required" - Require: a missing System fails Engine.build()
# "optional" - Find: a missing System resolves to None
DepLabel = Literal["required", "optional"]


class Dep(Generic[T]):
    """Static dependency declaration, resolved into the instance dict by Engine before build()."""

    label: ClassVar[DepLabel]

    def __init__(self, system_type: type[T]) -> None:
        self.system_type = system_type

    def __set_name__(self, owner: type[object], name: str) -> None:
        self._name = name


class Require(Dep[T]):
    """Mandatory dependency, with `self.<name>` typed as T."""

    label = "required"

    @overload
    def __get__(self, instance: None, owner: type[object]) -> "Require[T]": ...
    @overload
    def __get__(self, instance: object, owner: type[object]) -> T: ...
    def __get__(self, instance: object | None, owner: type[object]) -> "Require[T] | T":
        # Non-data descriptor: after resolution the System lives in the instance dict and shadows this
        if instance is None:
            return self
        gs.raise_exception(f"{self._name} is resolved by Engine before build().")


class Find(Dep[T]):
    """Optional dependency, with `self.<name>` typed as T | None."""

    label = "optional"

    @overload
    def __get__(self, instance: None, owner: type[object]) -> "Find[T]": ...
    @overload
    def __get__(self, instance: object, owner: type[object]) -> T | None: ...
    def __get__(self, instance: object | None, owner: type[object]) -> "Find[T] | T | None":
        # Non-data descriptor: after resolution the System lives in the instance dict and shadows this
        if instance is None:
            return self
        gs.raise_exception(f"{self._name} is resolved by Engine before build().")


class ActionCollectionDeclaration:
    """Class-level declaration created by @System.action_collection, whose method body documents the protocol.

    System.__init__ creates the ActionCollection in the instance dict, which shadows this descriptor.
    """

    def __init__(self, function: Callable[..., object], kind: ActionKind, call_args: tuple[str, ...]) -> None:
        self.__doc__ = function.__doc__
        self.kind: ActionKind = kind
        self.call_args = tuple(call_args)

    @overload
    def __get__(self, instance: None, owner: type[object]) -> "ActionCollectionDeclaration": ...
    @overload
    def __get__(self, instance: "System", owner: type[object]) -> ActionCollection: ...
    def __get__(
        self, instance: "System | None", owner: type[object]
    ) -> "ActionCollectionDeclaration | ActionCollection":
        return self


class ActionDeclaration(Generic[A]):
    """Class-level declaration created by @System.action."""

    def __init__(self, function: Callable[..., _ActionSpec], action_type: type[A]) -> None:
        self.__doc__ = function.__doc__
        self.func = function
        self.action_type = action_type

    def __set_name__(self, owner: type[object], name: str) -> None:
        self._name = name

    @overload
    def __get__(self, instance: None, owner: type[object]) -> "ActionDeclaration[A]": ...
    @overload
    def __get__(self, instance: "System", owner: type[object]) -> A: ...
    def __get__(self, instance: "System | None", owner: type[object]) -> "ActionDeclaration[A] | A":
        if instance is None:
            return self
        action = cast(A, self.action_type._from_declaration(instance, self))
        instance.__dict__[self._name] = action
        return action


def wrap_protocol(method: F, doc: str | None, references: tuple[str, ...]) -> F:
    """Wrap a protocol method with its build-time checks, and attach its guide and references."""
    signature = inspect.signature(method)
    hints: dict[str, Any] | None = None

    @functools.wraps(method)
    def wrapper(self: "System", *args: object, **kwargs: object) -> object:
        nonlocal hints
        where = f"{type(self).__name__}.{method.__name__}"
        # A protocol changes the schedule that build() freezes
        if not self.is_building:
            gs.raise_exception(f"{where} is a protocol, valid only during build().")
        # Resolved on the first call, since annotations may name classes defined after the decorated method
        if hints is None:
            hints = typing.get_type_hints(method)
        for name, value in signature.bind(self, *args, **kwargs).arguments.items():
            expected_type = hints.get(name)
            if (
                isinstance(expected_type, type)
                and issubclass(expected_type, Action)
                and not isinstance(value, expected_type)
            ):
                gs.raise_exception(f"{where}: `{name}` requires {expected_type.__name__}, got {type(value).__name__}.")
        return method(self, *args, **kwargs)

    if doc is not None or references:
        reference_lines = [f"Guide: {doc}"] if doc is not None else []
        reference_lines += [f"Reference: {reference}" for reference in references]
        wrapper.__doc__ = "\n\n".join(
            filter(None, [inspect.cleandoc(wrapper.__doc__ or ""), "\n".join(reference_lines)])
        )
    return cast(F, wrapper)


# ------------------------------------------------------------------------------------
# -------------------------------------- Systems -------------------------------------
# ------------------------------------------------------------------------------------


@qd.data_oriented
class System:
    """One backend unit of the simulation that owns its Data and composes with other Systems.

    A concrete Engine owns any product-specific construction context. Base System has no Scene contract. Behavior is
    shared by composition: System accepts one level of inheritance only.
    """

    # Read-only mapping from attribute name to its Require / Find declaration
    deps: ClassVar[MappingProxyType[str, Dep]] = MappingProxyType({})
    # Names of the declared ActionCollections
    action_collections: ClassVar[tuple[str, ...]] = ()
    # Names of the declared Actions
    actions: ClassVar[tuple[str, ...]] = ()
    # Names of custom data parameters declared with @System.data
    data_declarations: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Single-level inheritance with no intermediate bases or mixins, so behavior is composed
        check_inheritance_depth(cls, System, max_depth=1)
        cls.deps = MappingProxyType({name: dep for name, dep in vars(cls).items() if isinstance(dep, Dep)})
        cls.action_collections = tuple(
            name for name, value in vars(cls).items() if isinstance(value, ActionCollectionDeclaration)
        )
        cls.actions = tuple(name for name, value in vars(cls).items() if isinstance(value, ActionDeclaration))
        cls.data_declarations = tuple(name for name, value in vars(cls).items() if isinstance(value, DataDeclaration))

    def __init__(self) -> None:
        self.is_building = False
        # The Systems of the Engine, which it hands over once the System set is fixed (see find)
        self._systems: MappingProxyType[type["System"], "System"] | None = None
        self._custom_data_roots: dict[str, object] = {}
        for name in type(self).action_collections:
            declaration = vars(type(self))[name]
            self.__dict__[name] = ActionCollection(self, name, declaration.kind, declaration.call_args)

    @overload
    @staticmethod
    def protocol(method: F, /) -> F: ...
    @overload
    @staticmethod
    def protocol(*, doc: str | None = None, references: tuple[str, ...] = ()) -> Callable[[F], F]: ...
    @staticmethod
    def protocol(
        method: F | None = None,
        /,
        *,
        doc: str | None = None,
        references: tuple[str, ...] = (),
    ) -> "F | Callable[[F], F]":
        """Mark an `on_*` method that receives Actions, valid only during build().

        Parameters annotated with HostAction, InlineAction or StageAction state which kind of Action they require, and
        every call is checked against those annotations.

        A protocol is the contract other Systems implement, so it defines every Action parameter:
            Signature  function(*bound_args, <call arguments>), matching the `call_args` of the collection
            Invoked    when and where the manager invokes it, and its order relative to the other Actions
            Must       what the Action has to do, including any count or range it must respect
            Must not   what it may not touch at that point
        A short protocol does this in its docstring (bare `@System.protocol`). A protocol whose contract needs
        derivations or a paper keeps a short docstring and moves the contract into a companion Markdown guide:

            @System.protocol(doc="integrator_protocol.md", references=("https://...",))

        `doc` is a path relative to the file that defines the protocol.
        `references` are links to papers or specifications. Both are appended to the docstring, so they show on hover.
        """
        if method is not None:
            return wrap_protocol(method, None, ())
        return lambda method: wrap_protocol(method, doc, tuple(references))

    @staticmethod
    def action_collection(
        *, kind: ActionKind, call_args: tuple[str, ...] = ()
    ) -> Callable[[Callable[..., object]], ActionCollectionDeclaration]:
        """Declare one ActionCollection owned by a manager System.

        kind       The kind of every Action it accepts: "host", "inline" or "stage" (see System.action). It matches
                   where the manager invokes them: host code, inside its own loop, or at the top level.
        call_args  The names of the arguments the manager supplies after the bound arguments of each Action. A function
                   whose remaining parameters differ in count, order or name is rejected before invocation.

        The decorated method body is never executed, and its docstring documents the protocol.
        """
        return lambda function: ActionCollectionDeclaration(function, kind, call_args)

    @staticmethod
    def data(function: Callable[[object], object]) -> DataDeclaration:
        """Expose one custom device-data parameter through GlobalDataContext.

        Direct System attributes inheriting Data are included automatically and need no declaration. Use this
        decorator for one custom dataclass, @qd.data_oriented object or ndarray per function. The function must return
        a stable object created with the System; initialization may fill its buffers but must not replace the root.
        """
        return DataDeclaration(function)

    @overload
    @staticmethod
    def action(*, kind: Literal["host"]) -> Callable[[Callable[..., _ActionSpec]], ActionDeclaration[HostAction]]: ...
    @overload
    @staticmethod
    def action(
        *, kind: Literal["inline"]
    ) -> Callable[[Callable[..., _ActionSpec]], ActionDeclaration[InlineAction]]: ...
    @overload
    @staticmethod
    def action(*, kind: Literal["stage"]) -> Callable[[Callable[..., _ActionSpec]], ActionDeclaration[StageAction]]: ...
    @staticmethod
    def action(*, kind: ActionKind) -> Callable[[Callable[..., _ActionSpec]], ActionDeclaration[Action]]:
        """Declare one Action of the given kind.

        The decorated method defines the Action closure by returning ``function, *bound_args``.

        `kind` states where the manager invokes the Action. It is required, and it matches the function:

        kind="host"
            Function:  a plain Python function, bound method or @qd.kernel.
            Invoked:   from host code, e.g. the host function of an init pipeline.
            Runs:      from the host, and may allocate, call from_numpy, or directly launch a kernel.
            Use for:   one-time setup that requires the host. Uncommon, since device-capable bookkeeping such as
                       counting or offsets belongs in a kernel.
            Typed as:  HostAction.

        kind="inline"
            Function:  a @qd.func without requires_top_level.
            Invoked:   inside the device loop of the manager, once per iteration or once per participant.
            Runs:      inlined into that loop body, as part of the parallel task of the loop.
            Use for:   small per-element or per-participant work, such as reporting a count or returning the
                       contribution of one element to a reduction.
            Typed as:  InlineAction.

        kind="stage"
            Function:  a @qd.func(requires_top_level=True).
            Invoked:   at the top level of a kernel or graph body.
            Runs:      each of its top-level for-loops becomes a parallel task (a graph node), i.e. one pipeline stage.
            Use for:   work that is parallel over the data of the participant, such as a loop over all its DOFs.
            Typed as:  StageAction.

        Each kind has one valid call site: a host function fails when traced into device code, a stage called inside a
        loop has its top-level loops demoted and loses stage ordering, and an inline function called at top level runs
        as one serial step. Kind and signature are validated before invocation.
        """
        return lambda function: ActionDeclaration(function, ACTION_TYPES_MAP[kind])

    def find(self, system_type: type[T]) -> T | None:
        """Return the System of the given class the Engine holds, or None, valid only during build().

        A dependency is declared on the class with Require / Find unless the two modules would then import each other.
        Such a dependency is looked up here instead, with the class imported inside build(). The lookup adds nothing:
        the System set is fixed before build() runs.
        """
        if not self.is_building:
            gs.raise_exception(f"{type(self).__name__}.find is valid only during build().")
        return self._systems.get(system_type)

    def require(self, system_type: type[T]) -> T:
        """Return the System of the given class the Engine holds, valid only during build() (see find).

        The System must already be held, which a static Require or a registration ensures: the lookup adds nothing.
        """
        system = self.find(system_type)
        if system is None:
            gs.raise_exception(
                f"{type(self).__name__} requires {system_type.__name__}, which the Engine does not hold."
            )
        return system

    def action_schedule(self, **roots: "Action | ActionCollection | None") -> "ActionSchedule":
        """Create a stage schedule whose roots are Actions or collections owned by this System."""
        if self._systems is None or self.is_building:
            gs.raise_exception(f"{type(self).__name__}.action_schedule requires a completed Engine build.")
        return _compile_action_schedule(type(self).__name__, {self}, roots)

    def build(self) -> None:
        pass


def get_required_system_types(owner_types: Iterable[type[Any]]) -> tuple[type[System], ...]:
    """Return the breadth-first, ordered unique closure of required System types absent from the initial owner types."""
    pending_types = deque(owner_types)
    seen_types = set(pending_types)
    required_system_types: list[type[System]] = []
    while pending_types:
        owner_type = pending_types.popleft()
        for dep in owner_type.deps.values():
            if dep.label == "required" and dep.system_type not in seen_types:
                seen_types.add(dep.system_type)
                pending_types.append(dep.system_type)
                required_system_types.append(dep.system_type)
    return tuple(required_system_types)


# ------------------------------------------------------------------------------------
# ------------------------------- Pipelines and engine -------------------------------
# ------------------------------------------------------------------------------------

# "host"   - entry is a Python function
# "device" - entry is a @qd.kernel, which may yield and resume when declared with graph=True and checkpoints=True
PipelineKind = Literal["host", "device"]


class Pipeline:
    """One entry point with its bound arguments, recognized from what it wraps: a Python function or a @qd.kernel.

    `bound_args` are passed to the entry on every launch and on every resume, followed by the call arguments of that
    run: `Pipeline(kernel_substep, engine).run(f)` runs `kernel_substep(engine, f)`. A device Pipeline that supplies a
    GlobalDataContext binds it first, before control objects such as a schedule. Pipeline is final.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        check_inheritance_depth(cls, Pipeline, max_depth=0)

    def __init__(
        self,
        entry: Callable[..., object],
        *bound_args: object,
        yield_callbacks: dict[int, Callable[..., object]] | None = None,
    ) -> None:
        # A Quadrants callable keeps the decorated Python function as `fn`
        entry_name = getattr(getattr(entry, "fn", entry), "__qualname__", repr(entry))
        if getattr(entry, "_is_wrapped_kernel", False):
            kind: PipelineKind = "device"
        elif getattr(entry, "_is_quadrants_function", False):
            gs.raise_exception(f"{entry_name} is a @qd.func, and a device entry is a @qd.kernel that calls it.")
        else:
            kind = "host"
        # Only a checkpointed graph kernel yields, so callbacks anywhere else would never run
        primal = getattr(getattr(entry, "quadrants_callable", entry), "_primal", None)
        has_checkpoints = kind == "device" and getattr(primal, "use_checkpoints", False)
        if yield_callbacks and not has_checkpoints:
            gs.raise_exception(
                f"{entry_name} has no checkpoints, so it takes no yield callbacks. Declare it "
                "@qd.kernel(graph=True, checkpoints=True)."
            )
        # A Quadrants kernel exposes resume(), which Callable cannot express
        self.entry: Any = entry
        self.bound_args = bound_args
        self._prepared_bound_args: tuple[object, ...] | None = None
        self.kind: PipelineKind = kind
        self.yield_callbacks = dict(yield_callbacks or {})

    def _prepare_bound_args(self) -> tuple[object, ...]:
        """Expand every deferred ActionSchedule once, immediately before this Pipeline's first entry call."""
        if self._prepared_bound_args is None:
            from genesis.engine.action_schedule import ActionSchedule

            self._prepared_bound_args = tuple(
                arg.expand() if isinstance(arg, ActionSchedule) else arg for arg in self.bound_args
            )
        return self._prepared_bound_args

    def run(self, *call_args: object) -> None:
        """Run the entry once with the given call arguments, dispatching yield callbacks and resuming until a graph
        kernel completes."""
        args = (*self._prepare_bound_args(), *call_args)
        status = self.entry(*args)
        # A host entry and a plain kernel return None, and a graph kernel returns a GraphStatus that may have yielded
        while status is not None and status.yielded:
            checkpoint = int(status.checkpoint)
            callback = self.yield_callbacks.get(checkpoint)
            if callback is None:
                gs.raise_exception(f"No yield callback for checkpoint {checkpoint}.")
            resume_from = callback(status)
            status = self.entry.resume(*args, from_checkpoint=checkpoint if resume_from is None else int(resume_from))


SystemT = TypeVar("SystemT", bound=System)
CreateFunT = TypeVar("CreateFunT", bound=Callable[["Engine"], System | None])

# Lists rather than sets, since the registration order fixes the order in which an Engine adds the Systems
_system_registry: dict[type["Engine"], list[Callable[["Engine"], System | None]]] = {}


def register_system(engine_cls: type["Engine"]) -> Callable[[CreateFunT], CreateFunT]:
    """Register, for one final Engine class, a concrete-context factory returning one System or None.

    Used as `@register_system(MyEngine)`, once per Engine class the function serves. Registering the same function again
    for the same Engine class does nothing. Base Engine build looks up the exact runtime subclass and passes that
    concrete Engine instance to every registered factory.
    """
    # Engine.__init_subclass__ makes every proper subclass of Engine final
    if not isinstance(engine_cls, type) or not issubclass(engine_cls, Engine) or engine_cls is Engine:
        gs.raise_exception(f"Systems are registered for a final Engine subclass, got {engine_cls!r}.")

    def register(create_fun: CreateFunT) -> CreateFunT:
        # A factory decides from concrete Engine context whether the System exists, which a class cannot express
        if isinstance(create_fun, type) or not isinstance(create_fun, Callable):
            gs.raise_exception(f"register_system takes a function that creates a System, got {create_fun!r}.")
        create_funs = _system_registry.setdefault(engine_cls, [])
        if create_fun not in create_funs:
            create_funs.append(create_fun)
        return create_fun

    return register


def unregister_system(engine_cls: type["Engine"], create_fun: Callable[["Engine"], System | None]) -> None:
    assert isinstance(create_fun, Callable)
    _system_registry[engine_cls].remove(create_fun)


def registered_system_factories(engine_cls: type["Engine"]) -> tuple[Callable[["Engine"], System | None], ...]:
    """Return the exact-class factories for invocation by that concrete Engine."""
    return tuple(_system_registry.get(engine_cls, ()))


def _compile_action_schedule(
    owner_name: str,
    owned_systems: set[System],
    roots: dict[str, Action | ActionCollection | None],
) -> "ActionSchedule":
    from genesis.engine.action_schedule import compile_action_schedule

    if not roots:
        gs.raise_exception(f"{owner_name}.action_schedule requires at least one explicit root.")

    seen: set[int] = set()
    for alias, root in roots.items():
        if not alias.isidentifier() or keyword.iskeyword(alias):
            gs.raise_exception(f"{owner_name}.action_schedule alias {alias!r} is not a valid identifier.")
        if root is None:
            continue
        if not isinstance(root, (Action, ActionCollection)):
            gs.raise_exception(
                f"{owner_name}.action_schedule root {alias!r} must be an Action, ActionCollection or None."
            )
        if id(root) in seen:
            gs.raise_exception(f"{owner_name}.action_schedule binds the same root more than once.")
        seen.add(id(root))
        if root._owner not in owned_systems:
            gs.raise_exception(f"Schedule root {alias!r} is not owned by {owner_name}.")
        kind = root.kind if isinstance(root, Action) else root._kind
        if kind != "stage":
            gs.raise_exception(f"Schedule root {alias!r} must be stage, got {kind!r}.")
        if isinstance(root, ActionCollection):
            root._frozen_actions()

    return compile_action_schedule(roots)


class Engine:
    """The owner of all Systems of one simulation, which adds and builds them and holds the pipelines.

    A concrete Engine inherits directly from Engine and is final: subclassing it again is rejected.
    """

    # Read-only mapping from attribute name to its Require / Find declaration
    deps: ClassVar[MappingProxyType[str, Dep]] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Every Engine subclass is final, so the Systems registered for it apply to exactly one class
        check_inheritance_depth(cls, Engine, max_depth=1)
        cls.deps = MappingProxyType({name: dep for name, dep in vars(cls).items() if isinstance(dep, Dep)})

    def __init__(self) -> None:
        self.systems: dict[type[System], System] = {}
        self._is_resolved = False

    def add_system(self, system: SystemT) -> SystemT:
        """Add a System instance, at most one per class, and return it."""
        if type(system) in self.systems:
            gs.raise_exception(f"{type(system).__name__} is already added.")
        self.systems[type(system)] = system
        return system

    def find(self, system_type: type[SystemT]) -> SystemT | None:
        """Return the System of the given class, or None, valid once build() has fixed the System set.

        For a System whose module would import this Engine's module if the Engine declared it with Require / Find,
        such as one that registers itself for this Engine. The class is imported inside the overriding build(), after
        super().build(). The lookup adds nothing.
        """
        if not self._is_resolved:
            gs.raise_exception(f"{type(self).__name__}.find is valid only once build() has fixed the System set.")
        return self.systems.get(system_type)

    def require(self, system_type: type[SystemT]) -> SystemT:
        """Return the System of the given class, valid once build() has fixed the System set (see find).

        The System must already be held, which a static Require or a registration ensures: the lookup adds nothing.
        """
        system = self.find(system_type)
        if system is None:
            gs.raise_exception(f"{type(self).__name__} requires {system_type.__name__}, which it does not hold.")
        return system

    def action_schedule(self, **roots: "Action | ActionCollection | None") -> "ActionSchedule":
        """Create one explicitly aliased stage schedule from Actions/Collections owned by this Engine's Systems."""
        if not self._is_resolved:
            gs.raise_exception(f"{type(self).__name__}.action_schedule requires a resolved Engine.")
        return _compile_action_schedule(type(self).__name__, set(self.systems.values()), roots)

    def create_context(self) -> "GlobalDataContext":
        """Collect initialized System data parameters and create the first argument of device Pipelines."""
        from genesis.utils.qd_ext import GlobalDataContext

        data: list[object] = []
        seen: set[int] = set()

        def add(system: System, name: str, value: object) -> None:
            if isinstance(value, (tuple, list, dict)):
                gs.raise_exception(
                    f"{type(system).__name__}.{name}: @System.data must return exactly one data parameter."
                )
            try:
                GlobalDataContext((value,))
            except TypeError as error:
                gs.raise_exception(f"{type(system).__name__}.{name}: {error}")
            value_id = id(value)
            if value_id not in seen:
                seen.add(value_id)
                data.append(value)

        for system in self.systems.values():
            for name, value in vars(system).items():
                if isinstance(value, Data):
                    add(system, name, value)
            for name in type(system).data_declarations:
                declaration = vars(type(system))[name]
                value = declaration.function(system)
                previous = system._custom_data_roots.get(name)
                if previous is None:
                    system._custom_data_roots[name] = value
                elif value is not previous:
                    gs.raise_exception(f"{type(system).__name__}.{name}: @System.data root identity changed.")
                add(system, name, system._custom_data_roots[name])

        return GlobalDataContext(data)

    def build(self) -> None:
        """Invoke exact-subclass factories, admit required Systems, resolve/build them, then freeze collections."""
        for create in registered_system_factories(type(self)):
            system = create(self)
            # Importing the module of a System class registers its function, which must not admit the class twice
            if system is not None and type(system) not in self.systems:
                self.add_system(system)
        # Compute the ordered unique closure of required System types before constructing any of them
        for system_type in get_required_system_types((type(self), *self.systems)):
            self.add_system(system_type())
        # Resolve required and optional dependencies once the complete System set is fixed
        owners: list[Engine | System] = [self, *self.systems.values()]
        for owner in owners:
            for name, dep in owner.deps.items():
                owner.__dict__[name] = self.systems.get(dep.system_type)
        self._is_resolved = True
        # Protocols compose Actions only while the build window is open.
        for system in self.systems.values():
            system._systems = MappingProxyType(self.systems)
            system.is_building = True
        for system in self.systems.values():
            system.build()
        # Closing the window freezes every ActionCollection's topology.
        for system in self.systems.values():
            system.is_building = False
            for name in type(system).action_collections:
                system.__dict__[name].freeze()
