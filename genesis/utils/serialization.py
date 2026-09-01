"""Export what Genesis holds to a file that anyone can load back, and load one back.

A file holds JavaScript Object Notation (JSON) and arrays, and what each value in it is comes from the type the
loader expects: the caller says what it exported and what it is loading, and the declared types of the descriptions
and options take it from there. So a file holds no module path, no import, and no code, and a class name appears only
where a declared type has subclasses and the file must say which one.
"""

import dataclasses
import hashlib
import io
import json
import pathlib
import types
import typing
import zipfile
from collections.abc import Sequence, Set
from enum import IntEnum
from functools import partial
from typing import Callable, NamedTuple

import numpy as np

import genesis as gs
from genesis.options.options import Options

# What every member of an archive is stamped with, so a file is the bytes its scene comes to and nothing else.
EPOCH = (1980, 1, 1, 0, 0, 0)
_SOURCE_DIGEST: str | None = None
MANIFEST_NAME = "scene.json"
ARRAY_MEMBER = "arrays.npz"


class Exporting(NamedTuple):
    """What a class exporting itself is given: somewhere to put an array, and the export of any value it holds."""

    array: Callable[[np.ndarray], int]
    value: Callable[..., object]


class Loading(NamedTuple):
    """What a class loading itself back is given: the arrays a file holds, and the load of any value it holds."""

    array: Callable[[int], np.ndarray]
    value: Callable[..., object]


class SerializationMixin:
    """Declares a class serialisable, the class itself saying how it travels in a file.

    A class whose declared fields hold its values needs none of this: those fields are what a file carries. Mix it in
    where the fields are insufficient, for example geometry that belongs in an array, or a constructor that works on
    what it is given rather than keeping it. The registry beside it does the same for a class from another library,
    which cannot be given a base.
    """

    def export(self, exporting: Exporting):
        """Return everything this instance holds in a form JSON can carry.

        The 'exporting' handle sets arrays aside.

        Must be overridden by the class declaring itself serialisable.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, raw, loading: Loading) -> "SerializationMixin":
        """Recreate the instance 'export' encoded, reading back through 'loading' the arrays it put aside.

        Must be overridden by the class declaring itself serialisable.
        """
        raise NotImplementedError


# This dict records how to write a class from another library to a file and read it back. Such a class cannot
# declare this itself. '_class_codec' walks the method resolution order, so a subclass falls back to its base's pair.
_REGISTERED: dict[type, tuple[Callable, Callable]] = {}


def register(cls: type, export: Callable, load: Callable) -> None:
    """Record how a class from another library travels in a file, which lets a file hold a class Genesis does not own.

    'export(value, exporting)' returns what JSON can hold and 'load(raw, loading)' gives the value back, both of them
    handed an 'Exporting' or a 'Loading' to put arrays aside and to export or load the values the class holds. A class
    Genesis declares mixes in 'SerializationMixin' and states the same two things as its own methods.
    """
    _REGISTERED[cls] = (export, load)


def _exported_array(value: np.ndarray, exporting: Exporting) -> int:
    """Store the array in the file's array member and return its index, which the values carry in its place."""
    return exporting.array(value)


def _loaded_array(raw: int, loading: Loading) -> np.ndarray:
    """Return the stored array the exported index points at."""
    return loading.array(raw)


register(np.ndarray, _exported_array, _loaded_array)


def _class_codec(cls) -> tuple[Callable, Callable] | None:
    """Return the export and load pair for a class.

    A self-serialisable class supplies its own methods; otherwise its own or its base's registered pair applies.
    """
    if not isinstance(cls, type):
        return None
    if issubclass(cls, SerializationMixin):
        return (cls.export, cls.load)
    for held in cls.__mro__:
        if held in _REGISTERED:
            return _REGISTERED[held]
    return None


# The names the descriptions annotate their fields with that are only imported for type checking, since importing
# them at runtime would create a cycle: a parser imports the descriptions, and the options import the parsers.
_DEFERRED_TYPES: dict[str, type[Options]] = {}


def _deferred_types() -> dict[str, type[Options]]:
    """The annotation names a description module imports for type checking alone, resolved here to break the cycle."""
    if not _DEFERRED_TYPES:
        from genesis.engine.materials.base import Material
        from genesis.engine.scene import SceneOptions
        from genesis.options.morphs import Morph
        from genesis.options.surfaces import Surface

        _DEFERRED_TYPES.update(Material=Material, SceneOptions=SceneOptions, Morph=Morph, Surface=Surface)
    return _DEFERRED_TYPES


def _declared_fields(cls: type) -> dict[str, type]:
    """Return the declared type of each field of a class. Those types drive how a value loads back."""
    if issubclass(cls, Options):
        return {name: info.annotation for name, info in cls.model_fields.items()}
    return typing.get_type_hints(cls, localns=_deferred_types())


def _unwrap_annotation(expect):
    """What a declaration asks for, with the wrappers that do not change it taken off.

    A file says by holding null whether an optional value is there, so 'X | None' asks for X. What 'Annotated'
    carries is for validation, leaving what the value is untouched.
    """
    if typing.get_origin(expect) is typing.Annotated:
        expect = typing.get_args(expect)[0]
    if typing.get_origin(expect) in (types.UnionType, typing.Union):
        stated = [arm for arm in typing.get_args(expect) if arm is not type(None)]
        return _unwrap_annotation(stated[0]) if len(stated) == 1 else expect
    return expect


def _accepted_classes(expect) -> tuple:
    """The classes a declaration accepts, which is one for 'Mesh' and two for 'Box | Sphere', 'None' left out.

    A single-class declaration needs no class name in the file. A declaration accepting several makes the file name
    which class it holds.
    """
    if typing.get_origin(expect) in (types.UnionType, typing.Union):
        return tuple(arm for arm in typing.get_args(expect) if arm is not type(None))
    return (expect,)


def _classes_by_name(expect) -> dict[str, type]:
    """Every class a file may name where a declaration stands, under the name it is written as.

    A declaration accepting 'Morph' accepts every kind of morph, so what may stand there is 'Morph' and everything
    extending it. A name read back is looked up here, so a file naming a class outside the declaration is refused.
    """
    held = {}
    pending = [_unwrap_annotation(arm) for arm in _accepted_classes(expect)]
    while pending:
        cls = pending.pop()
        held[cls.__name__] = cls
        pending.extend(cls.__subclasses__())
    return held


def _is_sequence(cls) -> bool:
    """Whether instances of 'cls' are carried as a JSON list. Sequences and sets qualify, and strings stay scalars."""
    return isinstance(cls, type) and issubclass(cls, (Sequence, Set)) and not issubclass(cls, str)


def _has_declared_fields(expect) -> bool:
    """Tell whether the classes a declaration accepts travel field by field, as dataclasses and options do.

    Every class the declaration accepts must qualify.
    """
    arms = [_unwrap_annotation(arm) for arm in _accepted_classes(expect)]
    return bool(arms) and all(
        isinstance(arm, type) and (dataclasses.is_dataclass(arm) or issubclass(arm, Options)) for arm in arms
    )


def _schema(expect, held: dict, seen: set) -> dict:
    """What every class reachable from a declaration is made of: its field names, or its members where it enumerates.

    A field renamed or an enumeration member renumbered leaves every value in a file intact while changing what it
    means. A file therefore carries this of the classes it holds, and a load compares it class by class, so a class
    that gained a field since keeps loading while one whose fields moved is refused by name.
    """
    expect = _unwrap_annotation(expect)
    for arg in typing.get_args(expect):
        _schema(arg, held, seen)
    if not isinstance(expect, type) or expect in seen:
        return held
    seen.add(expect)
    if issubclass(expect, IntEnum):
        held[expect.__name__] = {member.name: member.value for member in expect}
    elif _has_declared_fields(expect):
        for cls in _classes_by_name(expect).values():
            if cls in seen and cls is not expect:
                continue
            seen.add(cls)
            fields = _declared_fields(cls)
            held[cls.__name__] = sorted(fields)
            for annotation in fields.values():
                _schema(annotation, held, seen)
    return held


def schema(expect: dict) -> dict:
    """What the classes a caller exports or loads are made of, under the name each is written as."""
    held = {}
    for annotation in expect.values():
        _schema(annotation, held, set())
    return held


def _refuse_moved_schema(path: str, written: dict, expect: dict) -> None:
    """Refuse a file whose classes mean something else here, naming the class and what moved in it.

    A class the file holds that no longer exists, a field it holds that is no longer declared, and an enumeration
    member whose value moved all change what the values in the file mean. A field or a member added since does not:
    an option states which of its fields were given, and a description field left out stands at its declared default.
    """
    held = schema(expect)
    for name, stated in written.items():
        current = held.get(name)
        if current is None:
            gs.raise_exception(f"'{path}' holds a '{name}', which this version of Genesis no longer declares.")
        if isinstance(stated, dict):
            moved = sorted(f"{member}={value}" for member, value in stated.items() if current.get(member) != value)
            if moved:
                gs.raise_exception(f"'{path}' was written where '{name}' stated {', '.join(moved)}, which has moved.")
        else:
            gone = sorted(set(stated) - set(current))
            if gone:
                gs.raise_exception(f"'{path}' holds {gone} of '{name}', which this version of Genesis no longer has.")


def source_digest() -> str:
    """Digest the Genesis sources this process runs, so a file says which code wrote it.

    The release version alone answers for nothing between two releases, which is where a branch and a local edit both
    live, and either changes what a scene simulates. Reading the sources answers for both, and the digest is taken
    once per process.
    """
    global _SOURCE_DIGEST
    if _SOURCE_DIGEST is None:
        root = pathlib.Path(gs.__file__).parent
        held = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            held.update(path.relative_to(root).as_posix().encode())
            held.update(path.read_bytes())
        _SOURCE_DIGEST = held.hexdigest()[:16]
    return _SOURCE_DIGEST


def _store_array(value, arrays: list[np.ndarray]) -> int:
    """Store one array with the arrays the file holds and return its index.

    The copy is made contiguous in row order, so the file holds the same bytes however numpy laid the values out.
    """
    arrays.append(np.ascontiguousarray(value))
    return len(arrays) - 1


def _exporting_into(arrays: list[np.ndarray]) -> Exporting:
    """What a class is handed to export itself into this file."""
    return Exporting(array=partial(_store_array, arrays=arrays), value=partial(_export_value, arrays=arrays))


def _loading_from(arrays: list[np.ndarray]) -> Loading:
    """What a class is handed to load itself back out of this file."""
    return Loading(array=arrays.__getitem__, value=partial(_load_value, arrays=arrays))


def _export_value(value, expect, arrays: list[np.ndarray]):
    """Export one value as what JSON can hold, against the type it is declared to be.

    What the declaration settles is left unsaid: a field declared to hold one class says only its values, and only a
    field whose declared type has subclasses says which one it holds.
    """
    expect = _unwrap_annotation(expect)
    if value is None:
        return None
    # A field declared to hold anything tells nothing about its content, so the value records its own class.
    if expect is typing.Any or expect is object:
        return _export_free_form(value, arrays)
    if isinstance(value, (np.generic, IntEnum)):
        return value.name if isinstance(value, IntEnum) else value.item()
    held = _class_codec(type(value))
    if held is not None:
        return held[0](value, _exporting_into(arrays))
    if _has_declared_fields(expect):
        given = vars(value)
        held = {
            name: _export_value(given[name], annotation, arrays)
            for name, annotation in _declared_fields(type(value)).items()
        }
        if issubclass(type(value), Options):
            # An option object carries the names of the fields the user gave, since rebuilding one needs them. They
            # are sorted, so exporting the same scene twice gives the same file.
            held = {"values": held, "given": sorted(value.model_fields_set)}
        return held if _accepted_classes(expect) == (type(value),) else {"@": type(value).__name__, **held}
    origin = typing.get_origin(expect)
    if _is_sequence(origin):
        args = typing.get_args(expect)
        if issubclass(origin, tuple) and len(args) > 1 and args[1] is not Ellipsis:
            return [_export_value(item, arg, arrays) for item, arg in zip(value, args)]
        held = args[0] if args else None
        return [_export_value(item, held, arrays) for item in value]
    if origin is dict:
        key, item = typing.get_args(expect) or (None, None)
        return [[_export_value(k, key, arrays), _export_value(v, item, arrays)] for k, v in value.items()]
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return _export_free_form(value, arrays)
    gs.raise_exception(f"A Genesis file cannot hold {type(value).__name__} where {expect} was declared.")


def _export_free_form(value, arrays: list[np.ndarray]):
    """Export a value whose declaration says nothing about it, such as the entries of a free-form mapping.

    The value records its JSON kind, a named class records how it travels, and it loads back as exported.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {"@dict": [[_export_free_form(k, arrays), _export_free_form(v, arrays)] for k, v in value.items()]}
    if _is_sequence(type(value)):
        return {"@list": [_export_free_form(item, arrays) for item in value]}
    # A value whose declaration says nothing about it records its own class name. This lets a morph carry the meshes
    # it received rather than a path to read them from.
    held = _class_codec(type(value))
    if held is not None:
        return {"@class": type(value).__name__, "raw": held[0](value, _exporting_into(arrays))}
    gs.raise_exception(f"A Genesis file cannot hold {type(value).__name__} among free-form values.")


def _load_free_form(raw, arrays: list[np.ndarray]):
    """Load back one value a free-form mapping holds, as what it was exported as."""
    if not isinstance(raw, dict):
        return raw
    if "@list" in raw:
        return [_load_free_form(item, arrays) for item in raw["@list"]]
    if "@dict" in raw:
        return {_load_free_form(k, arrays): _load_free_form(v, arrays) for k, v in raw["@dict"]}
    if "@class" in raw:
        pending = [SerializationMixin, *_REGISTERED]
        while pending:
            cls = pending.pop()
            pending.extend(cls.__subclasses__())
            if cls.__name__ == raw["@class"]:
                return _class_codec(cls)[1](raw["raw"], _loading_from(arrays))
        gs.raise_exception(f"A Genesis file holds a '{raw['@class']}', which no class says how it is loaded.")
    gs.raise_exception(f"A Genesis file holds a free-form value this version cannot load: {sorted(raw)}.")


def _named_class(raw: dict, expect) -> type:
    """Which class a file says it holds, refused unless the declaration accepts it."""
    if "@" not in raw:
        return _accepted_classes(expect)[0]
    kinds = _classes_by_name(expect)
    cls = kinds.get(raw["@"])
    if cls is None:
        gs.raise_exception(
            f"A Genesis file holds a '{raw['@']}' where {expect} was declared, and Genesis declares no such class "
            f"there (it declares {', '.join(sorted(kinds))})."
        )
    return cls


def _load_value(raw, expect, arrays: list[np.ndarray]):
    """Load back one value as the type it is declared to be, creating only what that declaration allows."""
    if raw is None:
        if typing.get_origin(expect) is typing.Annotated:
            expect = typing.get_args(expect)[0]
        if type(None) not in typing.get_args(expect) and expect is not typing.Any and expect is not object:
            gs.raise_exception(f"A Genesis file holds nothing where a {expect} was declared.")
        return None
    expect = _unwrap_annotation(expect)
    if expect is typing.Any or expect is object:
        return _load_free_form(raw, arrays)
    if isinstance(expect, type) and issubclass(expect, IntEnum):
        return expect[raw]
    held = _class_codec(expect)
    if held is not None:
        return held[1](raw, _loading_from(arrays))
    if _has_declared_fields(expect):
        cls = _named_class(raw, expect)
        declared = _declared_fields(cls)
        if issubclass(cls, Options):
            values = {name: _load_value(raw["values"][name], declared[name], arrays) for name in raw["values"]}
            # These values were validated when the option was authored, so they are not validated again. The fields
            # marked as given are the ones the user gave then, since one option inherits from another only what the
            # receiver left unset.
            return cls.model_construct(_fields_set=set(raw["given"]), **values)
        return cls(**{name: _load_value(raw[name], declared[name], arrays) for name in declared if name in raw})
    origin = typing.get_origin(expect)
    if _is_sequence(origin):
        args = typing.get_args(expect)
        if issubclass(origin, tuple) and len(args) > 1 and args[1] is not Ellipsis:
            return tuple(_load_value(item, arg, arrays) for item, arg in zip(raw, args))
        held = args[0] if args else None
        items = [_load_value(item, held, arrays) for item in raw]
        # An abstract sequence declaration loads as a plain list, a concrete one as the class it names.
        return items if origin in (list, Sequence) else origin(items)
    if origin is dict:
        key, item = typing.get_args(expect) or (None, None)
        return {_load_value(k, key, arrays): _load_value(v, item, arrays) for k, v in raw}
    if isinstance(raw, (bool, int, float, str)):
        return raw
    return _load_free_form(raw, arrays)


def export(path: str, held: dict, redact: dict[str, str]) -> None:
    """Export what a caller holds to a file that loads back as data alone, holding no code and reading no asset.

    'held' names each value it carries, and loading the file asks for those names back. The file is a zip archive of
    one manifest and one member holding every array, and what each value is comes from the declared type of what
    holds it, so loading one creates the descriptions, options and meshes those declarations allow and nothing else.

    'redact' names what no file may hold, each with what stands for it. A directory an asset was read from reaches
    values a caller never sees - the identifier of a mesh, the provenance of another - so the substitution is made
    over the whole manifest rather than value by value.
    """
    arrays: list[np.ndarray] = []
    manifest = {
        "genesis": gs.__version__,
        "source": source_digest(),
        "schema": schema({name: type(value) for name, value in held.items()}),
        "held": {name: _export_value(value, type(value), arrays) for name, value in held.items()},
    }
    text = json.dumps(manifest, indent=1)
    for held_path, stands_for in redact.items():
        text = text.replace(json.dumps(held_path)[1:-1], json.dumps(stands_for)[1:-1])
    member = io.BytesIO()
    # Every array is compressed where it is written: the member holding them is itself an archive, so an archive
    # around it would find nothing left to compress.
    np.savez_compressed(member, **{str(index): array for index, array in enumerate(arrays)})
    with zipfile.ZipFile(path, "w") as archive:
        for name, said in ((MANIFEST_NAME, text.encode()), (ARRAY_MEMBER, member.getvalue())):
            # An archive stamps each member with the time it was written unless it is told one, and two exports of
            # one scene are the same file, so the time is stated rather than taken from the clock.
            held = zipfile.ZipInfo(name, date_time=EPOCH)
            held.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(held, said)


def load(path: str, expect: dict) -> dict:
    """Load back what 'export' wrote, each value as the type the caller says it expects.

    A file arrives from wherever it was written, so anything about it may be wrong: it may be truncated, hold neither
    of the two members a file is written as, or have been exported when a class meant something else. Each failure is
    refused with a message naming what was expected, so a reader knows whether to ask for the file again or for
    another version.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read(MANIFEST_NAME))
            member = np.load(io.BytesIO(archive.read(ARRAY_MEMBER)))
            arrays = [member[str(index)] for index in range(len(member.files))]
    except zipfile.BadZipFile as e:
        gs.raise_exception_from(f"'{path}' is not a Genesis file, or was truncated on its way here.", e)
    except KeyError as e:
        gs.raise_exception_from(
            f"'{path}' is missing what a Genesis file is written as: a '{MANIFEST_NAME}' manifest and an "
            f"'{ARRAY_MEMBER}' member holding its arrays.",
            e,
        )
    except (json.JSONDecodeError, ValueError) as e:
        gs.raise_exception_from(f"'{path}' holds a '{MANIFEST_NAME}' or an '{ARRAY_MEMBER}' Genesis cannot read.", e)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("held"), dict) or "schema" not in manifest:
        gs.raise_exception(f"'{path}' holds a '{MANIFEST_NAME}' that is not the manifest Genesis writes.")
    _refuse_moved_schema(path, manifest["schema"], expect)
    if manifest.get("source") != source_digest():
        gs.logger.warning(
            f"'{path}' was exported by Genesis {manifest.get('genesis')} ({manifest.get('source')}) and is being "
            f"loaded by {gs.__version__} ({source_digest()}). What the file holds still means what it says, and the "
            "simulation it describes may run differently."
        )
    missing = sorted(set(expect) - set(manifest["held"]))
    if missing:
        gs.raise_exception(f"'{path}' holds {sorted(manifest['held'])} rather than the {missing} asked for.")
    # A file written elsewhere may hold anything, so a value that contradicts its declaration is refused here
    # rather than reaching a constructor.
    try:
        return {name: _load_value(manifest["held"][name], annotation, arrays) for name, annotation in expect.items()}
    except (TypeError, KeyError, IndexError, ValueError) as e:
        gs.raise_exception_from(f"'{path}' holds values a Genesis scene is not made of.", e)
