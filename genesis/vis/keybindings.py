from enum import IntEnum
from typing import Callable, NamedTuple


class _LazyPygletKeyModule:
    """
    Lazy load pyglet to avoid premature initialization.

    Note
    ----
    Importing pyglet before OpenGL context is created can lead to segmentation faults on some platforms.
    """

    _module = None

    def __getattr__(self, name):
        if type(self)._module is None:
            from pyglet.window import key as pyglet_key

            type(self)._module = pyglet_key
        return getattr(type(self)._module, name)

    def __dir__(self):
        if type(self)._module is None:
            from pyglet.window import key as pyglet_key

            type(self)._module = pyglet_key
        return dir(type(self)._module)


Key = _LazyPygletKeyModule()

KEY_STRING_TO_CHAR = {
    "backslash": "\\",
    "slash": "/",
    "comma": ",",
    "period": ".",
    "bracketleft": "[",
    "bracketright": "]",
    "semicolon": ";",
    "minus": "-",
    "equal": "=",
}


class KeyAction(IntEnum):
    PRESS = 0
    HOLD = 1
    RELEASE = 2


def get_key_hash(key_code: int, modifiers: int | None, action: KeyAction) -> int:
    """Generate a unique hash for a key combination.

    Parameters
    ----------
    key_code : int
        The key code from pyglet.
    modifiers : int | None
        The modifier keys pressed.
    action : KeyAction
        The type of key action (press, hold, release).

    Returns
    -------
    int
        A unique hash for this key combination.
    """
    return hash((key_code, modifiers, action))


def get_keycode_string(key_code: int) -> str:
    from pyglet.window.key import symbol_string

    symbol = symbol_string(key_code).lower()
    if symbol in KEY_STRING_TO_CHAR:
        return KEY_STRING_TO_CHAR[symbol]
    return symbol


class Keybind(NamedTuple):
    key_code: int
    key_action: KeyAction = KeyAction.PRESS
    name: str = ""
    callback: Callable[[], None] | None = None
    modifiers: int | None = None
    args: tuple = ()
    kwargs: dict = {}

    def key_hash(self) -> int:
        """Generate a unique hash for the keybind based on key code and modifiers."""
        return get_key_hash(self.key_code, self.modifiers, self.key_action)


class Keybindings:
    def __init__(self, keybinds: tuple[Keybind] = ()):
        self._keybinds_map: dict[int, Keybind] = {kb.key_hash(): kb for kb in keybinds}

    def register(self, keybind: Keybind) -> None:
        if keybind.key_hash() in self._keybinds_map:
            existing_kb = self._keybinds_map[keybind.key_hash()]
            raise ValueError(
                f"Key '{get_keycode_string(keybind.key_code)}' is already assigned to '{existing_kb.name}'."
            )
        self._keybinds_map[keybind.key_hash()] = keybind

    def rebind(
        self,
        name: str,
        new_key_code: int | None,
        new_modifiers: int | None = None,
        new_key_action: KeyAction | None = None,
    ) -> None:
        for kb in self._keybinds_map.values():
            if kb.name == name:
                new_kb = Keybind(
                    name=kb.name,
                    key_code=new_key_code or kb.key_code,
                    key_action=new_key_action or kb.key_action,
                    modifiers=new_modifiers or kb.modifiers,
                    callback=kb.callback,
                    args=kb.args,
                    kwargs=kb.kwargs,
                )
                del self._keybinds_map[kb.key_hash()]
                self._keybinds_map[new_kb.key_hash()] = new_kb
                return
        raise ValueError(f"No keybind found with name '{name}'.")

    def get(self, key: int, modifiers: int, key_action: KeyAction) -> Keybind | None:
        key_hash = get_key_hash(key, modifiers, key_action)
        if key_hash in self._keybinds_map:
            return self._keybinds_map[key_hash]

        # Try ignoring modifiers (for keybinds where modifiers=None)
        key_hash_no_mods = get_key_hash(key, None, key_action)
        if key_hash_no_mods in self._keybinds_map:
            return self._keybinds_map[key_hash_no_mods]

        return None

    def get_by_name(self, name: str) -> Keybind | None:
        for kb in self._keybinds_map.values():
            if kb.name == name:
                return kb
        return None

    def __len__(self) -> int:
        return len(self._keybinds_map)

    @property
    def keys(self) -> tuple[str]:
        """Return a list of all registered keys as ASCII characters."""
        return tuple(get_keycode_string(kb.key_code) for kb in self._keybinds_map.values())

    @property
    def keybinds(self) -> tuple[Keybind]:
        """Return a tuple of all registered Keybinds."""
        return tuple(self._keybinds_map.values())
