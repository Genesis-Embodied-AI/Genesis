from pyglet.window.key import symbol_string


class Keybindings:
    def __init__(self, map: dict[str, int] = {}, **kwargs: dict[str, int]):
        self._map: dict[str, int] = {**map, **kwargs}
    
    def __getattr__(self, name: str) -> int:
        if name in self._map:
            return self._map[name]
        raise AttributeError(f"Action '{name}' not found in keybindings.")

    def as_instruction_texts(self, padding, exclude: tuple[str]) -> list[str]:
        width = 4 + padding
        return [
            f"{'[' + symbol_string(self._map[action]).lower():>{width}}]: " +
            action.replace('_', ' ') for action in self._map.keys() if action not in exclude
        ]
    
    def extend(self, mapping: dict[str, int], replace_only: bool = False) -> None:
        current_keys = self._map.keys()
        for action, key in mapping.items():
            if replace_only and action not in self._map:
                raise KeyError(f"Action '{action}' not found. Available actions: {list(self._map.keys())}")
            if key in current_keys:
                raise ValueError(f"Key '{symbol_string(key)}' is already assigned to another action.")
            self._map[action] = key