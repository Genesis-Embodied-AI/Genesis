import inspect

import genesis as gs
import genesis.utils.repr as ru
from genesis.styles import colors, formats, styless


class RBC:
    """
    REPR Base Class.
    All class that inherits this class will have an artist-level __repr__ method that prints out all the properties (decorated by @property) of the class, ordered by length of the property name.
    """

    @classmethod
    def _repr_type(cls):
        return f"<gs.{cls.__name__}>"

    def _repr_briefer(self):
        repr_str = self._repr_type()
        if hasattr(self, "id"):
            repr_str += f"(id={self.id})"
        return repr_str

    def _repr_brief(self):
        repr_str = self._repr_type()
        if hasattr(self, "id"):
            repr_str += f": {self.id}"
        if hasattr(self, "idx"):
            repr_str += f", idx: {self.idx}"
        if hasattr(self, "morph"):
            repr_str += f", morph: {self.morph}"
        if hasattr(self, "material"):
            repr_str += f", material: {self.material}"
        return repr_str

    def _is_debugger(self) -> bool:
        """Detect if running under a debugger (VSCode or PyCharm)."""
        for frame in inspect.stack():
            if any(module in frame.filename for module in ("debugpy", "ptvsd", "pydevd")):
                return True
        return False

    def __repr__(self):
        if not self._is_debugger():
            return self.__colorized__repr__()

    def __colorized__repr__(self) -> str:
        all_attrs = self.__dir__()
        property_attrs = []

        for attr in all_attrs:
            if isinstance(getattr(self.__class__, attr, None), property):
                property_attrs.append(attr)

        max_attr_len = max([len(attr) for attr in property_attrs])

        repr_str = ""
        # sort property attrs
        property_attrs = sorted(property_attrs, key=lambda x: (len(x.split("_")[0]), x.split("_")[0], len(x)))

        for attr in property_attrs:
            formatted_str = f"{colors.BLUE}'{attr}'{formats.RESET}"

            # content example: <gs.List>(len=0, [])
            try:
                content = ru.brief(getattr(self, attr))
            except:
                continue
            idx = content.find(">")
            # format with italic and color
            formatted_content = f"{colors.MINT}{formats.ITALIC}{content[:idx + 1]}{formats.RESET}{colors.MINT}{content[idx + 1:]}{formats.RESET}"
            # in case it's multi-line
            if isinstance(getattr(self, attr), gs.List):
                # 4 = 2 x ' + : + space
                offset = max_attr_len + 4
            else:
                # offset by class name length
                offset = max_attr_len + idx + 7

            formatted_content = formatted_content.replace("\n", "\n" + " " * offset)

            repr_str += f"{formatted_str:>{max_attr_len + 17}}{colors.GRAY}:{formats.RESET} {formatted_content}\n"

        # length of the first line
        first_line = styless(repr_str.split("\n")[0])
        header_len = len(first_line)
        line_len = header_len - len(self._repr_type()) - 2
        left_line_len = line_len // 2
        right_line_len = line_len - left_line_len

        # minimum length need to match the first colon
        min_line_len = len(first_line.split(":")[0])
        left_line_len = max(left_line_len, min_line_len)
        right_line_len = max(right_line_len, min_line_len)

        repr_str = (
            f"{colors.CORN}{'─' * left_line_len} {formats.BOLD}{formats.ITALIC}{self._repr_type()}{formats.RESET} {colors.CORN}{'─' * right_line_len}\n"
            + repr_str
        )

        return repr_str

    def __format__(self, format_spec):
        repr_str = self._repr_type()
        return repr_str
