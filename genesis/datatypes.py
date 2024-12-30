import os

import genesis as gs
from genesis.repr_base import RBC
from genesis.styles import colors, formats, styless


class List(list, RBC):
    """
    Custom list with more informative repr.
    Elements in the list should also inherit from RBC.
    """

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return self.__class__(result)
        else:
            return result

    def common_ancestor(self):
        common_ancestor = None
        if len(self) > 0:
            classes = [obj.__class__ for obj in self]
            for obj_cls in classes[0].__mro__:
                if all(obj_cls in obj.__mro__ for obj in classes[1:]):
                    common_ancestor = obj_cls
                    break
        return common_ancestor

    def common_length(self):
        if len(self) > 0:
            return max([len(obj._repr_type()) for obj in self])
        else:
            return 0

    def _repr_elem(self, elem, common_length=0):
        spaces = common_length - len(elem._repr_type())
        repr_str = " " * spaces + elem._repr_brief()
        return repr_str

    def _repr_elem_colorized(self, elem, common_length=0):
        content = self._repr_elem(elem, common_length)
        idx = content.find(">")
        formatted_content = f"{colors.BLUE}{formats.ITALIC}{content[:idx + 1]}{formats.RESET}{content[idx + 1:]}"
        idx = formatted_content.find(":")
        if idx >= 0:
            formatted_content = f"{formatted_content[:idx]}{colors.GRAY}:{colors.MINT}{formatted_content[idx + 1:]}"
        formatted_content += formats.RESET
        return formatted_content

    def _repr_brief(self):
        repr_str = f"{self._repr_type()}(len={len(self)}, ["

        if len(self) == 0:
            repr_str += "])"

        elif len(self) < 15:
            repr_str += "\n"
            for element in self:
                repr_str += f"    {self._repr_elem(element)},\n"
            repr_str += "])"

        else:
            repr_str += "\n"
            for element in self[:9]:
                repr_str += f"    {self._repr_elem(element)},\n"
            repr_str += "    ...\n"
            for element in self[-1:]:
                repr_str += f"    {self._repr_elem(element)},\n"
            repr_str += "])"

        return repr_str

    def __repr__(self):
        return RBC.__repr__(self)

    def __colorized__repr__(self):
        repr_str = f"{colors.BLUE}{self._repr_type()}(len={colors.MINT}{formats.UNDERLINE}{len(self)}{formats.RESET}{colors.BLUE}, ["

        if len(self) == 0:
            repr_str += f"{colors.BLUE}])"

        else:
            common_length = self.common_length()

            if len(self) < 15 or gs._verbose:
                repr_str += "\n"
                for element in self:
                    repr_str += (
                        f"    {self._repr_elem_colorized(element, common_length)}{colors.GRAY},{formats.RESET}\n"
                    )
                repr_str += f"{colors.BLUE}]){formats.RESET}"

            else:
                repr_str += "\n"
                for element in self[:9]:
                    repr_str += (
                        f"    {self._repr_elem_colorized(element, common_length)}{colors.GRAY},{formats.RESET}\n"
                    )
                repr_str += f"    {colors.GRAY}...{formats.RESET}\n"
                for element in self[-1:]:
                    repr_str += (
                        f"    {self._repr_elem_colorized(element, common_length)}{colors.GRAY},{formats.RESET}\n"
                    )
                repr_str += f"{colors.BLUE}]){formats.RESET}"

        min_len = 15
        if len(self) == 0:
            first_line = styless(repr_str)
            header = self._repr_type()
            line_len = min_len

        else:
            common_class_name = self.common_ancestor()._repr_type()
            first_line = styless(repr_str.split("\n")[1])
            header = f"{self._repr_type()} of {common_class_name}"
            header_len = len(first_line)
            line_len = min(max(min_len, header_len - len(header) - 2), os.get_terminal_size()[0] - len(header) - 2)

        left_line_len = line_len // 2
        right_line_len = line_len - left_line_len

        repr_str = (
            f"{colors.CORN}{'─' * left_line_len} {formats.BOLD}{formats.ITALIC}{header}{formats.RESET} {colors.CORN}{'─' * right_line_len}{formats.RESET}\n"
            + repr_str
        )

        return repr_str
