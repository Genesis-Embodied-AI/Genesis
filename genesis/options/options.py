from pydantic import BaseModel, ValidationError

import genesis as gs
import genesis.utils.repr as ru
from genesis.repr_base import RBC
from genesis.styles import colors, formats


class Options(BaseModel, RBC):
    """
    This is the base class for all `gs.options.*` classes. An `Options` object is a group of parameters for setting a specific component in the scene.

    Note
    ----
    This class should *not* be instantiated directly.

    Tip
    ----
    We build multiple classes based on this concept throughout Genesis, such as `gs.options.morphs`, `gs.renderers`, `gs.surfaces`, and `gs.textures`. Note that some of them, although inheriting from `Options`, are accessible directly under the `gs` namespace for convenience.
    """

    def __init__(self, **data):
        # make sure input parameters are supported
        allowed_params = self.model_fields.keys()
        for key in data.keys():
            if key not in allowed_params:
                gs.raise_exception(f"Unrecognized attribute: {key}")

        # format pydantic error message to be more informative
        try:
            super().__init__(**data)
        except ValidationError as e:
            errors = e.errors()[0]
            gs.raise_exception(f"Invalid '{errors['loc'][0]}': {errors['msg'].lower()}.")

    def copy_attributes_from(self, options, override=False):
        for field in options.model_fields:
            if field in self.model_fields:
                if override or getattr(self, field) is None:
                    setattr(self, field, getattr(options, field))

    @classmethod
    def _repr_type(cls):
        return f"<{cls.__module__}.{cls.__qualname__}>".replace("genesis", "gs")

    def __repr__(self):
        if not __debug__:
            self.__colorized__repr__()

    def __colorized__repr__(self) -> str:
        property_attrs = self.__dict__.keys()
        max_attr_len = max([len(attr) for attr in property_attrs])

        repr_str = f"{colors.CORN}{'─' * (max_attr_len + 3)} {formats.BOLD}{formats.ITALIC}{self._repr_type()}{formats.RESET} {colors.CORN}{'─' * (max_attr_len + 3)}\n"

        for attr in property_attrs:
            formatted_str = f"{colors.BLUE}'{attr}'{formats.RESET}"

            content = ru.brief(getattr(self, attr))
            idx = content.find(">")
            formatted_content = f"{colors.MINT}{formats.ITALIC}{content[:idx + 1]}{formats.RESET}{colors.MINT}{content[idx + 1:]}{formats.RESET}"
            # in case it's multi-line
            formatted_content = formatted_content.replace("\n", "\n" + " " * (max_attr_len + 4))

            repr_str += f"{formatted_str:>{max_attr_len + 17}}{colors.GRAY}:{formats.RESET} {formatted_content}\n"

        return repr_str

    def copy(self):
        return self.model_copy()
