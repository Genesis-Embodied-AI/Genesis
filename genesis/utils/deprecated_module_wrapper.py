import sys
from types import ModuleType
import genesis as gs


class _DeprecatedModuleWrapper(ModuleType):
    """
    A module wrapper that shows a deprecation warning when accessed.
    This allows us to support the old module name while
    warning users to update their imports.
    """

    def __init__(self, actual_module, old_name, new_name):
        super().__init__(old_name)
        self._actual_module = actual_module
        self._old_name = old_name
        self._new_name = new_name
        self._warned = False
        self.__file__ = getattr(actual_module, "__file__", None)
        self.__package__ = ".".join(old_name.split(".")[:-1])

    def __getattr__(self, name):
        if not self._warned:
            gs.logger.warning(f"Deprecated import: {self._old_name} has been renamed to {self._new_name}.")
            self._warned = True
        return getattr(self._actual_module, name)

    def __dir__(self):
        return dir(self._actual_module)


def create_virtual_deprecated_module(module_name: str, deprecated_name: str) -> None:
    """
    Call from new module with:
    - module_name=__name__
    - deprecated_name is full dotted path, e.g.
      "genesis.engine.solvers.rigid.rigid_solver_decomp"
    """
    current_module = sys.modules[module_name]
    wrapper = _DeprecatedModuleWrapper(current_module, deprecated_name, module_name)
    sys.modules[deprecated_name] = wrapper
