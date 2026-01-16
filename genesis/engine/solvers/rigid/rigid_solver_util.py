import gstaichi as ti

import genesis.utils.array_class as array_class


@ti.func
def func_wakeup_entity_and_its_temp_island(
    i_e,
    i_b,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    contact_island_state: array_class.ContactIslandState,
):
    # Note: Original function handled non-hibernated & fixed entities.
    # Now, we require a properly hibernated entity to be passed in.
    island_idx = contact_island_state.entity_island[i_e, i_b]

    for ei in range(contact_island_state.island_entity.n[island_idx, i_b]):
        entity_ref = contact_island_state.island_entity.start[island_idx, i_b] + ei
        entity_idx = contact_island_state.entity_id[entity_ref, i_b]

        is_entity_hibernated = entities_state.hibernated[entity_idx, i_b]

        if is_entity_hibernated:
            contact_island_state.entity_idx_to_next_entity_idx_in_hibernated_island[entity_idx, i_b] = -1

            entities_state.hibernated[entity_idx, i_b] = False
            n_awake_entities = ti.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
            rigid_global_info.awake_entities[n_awake_entities, i_b] = entity_idx

            n_dofs = entities_info.n_dofs[entity_idx]
            base_entity_dof_idx = entities_info.dof_start[entity_idx]
            base_awake_dof_idx = ti.atomic_add(rigid_global_info.n_awake_dofs[i_b], n_dofs)
            for i in range(n_dofs):
                i_d = base_entity_dof_idx + i
                dofs_state.hibernated[i_d, i_b] = False
                rigid_global_info.awake_dofs[base_awake_dof_idx + i, i_b] = i_d

            n_links = entities_info.n_links[entity_idx]
            base_entity_link_idx = entities_info.link_start[entity_idx]
            base_awake_link_idx = ti.atomic_add(rigid_global_info.n_awake_links[i_b], n_links)
            for i in range(n_links):
                i_l = base_entity_link_idx + i
                links_state.hibernated[i_l, i_b] = False
                rigid_global_info.awake_links[base_awake_link_idx + i, i_b] = i_l

            for i_g in range(entities_info.geom_start[entity_idx], entities_info.geom_end[entity_idx]):
                geoms_state.hibernated[i_g, i_b] = False


# ------------------------------------------------------------------------------------
# ------------------------ Backward Compatibility Shim ------------------------------
# ------------------------------------------------------------------------------------
# This section creates a deprecated alias module for the old name 'rigid_solver_util_decomp'

import sys
import types


def _show_deprecation_warning_rigidsolverutil():
    """Show a deprecation warning for the old module name."""
    try:
        import genesis as gs
        gs.logger.warning(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════════════╗\n"
            f"║                         DEPRECATION WARNING                              ║\n"
            f"╠══════════════════════════════════════════════════════════════════════════╣\n"
            f"║ The module 'rigid_solver_util_decomp' has been renamed to               ║\n"
            f"║ 'rigid_solver_util'                                                      ║\n"
            f"║                                                                          ║\n"
            f"║ Please update your imports:                                              ║\n"
            f"║   OLD: from genesis.engine.solvers.rigid import rigid_solver_util_decomp ║\n"
            f"║   NEW: from genesis.engine.solvers.rigid import rigid_solver_util        ║\n"
            f"║                                                                          ║\n"
            f"║ This compatibility shim will be removed in a future release.            ║\n"
            f"╚══════════════════════════════════════════════════════════════════════════╝"
        )
    except:
        import warnings
        warnings.warn(
            "Module 'genesis.engine.solvers.rigid.rigid_solver_util_decomp' has been renamed to "
            "'genesis.engine.solvers.rigid.rigid_solver_util'. Please update your imports. "
            "This compatibility shim will be removed in a future release.",
            DeprecationWarning,
            stacklevel=4,
        )


class _DeprecatedModuleWrapper_rigidsolverutil(types.ModuleType):
    """
    A module wrapper that shows a deprecation warning when accessed.
    This allows us to support the old module name 'rigid_solver_util_decomp' while
    warning users to update their imports.
    """

    def __init__(self, actual_module, old_name, new_name):
        super().__init__(old_name)
        self._actual_module = actual_module
        self._old_name = old_name
        self._new_name = new_name
        self._warned = False
        self.__file__ = getattr(actual_module, '__file__', None)
        self.__package__ = '.'.join(old_name.split('.')[:-1])
        
        try:
            import inspect
            frame = inspect.currentframe()
            for _ in range(10):
                if frame is None:
                    break
                frame = frame.f_back
                if frame and 'rigid_solver_util_decomp' in str(frame.f_code.co_filename):
                    continue
                if frame and frame.f_code.co_name in ('_find_and_load', '_handle_fromlist', 'import_module'):
                    _show_deprecation_warning_rigidsolverutil()
                    self._warned = True
                    break
        except:
            pass

    def __getattr__(self, name):
        if not self._warned:
            _show_deprecation_warning_rigidsolverutil()
            self._warned = True
        return getattr(self._actual_module, name)

    def __dir__(self):
        return dir(self._actual_module)


_current_module_rigidsolverutil = sys.modules[__name__]
_deprecated_name_rigidsolverutil = "genesis.engine.solvers.rigid.rigid_solver_util_decomp"
_wrapper_rigidsolverutil = _DeprecatedModuleWrapper_rigidsolverutil(
    _current_module_rigidsolverutil, _deprecated_name_rigidsolverutil, __name__
)
sys.modules[_deprecated_name_rigidsolverutil] = _wrapper_rigidsolverutil
