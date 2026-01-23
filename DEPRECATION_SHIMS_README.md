# Deprecation Shims for Renamed Rigid Solver Modules

## Summary

Added backward compatibility shims to all renamed rigid solver modules to support legacy code that imports the old `*_decomp` module names. When users import the old names, they will see a clear deprecation warning but their code will continue to work.

## Modules Updated

The following 9 modules in `genesis/engine/solvers/rigid/` now have deprecation shims:

1. **rigid_solver.py** - Old name: `rigid_solver_decomp`
2. **collider.py** - Old name: `collider_decomp`
3. **constraint_solver.py** - Old name: `constraint_solver_decomp`
4. **constraint_solver_island.py** - Old name: `constraint_solver_island_decomp`
5. **diff_gjk.py** - Old name: `diff_gjk_decomp`
6. **gjk.py** - Old name: `gjk_decomp`
7. **mpr.py** - Old name: `mpr_decomp`
8. **rigid_solver_util.py** - Old name: `rigid_solver_util_decomp`
9. **support_field.py** - Old name: `support_field_decomp`

## How It Works

Each module now includes a backward compatibility shim at the end that:

1. **Creates a module wrapper** - A special `_DeprecatedModuleWrapper` class that inherits from `types.ModuleType`
2. **Registers the old name** - The wrapper is registered in `sys.modules` under the old `*_decomp` name
3. **Shows deprecation warning** - When users access attributes from the old module name, a clear warning is displayed (only once per deprecated module per session)
4. **Proxies to new module** - All attributes from the old name point to the actual new module, so functionality is preserved

**Note:** The warning only appears when you actually **use** something from the deprecated module (e.g., access a class or function), not just when you import it. This prevents spurious warnings during internal module loading.

## User Experience

### Before (would break):
```python
from genesis.engine.solvers.rigid import rigid_solver_decomp
# ImportError: No module named 'rigid_solver_decomp'
```

### After (works with warning):
```python
from genesis.engine.solvers.rigid import rigid_solver_decomp
solver = rigid_solver_decomp.RigidSolver  # Triggers deprecation warning

# Output:
# [Genesis] [WARNING]
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         DEPRECATION WARNING                              ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║ The module 'rigid_solver_decomp' has been renamed to 'rigid_solver'     ║
# ║                                                                          ║
# ║ Please update your imports:                                              ║
# ║   OLD: from genesis.engine.solvers.rigid import rigid_solver_decomp      ║
# ║   NEW: from genesis.engine.solvers.rigid import rigid_solver             ║
# ║                                                                          ║
# ║ This compatibility shim will be removed in a future release.            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
```

## Implementation Details

Each shim includes:

- **Unique identifiers** - Function and class names are suffixed with module-specific identifiers to avoid name collisions
- **Stack inspection** - Attempts to detect imports and show warning immediately (fallback to lazy warning on attribute access)
- **Genesis logger integration** - Uses `gs.logger.warning()` for consistent formatting, falls back to Python's `warnings` module
- **Single warning per session** - Uses `_warned` flag to prevent spamming the same warning multiple times
- **Complete attribute proxying** - All attributes (classes, functions, constants) from the new module are accessible via the old name

## Testing

Test scripts are provided:

1. **test_simple_deprecation.py** - Simple test matching user's scenario
2. **test_final_deprecation.py** - Comprehensive test for rigid_solver
3. **test_all_deprecation_shims.py** - Tests all 9 modules

Run any of these to verify the shims work correctly:
```bash
python test_all_deprecation_shims.py
```

## Future Removal

These shims are designed to be:
- **Temporary** - Should be removed in a future major version
- **Easy to remove** - Just delete the shim section from each file
- **Clear to users** - Warning messages clearly state this is temporary

The warning message explicitly tells users: "This compatibility shim will be removed in a future release."

