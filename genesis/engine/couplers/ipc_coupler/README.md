# IPC Coupler Notes

This folder contains the IPC coupler implementation used by Genesis rigid/FEM coupling.

## Internal debug surface export

`IPCCoupler` supports an internal debug mode that exports IPC scene surface snapshots
after every `retrieve()` call in the coupling loop.

### Options

Set these private options in `gs.options.IPCCouplerOptions`:

- `_export_ipc_surface` (`bool`, default `False`)
  - Export IPC scene surface right after each `self._ipc_world.retrieve()`.
- `_export_pre_coupling_surface` (`bool`, default `False`)
  - Export Genesis rigid surface after Genesis first solve and before IPC correction.
- `_export_post_coupling_surface` (`bool`, default `False`)
  - Export Genesis rigid surface after IPC correction writeback + explicit FK.
- `_export_surface_dir` (`str | None`, default `None`)
  - Output directory for exported files (`None` uses IPC workspace).

Set this public advanced option in `gs.options.IPCCouplerOptions` if needed:

- `ignore_end_effector_check` (`bool`, default `False`)
  - Bypasses articulated two-way coupling checks (explicit `coup_links` and end-effector-only).
  - Intended for advanced debugging/experiments; may cause non-physical behavior.

### Output format and naming

The coupler exports one surface snapshot per retrieve step using a zero-padded index:

- `ipc_surface_000000.obj`
- `ipc_surface_000001.obj`
- `...`

If export fails, the coupler logs a warning and continues simulation.

### Example

```python
scene = gs.Scene(
    coupler_options=gs.options.IPCCouplerOptions(
        ignore_end_effector_check=False,
        _export_ipc_surface=True,
        _export_pre_coupling_surface=True,
        _export_post_coupling_surface=True,
        _export_surface_dir="C:/tmp/ipc_surface_debug",
    ),
)
```

Genesis export files are named like:

- `genesis_surface_after_genesis_before_ipc_000000.obj`
- `genesis_surface_after_ipc_correction_000000.obj`
