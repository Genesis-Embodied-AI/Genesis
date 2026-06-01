import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array


@pytest.mark.required
def test_solver_state_change_subscribers(show_viewer):
    # Imported here rather than at module scope: pulling in the solver package at collection time would import its
    # quadrants kernels before genesis is initialized (gs.qd_float undefined until then).
    from genesis.engine.solvers.base_solver import StateChange, Subscriber

    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.5),
        ),
    )
    scene.build(n_envs=2)

    solver = scene.sim.rigid_solver

    # Eager mode: a callback fires immediately on each matching change and nothing is retained.
    eager_events = []
    eager = Subscriber(
        to=frozenset({StateChange.GEOMETRY}),
        callback=lambda change, envs_idx: eager_events.append((change, envs_idx)),
    )
    solver.subscribe(eager)
    # Lazy mode: matching changes accumulate on the Subscriber handle until cleared.
    lazy = Subscriber(to=frozenset({StateChange.GEOMETRY}))
    solver.subscribe(lazy)
    # A DYNAMICS-only subscriber must stay silent on GEOMETRY changes (filter).
    dynamics = Subscriber(to=frozenset({StateChange.DYNAMICS}))
    solver.subscribe(dynamics)

    cube.set_pos(torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=gs.tc_float, device=gs.device))
    # Eager fired once with the right category; envs_idx forwarded verbatim (None == every env).
    assert len(eager_events) == 1
    assert eager_events[0][0] is StateChange.GEOMETRY
    assert eager_events[0][1] is None
    # Lazy accumulated the category; the DYNAMICS subscriber saw nothing; eager retains nothing.
    assert lazy.pending == frozenset({StateChange.GEOMETRY})
    assert dynamics.pending == frozenset()
    assert eager.pending == frozenset()

    # A targeted setter forwards the exact env subset to the eager callback.
    cube.set_pos(torch.tensor([[0.0, 0.0, 3.0]], dtype=gs.tc_float, device=gs.device), envs_idx=[1])
    assert len(eager_events) == 2
    forwarded = eager_events[1][1]
    assert forwarded is not None
    assert int(np.atleast_1d(tensor_to_array(forwarded))[0]) == 1

    # Lazy state is idempotent across repeated changes and resets on clear().
    assert lazy.pending == frozenset({StateChange.GEOMETRY})
    lazy.clear()
    assert lazy.pending == frozenset()

    # Reads never notify.
    solver.get_links_pos()
    solver.get_links_quat()
    assert len(eager_events) == 2
    assert lazy.pending == frozenset()

    # Physics integration mutates state through kernels, not a tagged method, so it never notifies.
    scene.step()
    assert len(eager_events) == 2
    assert lazy.pending == frozenset()
