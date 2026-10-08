"""System of the collision detection of the rigid solver, which opens the second graph of every rigid substep."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine.core import Require, System
from genesis.engine.solvers.rigid.collider.collider import func_detection

from .rigid_solver_system import RigidSolverSystem

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class ColliderSystem(System):
    """System running the collision detection of the rigid solver, which the scene adds when it enables collisions.

    The collider of the rigid solver keeps owning its data, which RigidSolverData exposes to the substep graphs.
    """

    rigid_solver_system = Require(RigidSolverSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.collider = scene.sim.rigid_solver.collider

    @System.action(kind="host")
    def prepare(self):
        return self.clear_contact_data_cache

    @System.action(kind="stage")
    def detect(self):
        collider = self.collider
        return (
            func_substep_detection,
            self.rigid_solver_system.data,
            self.rigid_solver_system.rigid_config,
            collider.collider_config,
            collider.gjk.gjk_config,
            collider._n_possible_pairs > 0,
            collider._use_split_narrowphase,
            collider._use_coop_dedup,
        )

    def build(self):
        self.rigid_solver_system.on_detect(prepare=self.prepare, detect=self.detect)

    def clear_contact_data_cache(self) -> None:
        """Drop the contacts read back during the previous substep, which the detection is about to replace."""
        self.collider._contact_data_cache.clear()


@qd.func(requires_top_level=True)
def func_substep_detection(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    has_possible_pairs: qd.template(),
    split_narrowphase: qd.template(),
    coop_dedup: qd.template(),
):
    """Detect the collisions of a substep, configured by has_possible_pairs, split_narrowphase and coop_dedup (see
    func_detection)."""
    func_detection(
        geoms_init_AABB=rigid_data.geoms_init_AABB,
        dyn_state=rigid_data.dyn_state,
        collider_state=rigid_data.collider_state,
        mpr_state=rigid_data.mpr_state,
        gjk_state=rigid_data.gjk_state,
        diff_contact_input=rigid_data.gjk_state.diff_contact_input,
        contact0_mpr_state=rigid_data.contact0_mpr_state,
        contact0_gjk_state=rigid_data.contact0_gjk_state,
        multicontact_mpr_state=rigid_data.multicontact_mpr_state,
        multicontact_gjk_state=rigid_data.multicontact_gjk_state,
        constraint_state=rigid_data.constraint_state,
        dyn_info=rigid_data.dyn_info,
        rigid_info=rigid_data.rigid_info,
        collider_info=rigid_data.collider_info,
        rigid_config=rigid_config,
        collider_static_config=collider_static_config,
        gjk_static_config=gjk_static_config,
        has_possible_pairs=has_possible_pairs,
        split_narrowphase=split_narrowphase,
        coop_dedup=coop_dedup,
        errno=rigid_data.errno,
    )
