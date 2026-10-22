"""Participant System of the rigid joint limits, the position bounds of the revolute and prismatic joints."""

from collections.abc import Iterator

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.core import Data, Require, System, register_system
from genesis.engine.solvers.rigid.constraint.backward import kernel_manual_add_joint_limit_constraints_bw
from genesis.engine.solvers.rigid.constraint.solver import func_substep_add_joint_limit_rows

from .rigid_constraint import RigidConstraint, RowCategory
from .rigid_dynamics import RigidDynamics
from ..simulator_engine import SimulatorEngine


class RigidJointLimitConstraint(System):
    """Append one unilateral row per revolute or prismatic joint at one of its limits in every env."""

    # -------------------------------- Data --------------------------------

    @qd.data_oriented
    class Data(Data):
        """Where the joint-limit rows start in every env.

        ``@qd.data_oriented`` is a QD discovery workaround: ``GlobalDataContext`` can expose this root only in a form
        QD traverses. Remove it when QD derives ndarray parameters through ordinary non-data-oriented objects.
        """

        segment: array_class.ConstraintRowSegment

    data: Data

    # -------------------------------- Dependencies --------------------------------

    rigid_dynamics = Require(RigidDynamics)

    rigid_constraint = Require(RigidConstraint)

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()

    def build(self) -> None:
        self.rigid_constraint.on_rows(
            RowCategory.LIMIT,
            initialize=self.initialize,
            assemble=self.assemble,
            reverse=self.reverse,
        )

    # -------------------------------- Actions --------------------------------

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="stage")
    def assemble(self):
        rigid = self.rigid_dynamics
        return (
            func_substep_add_joint_limit_rows,
            self.data,
            self.rigid_constraint.data,
            rigid.data,
            rigid.rigid_config,
        )

    @System.action(kind="host")
    def reverse(self):
        rigid = self.rigid_dynamics
        return (
            kernel_manual_add_joint_limit_constraints_bw,
            self.data.segment,
            rigid.dyn_state,
            self.rigid_constraint.data.constraint_state,
            rigid.dyn_info,
            rigid.rigid_info,
            rigid.rigid_config,
        )

    # -------------------------------- Public get/set API --------------------------------

    def iter_data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        for item in array_class.iter_data(self.data, "segment"):
            yield item._replace(name=f"limit_rows.{item.name}")

    # -------------------------------- Implementation --------------------------------

    def initialize_data(self) -> None:
        """Allocate the segment record and reserve one row per revolute or prismatic joint, which is at one limit at a
        time."""
        rigid = self.rigid_dynamics
        self.data.segment = array_class.get_constraint_row_segment(batch_size=rigid._B)
        n_limited_joints = sum(
            joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC) for joint in rigid.joints
        )
        self.rigid_constraint.reserve_rows(RowCategory.LIMIT, n_limited_joints)


@register_system(SimulatorEngine)
def create_rigid_joint_limit_constraint(engine: SimulatorEngine) -> RigidJointLimitConstraint | None:
    """Create the joint-limit rows when a Rigid scene solves constraints and enforces the joint limits."""
    from genesis.engine.entities import RigidEntity

    rigid_options = engine.scene.options.rigid
    has_rigid = any(isinstance(entity, RigidEntity) for entity in engine.scene.entities)
    return (
        RigidJointLimitConstraint()
        if has_rigid and not rigid_options.disable_constraint and rigid_options.enable_joint_limit
        else None
    )
