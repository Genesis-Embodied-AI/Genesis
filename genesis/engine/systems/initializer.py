"""Manager System that initializes the data of the Systems once the solvers have built their entities."""

from typing import TYPE_CHECKING

from genesis.engine.core import Require, System, register_system

from .constraint_solver_system import ConstraintSolverSystem
from .rigid_rigid_contact_system import RigidRigidContactSystem
from .rigid_sdf_system import RigidSdfSystem
from .rigid_solver_system import RigidSolverSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class Initializer(System):
    """Initialize the Rigid owner Systems in dependency order."""

    rigid_sdf = Require(RigidSdfSystem)
    rigid = Require(RigidSolverSystem)
    rigid_rigid_contact = Require(RigidRigidContactSystem)
    constraint = Require(ConstraintSolverSystem)

    def initialize(self) -> None:
        """Initialize the data of every System."""
        # The signed distance fields of the rigid geoms are sized from the rigid model, and the collision detection
        # that the rigid data initialization constructs embeds them, so they are allocated in between
        self.rigid.initialize_model.invoke()
        self.rigid_sdf.initialize.invoke()
        self.rigid.initialize.invoke()
        self.rigid_rigid_contact.initialize.invoke()
        self.constraint.initialize.invoke()
        self.rigid.finalize.invoke()


@register_system(SimulatorEngine)
def create_initializer(scene: "Scene") -> Initializer:
    """Create the manager of the initialization, which every scene has."""
    return Initializer(scene)
