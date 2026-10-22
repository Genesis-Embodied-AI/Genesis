"""System of the signed distance fields of the rigid geoms."""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.core import Data, Require, System, register_system
from genesis.engine.entities import RigidEntity
from genesis.utils.sdf import SDF

from .rigid_dynamics import RigidDynamics
from ..simulator_engine import SimulatorEngine


class RigidSdf(System):
    """System owning the signed distance fields of the rigid geoms, a geometric description of each geom that does not
    depend on what queries it.

    The rigid-rigid contact queries them for non-convex geoms, through the collider info that embeds them. The couplings
    of particles with rigid geoms, the CPIC of the MPM solver and the tactile sensors query them too. SDF values are
    precomputed once per geometry and the System allocates their device representation.
    """

    # -------------------------------- Data --------------------------------

    @qd.data_oriented
    class Data(Data):
        sdf_config: array_class.SDFStaticConfig
        sdf_info: array_class.SDFInfo

    data: Data

    # -------------------------------- Dependencies --------------------------------

    rigid_dynamics = Require(RigidDynamics)

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()

    # -------------------------------- Actions --------------------------------

    @System.action(kind="host")
    def initialize(self):
        return self.allocate

    # -------------------------------- Implementation --------------------------------

    def allocate(self) -> None:
        """Allocate the fields from the model of the rigid geoms, ahead of the collision detection that embeds
        them."""
        rigid = self.rigid_dynamics
        geoms = tuple(rigid.geoms)
        self.data.sdf_config = array_class.SDFStaticConfig(
            has_terrain=any(geom.type == gs.GEOM_TYPE.TERRAIN for geom in geoms)
        )
        self.data.sdf_info = SDF(geoms, rigid._para_level).sdf_info


@register_system(SimulatorEngine)
def create_rigid_sdf(engine: SimulatorEngine) -> RigidSdf | None:
    """Admit SDF ownership only for a Scene with Rigid geometry."""
    return RigidSdf() if any(isinstance(entity, RigidEntity) for entity in engine.scene.entities) else None
