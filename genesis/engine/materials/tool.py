from typing import TYPE_CHECKING

import quadrants as qd

from .base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.tool_entity import ToolEntity


@qd.data_oriented
class Tool(Material["ToolEntity"]):
    def __init__(
        self,
        friction=0.0,
        coup_softness=0.01,
        collision=True,
        sdf_res=128,
    ):
        super().__init__()

        self.friction = friction
        self.coup_softness = coup_softness
        self.collision = collision
        self.sdf_res = sdf_res
