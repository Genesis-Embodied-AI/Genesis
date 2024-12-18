import copy

import numpy as np
import taichi as ti

import genesis as gs
from genesis.utils.misc import *
from genesis.utils.repr import brief


@ti.data_oriented
class CubeBoundary:
    def __init__(self, lower, upper, restitution=0.0):
        self.restitution = restitution

        self.upper = np.array(upper, dtype=gs.np_float)
        self.lower = np.array(lower, dtype=gs.np_float)
        assert (self.upper >= self.lower).all()

        self.upper_ti = ti.Vector(upper, dt=gs.ti_float)
        self.lower_ti = ti.Vector(lower, dt=gs.ti_float)

    @ti.func
    def impose_pos_vel(self, pos, vel):
        for i in ti.static(range(3)):
            if pos[i] >= self.upper_ti[i] and vel[i] >= 0:
                vel[i] *= -self.restitution
            elif pos[i] <= self.lower_ti[i] and vel[i] <= 0:
                vel[i] *= -self.restitution

        pos = ti.max(ti.min(pos, self.upper_ti), self.lower_ti)

        return pos, vel

    @ti.func
    def impose_pos(self, pos):
        pos = ti.max(ti.min(pos, self.upper_ti), self.lower_ti)
        return pos

    def is_inside(self, pos):
        return np.all(pos < self.upper) and np.all(pos > self.lower)

    def __repr__(self):
        return (
            f"{brief(self)}\n"
            f"lower       : {brief(self.lower)}\n"
            f"upper       : {brief(self.upper)}\n"
            f"restitution : {brief(self.restitution)}"
        )


@ti.data_oriented
class FloorBoundary:
    def __init__(self, height, restitution=0.0):
        self.height = height
        self.restitution = restitution

    @ti.func
    def impose_pos_vel(self, pos, vel):
        if pos[2] <= self.height and vel[2] <= 0:
            vel[2] *= -self.restitution

        pos[2] = ti.max(pos[2], self.height)

        return pos, vel

    @ti.func
    def impose_pos(self, pos):
        pos[2] = ti.max(pos[2], self.height)
        return pos

    def __repr__(self):
        return f"{brief(self)}\n" f"height      : {brief(self.height)}\n" f"restitution : {brief(self.restitution)}"
