import enum

import taichi as ti

# dynamic loading
ACTIVE = 1
INACTIVE = 0

# misc
EPS = None
SEED = None

# logging
logger = None
error_msg = None


class IntEnum(enum.IntEnum):
    def __repr__(self):
        return f"<gs.{self.__class__.__name__}.{self.name}: {self.value}>"

    def __format__(self, format_spec):
        return f"<{self.name}: {self.value}>"


# geom type in rigid solver
class GEOM_TYPE(IntEnum):
    PLANE = 0
    SPHERE = 1
    CYLINDER = 2
    CAPSULE = 3
    BOX = 4
    MESH = 5
    TERRAIN = 6


# joint type in rigid solver, ranked by number of dofs
class JOINT_TYPE(IntEnum):
    FIXED = 0
    REVOLUTE = 1
    PRISMATIC = 2
    SPHERICAL = 3
    FREE = 4
    PLANAR = 5


class EQUALITY_TYPE(IntEnum):
    CONNECT = 0
    WELD = 1
    JOINT = 2


# joint type in rigid solver, ranked by number of dofs
class CTRL_MODE(IntEnum):
    FORCE = 0
    VELOCITY = 1
    POSITION = 2


######### User accessible constants do not capitalize #########
# rigid solver intergrator
class integrator(IntEnum):
    Euler = 0
    implicitfast = 1
    approximate_implicitfast = 2


# rigid solver constraint solver
class constraint_solver(IntEnum):
    CG = 0
    Newton = 1


# backend
class backend(IntEnum):
    cpu = 0
    gpu = 1
    cuda = 2
    vulkan = 3
    metal = 4
    opengl = 5

    def __format__(self, format_spec):
        return f"gs.{self.name}"


GS_ARCH = {
    "macOS": {
        backend.cpu: backend.cpu,
        backend.gpu: backend.metal,
        backend.metal: backend.metal,
        backend.vulkan: backend.vulkan,
    },
    "Linux": {
        backend.cpu: backend.cpu,
        backend.gpu: backend.cuda,
        backend.cuda: backend.cuda,
        backend.vulkan: backend.vulkan,
    },
    "Windows": {
        backend.cpu: backend.cpu,
        backend.gpu: backend.cuda,
        backend.cuda: backend.cuda,
        backend.vulkan: backend.vulkan,
    },
}

TI_ARCH = {
    "macOS": {
        backend.cpu: ti.cpu,
        backend.metal: ti.metal,
        backend.vulkan: ti.vulkan,
    },
    "Linux": {
        backend.cpu: ti.cpu,
        backend.cuda: ti.cuda,
        backend.vulkan: ti.vulkan,
    },
    "Windows": {
        backend.cpu: ti.cpu,
        backend.cuda: ti.cuda,
        backend.vulkan: ti.vulkan,
    },
}


# parallelize
class PARA_LEVEL(IntEnum):
    NEVER = 0  # when using cpu
    PARTIAL = 1  # when using gpu for non-batched scene
    ALL = 2  # when using gpu for batched scene
