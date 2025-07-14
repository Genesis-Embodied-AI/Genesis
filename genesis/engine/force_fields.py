import numpy as np
import taichi as ti

import genesis as gs
from genesis.repr_base import RBC


@ti.data_oriented
class ForceField(RBC):
    """
    Base class for all force fields. This class should not be used directly.

    Note
    ----
    It's called `ForceField`, but it's actually an acceleration field, as force doesn't have a notion of spatial density.
    """

    def __init__(self):
        self._active = ti.field(gs.ti_bool, shape=())
        self._active[None] = False

    def activate(self):
        """
        Activate the force field.
        """
        self._active[None] = True

    def deactivate(self):
        """
        Deactivate the force field.
        """
        self._active[None] = False

    @ti.func
    def get_acc(self, pos, vel, t, i):
        acc = ti.Vector.zero(gs.ti_float, 3)
        if self._active[None]:
            acc = self._get_acc(pos, vel, t, i)
        return acc

    @property
    def active(self):
        """
        Whether the force field is active.
        """
        return self._active[None]


class Constant(ForceField):
    """
    Constant force field with a static acceleration everywhere.

    Parameters:
    -----------
    direction: array_like, shape=(3,)
        The direction of the force (acceleration). Will be normalized.
    strength: float
        The strength of the force (acceleration).
    """

    def __init__(self, direction=(1, 0, 0), strength=1.0):
        super().__init__()

        direction = np.array(direction)
        if direction.shape != (3,):
            raise ValueError("direction must have shape (3,)")

        self._direction = direction / np.linalg.norm(direction)
        self._strength = strength
        self._acc_ti = ti.Vector(self._direction * self._strength, dt=gs.ti_float)

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        return self._acc_ti

    @property
    def direction(self):
        return self._direction

    @property
    def strength(self):
        return self._strength


class Wind(ForceField):
    """
    Wind force field with a static acceleration in a cylindrical region.

    Parameters:
    -----------
    direction: array_like, shape=(3,)
        The direction of the wind. Will be normalized.
    strength: float
        The strength of the wind.
    radius: float
        The radius of the cylinder.
    center: array_like, shape=(3,)
        The center of the cylinder.
    """

    def __init__(self, direction=(1, 0, 0), strength=1.0, radius=1, center=(0, 0, 0)):
        super().__init__()

        direction = np.array(direction)
        if direction.shape != (3,):
            raise ValueError("direction must have shape (3,)")

        center = np.array(center)
        if center.shape != (3,):
            raise ValueError("center must have shape (3,)")

        self._center = center
        self._direction = direction / np.linalg.norm(direction)
        self._strength = strength
        self._radius = radius

        self._direction_ti = ti.Vector(self._direction, dt=gs.ti_float)
        self._center_ti = ti.Vector(self._center, dt=gs.ti_float)
        self._acc_ti = ti.Vector(self._direction * self._strength, dt=gs.ti_float)

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        # distance to the center of the cylinder pointing in the direction of the wind
        dist = (pos - self._center_ti).cross(self._direction_ti).norm()
        acc = self._acc_ti
        if dist > self._radius:
            acc = ti.Vector.zero(gs.ti_float, 3)
        return acc

    @property
    def direction(self):
        return self._direction

    @property
    def strength(self):
        return self._strength

    @property
    def radius(self):
        return self._radius

    @property
    def center(self):
        return self._center


class Point(ForceField):
    """
    Point force field gives a constant force towards (positive strength) or away from (negative strength) the point.

    Parameters:
    -----------
    strength: float
        The strength of the wind.
    position: array_like, shape=(3,)
        The position of the point.
    flow: float
        The flow of the force field.
    falloff_pow: float
        The power of the falloff.
    """

    def __init__(self, strength=1.0, position=(0, 0, 0), falloff_pow=0.0, flow=1.0):
        super().__init__()

        position = np.array(position)
        if position.shape != (3,):
            raise ValueError("position must have shape (3,)")

        self._strength = strength
        self._position = position
        self._falloff_pow = falloff_pow
        self._flow = flow

        self._position_ti = ti.Vector(self._position, dt=gs.ti_float)

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        relative_pos = pos - self._position_ti
        radius = relative_pos.norm(gs.EPS)
        direction = relative_pos / radius
        falloff = 1 / (radius + 1.0) ** self._falloff_pow
        acc = self._strength * direction

        # flow
        acc += (acc - vel) * self._flow

        acc *= falloff

        return acc

    @property
    def strength(self):
        return self._strength

    @property
    def position(self):
        return self._position


class Drag(ForceField):
    """
    Drag force field gives a force opposite to the velocity.

    Parameters:
    -----------
    linear: float
        The linear drag coefficient.
    quadratic: float
        The quadratic drag coefficient.
    """

    def __init__(self, linear=0.0, quadratic=0.0):
        super().__init__()

        self._linear = linear
        self._quadratic = quadratic

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        return -self._linear * vel - self._quadratic * vel.norm() * vel

    @property
    def linear(self):
        return self._linear

    @property
    def quadratic(self):
        return self._quadratic


class Noise(ForceField):
    """
    Noise force field samples random noise at each point.
    """

    def __init__(self, strength=1.0):
        super().__init__()

        self._strength = strength

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        noise = (
            ti.Vector(
                [
                    ti.random(gs.ti_float),
                    ti.random(gs.ti_float),
                    ti.random(gs.ti_float),
                ],
                dt=gs.ti_float,
            )
            * 2
            - 1
        )
        return noise * self._strength

    @property
    def strength(self):
        return self._strength


class Vortex(ForceField):
    """
    Vortex force field revolving around z-axis.

    Parameters:
    -----------
    strength_perpendicular: float
        The strength of the vortex flow in the perpendicular direction. Positive for counterclockwise, negative for clockwise.
    strength_radial: float
        The strength of the vortex flow in the radial direction. Positive for inward, negative for outward.
    center: array_like, shape=(3,)
        The center of the vortex.
    falloff_pow: float
        The power of the falloff.
    falloff_min: float
        The minimum distance (in meters) for the falloff. Under this distance, the force is effective with full strength.
    falloff_max: float
        The maximum distance (in meters) for the falloff. Above this distance, the force is ineffective.
    """

    def __init__(
        self,
        direction=(0.0, 0.0, 1.0),
        center=(0.0, 0.0, 0.0),
        strength_perpendicular=20.0,
        strength_radial=0.0,
        falloff_pow=2.0,
        falloff_min=0.01,
        falloff_max=np.inf,
        damping=0.0,
    ):
        super().__init__()

        direction = np.array(direction)
        if direction.shape != (3,):
            raise ValueError("direction must have shape (3,)")

        center = np.array(center)
        if center.shape != (3,):
            raise ValueError("center must have shape (3,)")

        self._center = center
        self._direction = direction / np.linalg.norm(direction)
        self._damping = damping

        self._strength_perpendicular = strength_perpendicular
        self._strength_radial = strength_radial

        self._falloff_pow = falloff_pow
        self._falloff_min = falloff_min
        self._falloff_max = falloff_max

        self._direction_ti = ti.Vector(self._direction, dt=gs.ti_float)
        self._center_ti = ti.Vector(self._center, dt=gs.ti_float)

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        relative_pos = ti.Vector([pos[0] - self._center_ti[0], pos[1] - self._center_ti[1]])
        radius = relative_pos.norm()
        perpendicular = ti.Vector([-relative_pos[1], relative_pos[0], 0.0], dt=gs.ti_float)
        radial = -ti.Vector([relative_pos[0], relative_pos[1], 0.0], dt=gs.ti_float)

        falloff = gs.ti_float(0.0)
        if radius < self._falloff_min:
            falloff = 1.0
        elif radius < self._falloff_max:
            falloff = 1 / (radius - self._falloff_min + 1.0) ** self._falloff_pow
        else:
            falloff = 0.0

        acceleration = falloff * (self._strength_perpendicular * perpendicular + self._strength_radial * radial)

        acceleration -= self._damping * vel

        return acceleration

    @property
    def direction(self):
        return self._direction

    @property
    def radius(self):
        return self._radius

    @property
    def center(self):
        return self._center

    @property
    def strength_perpendicular(self):
        return self._strength_perpendicular

    @property
    def strength_radial(self):
        return self._strength_radial

    @property
    def falloff_pow(self):
        return self._falloff_pow

    @property
    def falloff_min(self):
        return self._falloff_min

    @property
    def falloff_max(self):
        return self._falloff_max


class Turbulence(ForceField):
    """
    Turbulence force field generated using Perlin noise.

    Parameters:
    -----------
    strength: float
        The strength of the turbulence.
    frequency: float
        The spatial frequency of repeated patterns used for Perlin noise.
    flow: float
        The flow of the turbulence.
    seed: int | None
        The seed for the Perlin noise.
    """

    def __init__(self, strength=1.0, frequency=3, flow=0.0, seed=None):
        super().__init__()

        self._strength = strength
        self._frequency = frequency
        self._flow = flow

        self._perlin_x = PerlinNoiseField(frequency=self._frequency, seed=seed, seed_offset=0)
        self._perlin_y = PerlinNoiseField(frequency=self._frequency, seed=seed, seed_offset=1)
        self._perlin_z = PerlinNoiseField(frequency=self._frequency, seed=seed, seed_offset=2)

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        acc = ti.Vector(
            [
                self._perlin_x._noise(pos[0], pos[1], pos[2]),
                self._perlin_y._noise(pos[0], pos[1], pos[2]),
                self._perlin_z._noise(pos[0], pos[1], pos[2]),
            ],
            dt=gs.ti_float,
        )
        acc *= self._strength

        # flow
        acc += (acc - vel) * self._flow
        return acc

    @property
    def strength(self):
        return self._strength

    @property
    def frequency(self):
        return self._frequency


class Custom(ForceField):
    """
    Custom force field with a user-defined force(acceleration) function `f(pos, vel, t, i)`.

    Parameters:
    -----------
    func: A callable taichi func (a python function wrapped by `@ti.func`)
        The acceleration function. Must have the signature `f(pos: ti.types.vector(3), vel: ti.types.vector(3), t: ti.f32) -> ti.types.vector(3)`.
    """

    def __init__(self, func):
        super().__init__()

        self.get_acc = func


@ti.data_oriented
class PerlinNoiseField:
    """
    Perlin noise field for generating 3D noise.
    Each PerlinNoiseField object has will create a different noise field.
    """

    def __init__(self, wrap_size=256, frequency=10, seed=None, seed_offset=0):
        self._wrap_size = wrap_size
        self._permutation = np.arange(self._wrap_size, dtype=np.int32)
        self._frequency = frequency
        if seed is not None:
            state = np.random.get_state()
            np.random.seed(seed + seed_offset)
            np.random.shuffle(self._permutation)
            np.random.set_state(state)

        self._permutation_ti = ti.field(ti.i32, shape=(self._wrap_size * 2,))
        self._permutation_ti.from_numpy(np.concatenate([self._permutation, self._permutation]))

    @ti.func
    def _fade(self, t):
        """Fade function for smoothing the interpolation."""
        return t * t * t * (t * (t * 6 - 15) + 10)

    @ti.func
    def _lerp(self, t, a, b):
        """Linear interpolation between a and b."""
        return a + t * (b - a)

    @ti.func
    def _grad(self, hash, x, y, z):
        """Calculate dot product between gradient vector and distance vector."""
        h = hash & 15  # Convert low 4 bits of hash code
        u = x
        if h >= 8:
            u = y

        v = y
        if h >= 4:
            v = z
            if h == 12 or h == 14:
                v = x

        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    @ti.func
    def _noise(self, x, y, z):
        x *= self._frequency
        y *= self._frequency
        z *= self._frequency

        # Find unit cube that contains the point
        X = gs.ti_int(ti.floor(x)) & (self._wrap_size - 1)
        Y = gs.ti_int(ti.floor(y)) & (self._wrap_size - 1)
        Z = gs.ti_int(ti.floor(z)) & (self._wrap_size - 1)

        # Find relative x, y, z of point in the cube
        x -= ti.floor(x)
        y -= ti.floor(y)
        z -= ti.floor(z)

        # Compute fade curves for each coordinate
        u = self._fade(x)
        v = self._fade(y)
        w = self._fade(z)

        # Hash coordinates of the 8 cube corners
        A = self._permutation_ti[X] + Y
        AA = self._permutation_ti[A] + Z
        AB = self._permutation_ti[A + 1] + Z
        B = self._permutation_ti[X + 1] + Y
        BA = self._permutation_ti[B] + Z
        BB = self._permutation_ti[B + 1] + Z

        # Add blended results from the 8 corners of the cube
        return self._lerp(
            w,
            self._lerp(
                v,
                self._lerp(
                    u,
                    self._grad(self._permutation_ti[AA], x, y, z),
                    self._grad(self._permutation_ti[BA], x - 1, y, z),
                ),
                self._lerp(
                    u,
                    self._grad(self._permutation_ti[AB], x, y - 1, z),
                    self._grad(self._permutation_ti[BB], x - 1, y - 1, z),
                ),
            ),
            self._lerp(
                v,
                self._lerp(
                    u,
                    self._grad(self._permutation_ti[AA + 1], x, y, z - 1),
                    self._grad(self._permutation_ti[BA + 1], x - 1, y, z - 1),
                ),
                self._lerp(
                    u,
                    self._grad(self._permutation_ti[AB + 1], x, y - 1, z - 1),
                    self._grad(self._permutation_ti[BB + 1], x - 1, y - 1, z - 1),
                ),
            ),
        )
