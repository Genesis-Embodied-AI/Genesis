from genesis.ext.pyrender.interaction.vec3 import Vec3

class Ray:
    origin: Vec3
    direction: Vec3

    def __init__(self, origin: Vec3, direction: Vec3):
        self.origin = origin
        self.direction = direction

    def __repr__(self) -> str: return f"Ray(origin={self.origin}, direction={self.direction})"
