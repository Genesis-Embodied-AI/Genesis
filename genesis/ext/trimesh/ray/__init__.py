from . import ray_triangle

# optionally load an interface to the embree raytracer
try:
    from . import ray_pyembree

    has_embree = True
except BaseException as E:
    from .. import exceptions

    ray_pyembree = exceptions.ExceptionWrapper(E)
    has_embree = False

# add to __all__ as per pep8
__all__ = ["ray_triangle", "ray_pyembree"]
