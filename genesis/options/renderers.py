from typing import Optional

import genesis as gs
import numpy as np
from .options import Options
from .surfaces import Surface


class Renderer(Options):
    """
    This is the base class for all `gs.renderers.*` classes.
    Note that this is not an actual renderer, but rather a renderer configuration specifying which renderer to use and its parameters.
    """

    pass


class Rasterizer(Renderer):
    """
    Rasterizer renderer. This has no parameter to be configured.

    Note
    ----
    You can set which renderer to use for cameras, but the interactive viewer always uses the rasterizer rendering backend. If you want to configure properties like shadow, lights, etc., you should use `gs.options.VisOptions` instead.
    """

    pass


class RayTracer(Renderer):
    """
    RayTracer renderer.

    Note
    ----
    We use a environmental sphere wrapped around the scene to render the environment map (i.e. skybox).

    Parameters
    ----------
    cuda_device : int, optional
        CUDA device ID. Defaults to 0.
    logging_level : str, optional
        Logging level. Should be one of "debug", "info", "warning". Defaults to "warning".
    state_limit : int, optional
        State limit for raytracer integrator. Defaults to 2 ** 25.
    tracing_depth : int, optional
        Tracing depth. Defaults to 32.
    rr_depth : int, optional
        Russian Roulette depth. Defaults to 0.
    rr_threshold : float, optional
        Russian Roulette threshold. Defaults to 0.95.
    env_surface : Optional[Surface], optional
        Environment surface. Defaults to None.
    env_radius : float, optional
        Environment radius. Defaults to 1000.0.
    env_pos : tuple, optional
        Environment position. Defaults to (0.0, 0.0, 0.0).
    env_euler : tuple, optional
        Environment Euler angles. Defaults to (0.0, 0.0, 0.0).
    env_quat : Optional[tuple], optional
        Environment quaternion. Defaults to None.
    lights : list of dict, optional
        List of lights. Each light is a dictionary with keys 'pos', 'color', 'intensity', 'radius'. Defaults to [{'pos' : (0.0, 0.0, 10.0), 'color' : (1.0, 1.0, 1.0), 'intensity' : 20.0, 'radius' : 4.0}].
    normal_diff_clamp : float, optional
        Lower bound for direct face normal vs vertex normal for face normal interpolation. Range is [0, 180]. Defaults to 180.
    """

    cuda_device: int = 0
    logging_level: str = "warning"
    state_limit: int = 2**25
    tracing_depth: int = 32
    rr_depth: int = 0
    rr_threshold: float = 0.95

    # environment texure
    env_surface: Optional[Surface] = None
    env_radius: float = 1000.0
    env_pos: tuple = (0.0, 0.0, 0.0)
    env_euler: tuple = (0.0, 0.0, 0.0)
    env_quat: Optional[tuple] = None

    # sphere lights
    lights: list = [{"pos": (0.0, 0.0, 10.0), "color": (1.0, 1.0, 1.0), "intensity": 10.0, "radius": 4.0}]

    # lower bound for direct face normal vs vertex normal for face normal interpolation
    normal_diff_clamp: float = 180  # [0, 180]

    def __init__(self, **data):
        super().__init__(**data)

        if self.logging_level not in ["debug", "info", "warning"]:
            gs.raise_exception("Invalid logging level.")

        if self.env_euler is not None:
            if self.env_quat is None:
                self.env_quat = gs.utils.geom.xyz_to_quat(np.array(self.env_euler))
            else:
                gs.logger.warning("`env_euler` is ignored when `env_quat` is specified.")
