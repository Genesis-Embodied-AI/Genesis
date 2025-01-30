# import taichi while suppressing its output
from unittest.mock import patch

_ti_outputs = []


def fake_print(*args, **kwargs):
    output = "".join(args)
    _ti_outputs.append(output)


with patch("builtins.print", fake_print):
    import taichi as ti

import os
import sys
import torch
import atexit
import traceback
import numpy as np

from .constants import GS_ARCH, TI_ARCH
from .constants import backend as gs_backend
from .logging import Logger
from .version import __version__
from .utils import set_random_seed, get_platform, get_device

_initialized = False
backend = None
first_init = True
exit_callbacks = []
global_scene_list = set()


########################## init ##########################
def init(
    seed=None,
    precision="32",
    debug=False,
    eps=1e-12,
    logging_level=None,
    backend=gs_backend.gpu,
    theme="dark",
    logger_verbose_time=False,
):
    # genesis._initialized
    global _initialized
    if _initialized:
        raise_exception("Genesis already initialized.")
    _initialized = True

    # genesis._theme
    if theme not in ["dark", "light", "dumb"]:
        raise_exception(f"Unsupported theme: {theme}")
    global _theme
    _theme = theme

    # verbose repr
    global _verbose
    _verbose = False

    # genesis.logger
    global logger
    global first_init
    if first_init:
        logger = Logger(logging_level, debug, logger_verbose_time)
        atexit.register(_gs_exit)

        # greeting message
        _display_greeting(logger.INFO_length)

        first_init = False

    # genesis.backend
    global platform
    platform = get_platform()
    if backend not in GS_ARCH[platform]:
        raise_exception(f"backend ~~<{backend}>~~ not supported for platform ~~<{platform}>~~")

    # get default device and compute total device memory
    global device
    device, device_name, total_mem, backend = get_device(backend)

    _globalize_backend(backend)

    logger.info(
        f"Running on ~~<[{device_name}]>~~ with backend ~~<{backend}>~~. Device memory: ~~<{total_mem:.2f}>~~ GB."
    )

    # init taichi
    with patch("builtins.print", fake_print):
        # force_scalarize_matrix=True for speeding up kernel compilation
        ti.init(arch=TI_ARCH[platform][backend], debug=debug, force_scalarize_matrix=True)

    for ti_output in _ti_outputs:
        logger.debug(ti_output)

    # dtype
    global ti_float
    global np_float
    global tc_float
    if precision == "32":
        ti_float = ti.f32
        np_float = np.float32
        tc_float = torch.float32
    elif precision == "64":
        ti_float = ti.f64
        np_float = np.float64
        tc_float = torch.float64
    else:
        raise_exception(f"Unsupported precision type: ~~<{precision}>~~")

    # All int uses 32-bit precision, unless under special circumstances.
    global ti_int
    global np_int
    global tc_int
    ti_int = ti.i32
    np_int = np.int32
    tc_int = torch.int32

    # let's use GLSL convention: https://learnwebgl.brown37.net/12_shader_language/glsl_data_types.html
    global ti_vec2
    ti_vec2 = ti.types.vector(2, ti_float)
    global ti_vec3
    ti_vec3 = ti.types.vector(3, ti_float)
    global ti_vec4
    ti_vec4 = ti.types.vector(4, ti_float)
    global ti_vec6
    ti_vec6 = ti.types.vector(6, ti_float)
    global ti_vec7
    ti_vec7 = ti.types.vector(7, ti_float)
    global ti_mat3
    ti_mat3 = ti.types.matrix(3, 3, ti_float)
    global ti_mat4
    ti_mat4 = ti.types.matrix(4, 4, ti_float)
    global ti_ivec2
    ti_ivec2 = ti.types.vector(2, ti_int)
    global ti_ivec3
    ti_ivec3 = ti.types.vector(3, ti_int)
    global ti_ivec4
    ti_ivec4 = ti.types.vector(4, ti_int)

    global EPS
    EPS = eps

    # seed
    if seed is not None:
        global SEED
        SEED = seed
        set_random_seed(SEED)

    global exit_callbacks
    exit_callbacks = []

    logger.info(
        f"🚀 Genesis initialized. 🔖 version: ~~<{__version__}>~~, 🌱 seed: ~~<{seed}>~~, 📏 precision: '~~<{precision}>~~', 🐛 debug: ~~<{debug}>~~, 🎨 theme: '~~<{theme}>~~'."
    )


########################## init ##########################
def destroy():
    """
    A simple wrapper for ti.reset(). This call releases all gpu memories allocated and destroyes all runtime data, and also forces caching of compiled kernels.
    gs.init() needs to be called again to reinitialize the system after destroy.
    """
    # genesis._initialized
    global _initialized
    _initialized = False
    ti.reset()

    global global_scene_list
    for scene in global_scene_list:
        if scene._visualizer is not None:
            if scene._visualizer._rasterizer is not None:
                scene._visualizer._rasterizer.destroy()
    global_scene_list.clear()


def _globalize_backend(_backend):
    global backend
    backend = _backend


def _display_greeting(INFO_length):
    try:
        terminal_size = os.get_terminal_size()[0]
    except OSError as e:
        terminal_size = 80
    wave_width = int((terminal_size - INFO_length - 11) / 2)
    if wave_width % 2 == 0:
        wave_width -= 1
    wave_width = max(0, min(38, wave_width))
    bar_width = wave_width * 2 + 9
    wave = ("┈┉" * wave_width)[:wave_width]
    logger.info(f"~<╭{'─'*(bar_width)}╮>~")
    logger.info(f"~<│{wave}>~ ~~~~<Genesis>~~~~ ~<{wave}│>~")
    logger.info(f"~<╰{'─'*(bar_width)}╯>~")


def set_verbose(verbose):
    global _verbose
    _verbose = verbose


########################## Exception and exit handling ##########################


class GenesisException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _custom_excepthook(exctype, value, tb):
    if issubclass(exctype, GenesisException):
        # We don't want the traceback info to trace till this __init__.py file.
        stack_trace = "".join(traceback.format_exception(exctype, value, tb)[:-2])
        print(stack_trace)
    else:
        # Use the system's default excepthook for other exception types
        sys.__excepthook__(exctype, value, tb)


# Set the custom excepthook to handle GenesisException
sys.excepthook = _custom_excepthook


def _gs_exit():
    # display error if it exists
    if logger._error_msg is not None:
        logger.error(logger._error_msg)

    # This might raise error during unit test
    try:
        logger.info("💤 Exiting Genesis and caching compiled kernels...")
    except:
        pass

    for cb in exit_callbacks:
        cb()

    destroy()


########################## shortcut imports for users ##########################
from .constants import (
    IntEnum,
    JOINT_TYPE,
    GEOM_TYPE,
    EQUALITY_TYPE,
    CTRL_MODE,
    PARA_LEVEL,
    ACTIVE,
    INACTIVE,
    integrator,
    constraint_solver,
)

from .utils.uid import UID
from .utils import tools
from .utils.geom import *
from .utils.misc import assert_built, assert_unbuilt, assert_initialized, raise_exception

from .options import morphs
from .options import renderers
from .options import surfaces
from .options import textures

from .datatypes import List
from .grad.creation_ops import *

from .engine import states, materials, force_fields
from .engine.scene import Scene
from .engine.mesh import Mesh
from .engine.entities.emitter import Emitter

for name, member in gs_backend.__members__.items():
    globals()[name] = member
