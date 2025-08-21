import io
import os
import sys
import site
import atexit
import logging as _logging
import traceback
from platform import system
from contextlib import redirect_stdout

# Import gstaichi while collecting its output without printing directly
_ti_outputs = io.StringIO()

os.environ.setdefault("TI_ENABLE_PYBUF", "0" if sys.stdout is sys.__stdout__ else "1")

with redirect_stdout(_ti_outputs):
    import gstaichi as ti

try:
    import torch
except ImportError as e:
    raise ImportError(
        "'torch' module not available. Please install pytorch manually: https://pytorch.org/get-started/locally/"
    ) from e
import numpy as np

from .constants import GS_ARCH, TI_ARCH
from .constants import backend as gs_backend
from .logging import Logger
from .version import __version__
from .utils import redirect_libc_stderr, set_random_seed, get_platform, get_device, get_cache_dir
from .utils.misc import ALLOCATE_TENSOR_WARNING


os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(get_cache_dir(), "numba"))


_initialized = False
backend = None
exit_callbacks = []
global_scene_list = set()


########################## init ##########################
def init(
    seed=None,
    precision="32",
    debug=False,
    eps=1e-15,
    logging_level=None,
    backend=None,
    theme="dark",
    logger_verbose_time=False,
    performance_mode: bool = False,  # True: compilation up to 6x slower (GJK), but runs ~1-5% faster
):
    global _initialized
    if _initialized:
        raise_exception("Genesis already initialized.")

    # Make sure evertything is properly destroyed, just in case initialization failed previously
    destroy()

    # genesis._theme
    global _theme
    is_theme_valid = theme in ("dark", "light", "dumb")
    # Set fallback theme if necessary to be able to initialize logger
    _theme = theme if is_theme_valid else "dark"

    # genesis.logger
    global logger
    if logging_level is None:
        logging_level = _logging.DEBUG if debug else _logging.INFO
    logger = Logger(logging_level, logger_verbose_time)
    atexit.register(destroy)

    # FIXME: Disable this warning for now, because it is not useful without printing the entire traceback
    gs.logger.addFilter(lambda record: record.msg != ALLOCATE_TENSOR_WARNING)

    # Must delay raising exception after logger initialization
    if not is_theme_valid:
        raise_exception(f"Unsupported theme: {theme}")

    # Dealing with default backend
    global platform
    platform = get_platform()
    if backend is None:
        if debug:
            backend = gs_backend.cpu
        else:
            backend = gs_backend.gpu

    # verbose repr
    global _verbose
    _verbose = False

    # greeting message
    _display_greeting(logger.INFO_length)

    # genesis.backend
    if backend not in GS_ARCH[platform]:
        raise_exception(f"backend ~~<{backend}>~~ not supported for platform ~~<{platform}>~~")
    if backend == gs_backend.metal:
        logger.info("Beware Apple Metal backend may be unstable.")

    # get default device and compute total device memory
    global device
    device, device_name, total_mem, backend = get_device(backend)

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

    # Bool
    # Note that `ti.u1` is broken on Apple Metal and output garbage.
    global ti_bool
    global np_bool
    global tc_bool
    if backend == gs_backend.metal:
        ti_bool = ti.i32
        np_bool = np.int32
        tc_bool = torch.int32
    else:
        ti_bool = ti.u1
        np_bool = np.bool_
        tc_bool = torch.bool

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
    global ti_vec11
    ti_vec11 = ti.types.vector(11, ti_float)
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
    EPS = max(eps, np.finfo(np_float).eps)

    taichi_kwargs = {}
    if gs.logger.level == _logging.CRITICAL:
        taichi_kwargs.update(log_level=ti.CRITICAL)
    elif gs.logger.level == _logging.ERROR:
        taichi_kwargs.update(log_level=ti.ERROR)
    elif gs.logger.level == _logging.WARNING:
        taichi_kwargs.update(log_level=ti.WARN)
    elif gs.logger.level == _logging.INFO:
        taichi_kwargs.update(log_level=ti.INFO)
    elif gs.logger.level == _logging.DEBUG:
        taichi_kwargs.update(log_level=ti.INFO)
    if debug:
        if backend == gs_backend.cpu:
            taichi_kwargs.update(cpu_max_num_threads=1)
        else:
            logger.warning("Debug mode is partially supported for GPU backend.")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Beware running Genesis in debug mode dramatically reduces runtime speed.")

    if not performance_mode:
        logger.info(
            "Consider setting 'performance_mode=True' in production to maximise runtime speed, if significantly "
            "increasing compilation time is not a concern."
        )

    if seed is not None:
        global SEED
        SEED = seed
        set_random_seed(SEED)
        taichi_kwargs.update(
            random_seed=seed,
        )

    # It is necessary to disable Metal backend manually because it is not working at taichi-level due to a bug
    ti_arch = TI_ARCH[platform][backend]
    if (backend == gs_backend.metal) and (os.environ.get("TI_ENABLE_METAL") == "0"):
        ti_arch = TI_ARCH[platform][gs_backend.cpu]

    # init gstaichi
    with redirect_stdout(_ti_outputs):
        ti.init(
            arch=ti_arch,
            # Add a (hidden) mechanism to forceable disable taichi debug mode as it is still a bit experimental
            debug=debug and backend == gs.cpu and (os.environ.get("TI_DEBUG") != "0"),
            check_out_of_bound=debug,
            # force_scalarize_matrix=True for speeding up kernel compilation
            # Turning off 'force_scalarize_matrix' is causing numerical instabilities ('nan') on MacOS
            force_scalarize_matrix=True,
            # Turning off 'advanced_optimization' is causing issues on MacOS
            advanced_optimization=True,
            cfg_optimization=performance_mode,
            fast_math=not debug,
            default_ip=ti_int,
            default_fp=ti_float,
            **taichi_kwargs,
        )

    # Make sure that gstaichi arch is matching requirement
    ti_runtime = ti.lang.impl.get_runtime()
    ti_arch = ti_runtime.prog.config().arch
    if backend != gs.cpu and ti_arch in (ti._lib.core.Arch.arm64, ti._lib.core.Arch.x64):
        device, device_name, total_mem, backend = get_device(gs.cpu)

    _globalize_backend(backend)

    # Update torch default device
    torch.set_default_device(device)
    torch.set_default_dtype(tc_float)

    logger.info(
        f"Running on ~~<[{device_name}]>~~ with backend ~~<{backend}>~~. Device memory: ~~<{total_mem:.2f}>~~ GB."
    )

    for ti_output in _ti_outputs.getvalue().splitlines():
        logger.debug(ti_output)
    _ti_outputs.truncate(0)
    _ti_outputs.seek(0)

    global exit_callbacks
    exit_callbacks = []

    logger.info(
        f"🚀 Genesis initialized. 🔖 version: ~~<{__version__}>~~, 🌱 seed: ~~<{seed}>~~, 📏 precision: '~~<{precision}>~~', 🐛 debug: ~~<{debug}>~~, 🎨 theme: '~~<{theme}>~~'."
    )

    _initialized = True


########################## init ##########################
def destroy():
    """
    A simple wrapper for ti.reset(). This call releases all gpu memories allocated and destroyes all runtime data, and also forces caching of compiled kernels.
    gs.init() needs to be called again to reinitialize the system after destroy.
    """
    # Early return if not initialized
    global _initialized
    if not _initialized:
        return

    # Do not consider Genesis as initialized at this point
    _initialized = False

    # Unregister at-exit callback that is not longer relevant.
    # This is important when `init` / `destory` is called multiple times, which is typically the case for unit tests.
    atexit.unregister(destroy)

    # Display any buffered error message if logger is configured
    global logger
    if logger:
        logger.info("💤 Exiting Genesis and caching compiled kernels...")

    # Call all exit callbacks
    for cb in exit_callbacks:
        cb()
    exit_callbacks.clear()

    # Destroy all scenes
    global global_scene_list
    for scene in global_scene_list:
        if scene._visualizer is not None:
            scene._visualizer.destroy()
        del scene
    global_scene_list.clear()

    # Reset gstaichi
    ti.reset()

    # Delete logger
    logger.removeHandler(logger.handler)
    logger = None


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
    global logger
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
    print("".join(traceback.format_exception(exctype, value, tb)))

    # Logger the exception right before exit if possible
    global logger
    try:
        logger.error(f"{exctype.__name__}: {value}")
    except (AttributeError, NameError):
        # Logger may not be configured at this point
        pass


# Set the custom excepthook to handle GenesisException
sys.excepthook = _custom_excepthook

########################## shortcut imports for users ##########################

from .ext import _trimesh_patch
from .utils.misc import get_src_dir as _get_src_dir

with open(os.devnull, "w") as stderr, redirect_libc_stderr(stderr):
    try:
        sys.path.append(os.path.join(_get_src_dir(), "ext/LuisaRender/build/bin"))
        import LuisaRenderPy as _LuisaRenderPy
    except ImportError:
        pass

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
from .utils.misc import assert_built, assert_unbuilt, assert_initialized, raise_exception, raise_exception_from

from .options import morphs
from .options import renderers
from .options import surfaces
from .options import textures

from .datatypes import List
from .grad.creation_ops import *

with open(os.devnull, "w") as stderr, redirect_libc_stderr(stderr):
    from .engine import states, materials, force_fields
    from .engine.scene import Scene
    from .engine.mesh import Mesh
    from .engine.entities.emitter import Emitter

for name, member in gs_backend.__members__.items():
    globals()[name] = member
