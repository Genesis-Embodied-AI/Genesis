import io
import os
import sys
import atexit
import logging as _logging
import traceback
import weakref
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
from .utils import redirect_libc_stderr, set_random_seed, get_platform, get_device
from .utils.misc import ALLOCATE_TENSOR_WARNING


# Global state
_initialized: bool = False
_scene_registry: list[weakref.ReferenceType["Scene"]] = []
_theme: str | None = None
platform: str | None = None
logger: Logger | None = None
device: torch.device | None = None
backend: gs_backend | None = None
use_ndarray: bool | None = None
use_pure: bool | None = None
EPS: float | None = None


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
    performance_mode=False,
):
    global _initialized
    if _initialized:
        raise_exception("Genesis already initialized.")

    # Make sure evertything is properly destroyed, just in case initialization failed previously
    destroy()

    # Update theme if valid
    global _theme
    if theme not in ("dark", "light", "dumb"):
        raise_exception(f"Unsupported theme: ~~<{theme}>~~")
    _theme = theme

    # Dealing with default backend
    if backend is None:
        backend = gs_backend.cpu if debug else gs_backend.gpu

    # Determine the platform
    global platform
    platform = get_platform()

    # Make sure that specified arch and precision are supported
    if precision not in ("32", "64"):
        raise_exception(f"Unsupported precision type: ~~<{precision}>~~")
    if backend not in GS_ARCH[platform]:
        raise_exception(f"Backend ~~<{backend}>~~ not supported for platform ~~<{platform}>~~")

    # Initialize the logger and print greeting message
    global logger
    if logging_level is None:
        logging_level = _logging.DEBUG if debug else _logging.INFO
    logger = Logger(logging_level, logger_verbose_time)

    try:
        columns, _lines = os.get_terminal_size()
    except OSError:
        columns = 80
    wave_width = (columns - logger.INFO_length - 11) // 2
    if wave_width % 2 == 0:
        wave_width -= 1
    wave_width = max(0, min(38, wave_width))
    bar_width = wave_width * 2 + 9
    wave = ("┈┉" * wave_width)[:wave_width]
    logger.info(f"~<╭{'─'*(bar_width)}╮>~")
    logger.info(f"~<│{wave}>~ ~~~~<Genesis>~~~~ ~<{wave}│>~")
    logger.info(f"~<╰{'─'*(bar_width)}╯>~")

    # FIXME: Disable this warning for now, because it is not useful without printing the entire traceback
    logger.addFilter(lambda record: record.msg != ALLOCATE_TENSOR_WARNING)

    # Get concrete device and backend
    global device
    device, device_name, total_mem, backend = get_device(backend)
    if backend != gs.cpu and os.environ.get("GS_TORCH_FORCE_CPU_DEVICE") == "1":
        device, device_name, total_mem, _ = get_device(gs_backend.cpu)

    # It is necessary to disable Metal backend manually because it is not working at taichi-level due to a bug
    if backend == gs_backend.metal and os.environ.get("TI_ENABLE_METAL") == "0":
        backend = gs_backend.cpu

    # Configure GsTaichi fast cache and array type
    global use_ndarray, use_pure
    is_ndarray_disabled = (os.environ.get("GS_ENABLE_NDARRAY") or ("0" if sys.platform == "darwin" else "1")) == "0"
    if use_ndarray is None:
        # _use_ndarray = not (is_ndarray_disabled or performance_mode)
        _use_ndarray = os.environ.get("GS_ENABLE_NDARRAY", "0") == "1"
    else:
        _use_ndarray = use_ndarray
        if _use_ndarray and is_ndarray_disabled:
            raise_exception("Genesis previous initialized. GsTaichi dynamic array type cannot be disabled anymore.")
    if _use_ndarray and backend == gs_backend.metal:
        raise_exception("GsTaichi dynamic array type is not supported on Apple Metal GPU backend.")
    is_pure_disabled = os.environ.get("GS_ENABLE_FASTCACHE", "0") == "0"
    if use_pure is None:
        _use_pure = not is_pure_disabled and _use_ndarray
    else:
        _use_pure = use_pure
        if use_pure and is_pure_disabled:
            raise_exception("Genesis previous initialized. GsTaichi fast cache mode cannot be disabled anymore.")
    use_ndarray, use_pure = _use_ndarray, _use_pure

    # Define the right dtypes in accordance with selected backend and precision
    global ti_float, np_float, tc_float
    if precision == "32":
        ti_float = ti.f32
        np_float = np.float32
        tc_float = torch.float32
    else:  # precision == "64":
        if backend == gs_backend.metal:
            raise_exception("64bits precision is not supported on Apple Metal GPU.")
        ti_float = ti.f64
        np_float = np.float64
        tc_float = torch.float64

    # All int uses 32-bit precision, unless under special circumstances
    global ti_int, np_int, tc_int
    ti_int = ti.i32
    np_int = np.int32
    tc_int = torch.int32

    # Bool
    # Note that `ti.u1` is broken on Apple Metal and output garbage.
    global ti_bool, np_bool, tc_bool
    if backend == gs_backend.metal:
        ti_bool = ti.i32
        np_bool = np.int32
        tc_bool = torch.int32
    else:
        ti_bool = ti.u1
        np_bool = np.bool_
        tc_bool = torch.bool

    # Let's use GLSL convention: https://learnwebgl.brown37.net/12_shader_language/glsl_data_types.html
    global ti_vec2, ti_vec3, ti_vec4, ti_vec6, ti_vec7, ti_vec11, ti_mat3, ti_mat4, ti_ivec2, ti_ivec3, ti_ivec4
    ti_vec2 = ti.types.vector(2, ti_float)
    ti_vec3 = ti.types.vector(3, ti_float)
    ti_vec4 = ti.types.vector(4, ti_float)
    ti_vec6 = ti.types.vector(6, ti_float)
    ti_vec7 = ti.types.vector(7, ti_float)
    ti_vec11 = ti.types.vector(11, ti_float)
    ti_mat3 = ti.types.matrix(3, 3, ti_float)
    ti_mat4 = ti.types.matrix(4, 4, ti_float)
    ti_ivec2 = ti.types.vector(2, ti_int)
    ti_ivec3 = ti.types.vector(3, ti_int)
    ti_ivec4 = ti.types.vector(4, ti_int)

    # Update torch default dtype and device, just in case
    torch.set_default_device(device)
    torch.set_default_dtype(tc_float)

    # Define smallest float that is considered non-zero
    global EPS
    EPS = float(max(eps, np.finfo(np_float).eps))

    # Configure and initialize taichi
    taichi_kwargs = {}
    if gs.logger.level == _logging.CRITICAL:
        taichi_kwargs.update(log_level=ti.ERROR)
    elif gs.logger.level == _logging.ERROR:
        taichi_kwargs.update(log_level=ti.ERROR)
    elif gs.logger.level == _logging.WARNING:
        taichi_kwargs.update(log_level=ti.WARN)
    elif gs.logger.level == _logging.INFO:
        taichi_kwargs.update(log_level=ti.WARN)
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

    if seed is not None:
        global SEED
        SEED = seed
        set_random_seed(SEED)
        taichi_kwargs.update(
            random_seed=seed,
        )

    # init gstaichi
    with redirect_stdout(_ti_outputs):
        ti.init(
            arch=TI_ARCH[platform][backend],
            # Add a (hidden) mechanism to forceable disable taichi debug mode as it is still a bit experimental
            debug=debug and backend == gs.cpu and (os.environ.get("TI_DEBUG") != "0"),
            check_out_of_bound=debug,
            # force_scalarize_matrix=True for speeding up kernel compilation
            # Turning off 'force_scalarize_matrix' is causing numerical instabilities ('nan') on MacOS
            force_scalarize_matrix=True,
            # Turning off 'advanced_optimization' is causing issues on MacOS
            advanced_optimization=True,
            # This improves runtime speed by around 1%-5%, while it makes compilation up to 6x slower
            cfg_optimization=False,
            fast_math=not debug,
            default_ip=ti_int,
            default_fp=ti_float,
            **taichi_kwargs,
        )

    # Make sure that gstaichi arch is matching requirement, then set it in global scope
    ti_config = ti.lang.impl.current_cfg()
    if backend != gs.cpu and ti_config.arch in (ti._lib.core.Arch.arm64, ti._lib.core.Arch.x64):
        device, device_name, total_mem, backend = get_device(gs.cpu)
    globals()["backend"] = backend

    logger.info(
        f"Running on ~~<[{device_name}]>~~ with backend ~~<{backend}>~~. Device memory: ~~<{total_mem:.2f}>~~ GB."
    )

    for ti_output in _ti_outputs.getvalue().splitlines():
        logger.debug(ti_output)
    _ti_outputs.truncate(0)
    _ti_outputs.seek(0)

    # Redirect Taichi logging messages to unify logging management
    for ti_name, gs_name in (
        ("debug", "debug"),
        ("trace", "debug"),
        ("info", "debug"),
        ("warn", "info"),
        ("error", "warning"),
        ("critical", "error"),
    ):
        setattr(ti._logging, ti_name, getattr(logger, gs_name))

    # Dealing with default backend
    if use_pure:
        logger.debug("[GsTaichi] Enabling pure kernels for fast cache mode.")
    if use_ndarray:
        logger.debug("[GsTaichi] Enabling GsTaichi dynamic array type to avoid scene-specific compilation.")
    if backend == gs_backend.metal:
        logger.debug("[GsTaichi] Beware Apple Metal backend may be unstable.")

    msg_options = ", ".join(
        f"{name}: ~~<{val}>~~"
        for name, val in (
            ("🔖 version", __version__),
            ("🎨 theme", theme),
            ("🌱 seed", seed),
            ("🐛 debug", debug),
            ("📏 precision", precision),
            ("🏎️ performance", performance_mode),
            ("ℹ️ verbose", _logging.getLevelName(gs.logger.level)),
        )
    )
    logger.info(f"🚀 Genesis initialized. {msg_options}")

    atexit.register(destroy)
    _initialized = True


########################## destroy ##########################


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
    # This is important when `init` / `destroy` is called multiple times, which is typically the case for unit tests.
    atexit.unregister(destroy)

    # Display any buffered error message if logger is configured
    global logger
    if logger:
        logger.info("💤 Exiting Genesis and caching compiled kernels...")

    # Destroy all scenes
    global _scene_registry
    for scene_ref in _scene_registry.copy():
        if scene_ref:
            scene = scene_ref()
            scene.destroy()

    # Reset gstaichi
    ti.reset()

    # Restore original taichi logging facilities
    for ti_name, ti_level in (
        ("debug", ti._logging.DEBUG),
        ("trace", ti._logging.TRACE),
        ("info", ti._logging.INFO),
        ("warn", ti._logging.WARN),
        ("error", ti._logging.ERROR),
        ("critical", ti._logging.CRITICAL),
    ):
        setattr(ti._logging, ti_name, ti._logging._get_logging(ti_level))

    # Delete logger
    logger.removeHandler(logger.handler)
    logger = None

    # Clear global state
    global _theme, device, backend, EPS
    _theme = None
    device = None
    backend = None
    EPS = None


########################## Exception and exit handling ##########################


class GenesisException(Exception):
    pass


def _custom_excepthook(exctype, value, tb):
    print("".join(traceback.format_exception(exctype, value, tb)))

    # Log the exception right before exit if possible
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
    from pygel3d import graph, hmesh

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
from .options import sensors
from .options import renderers
from .options import surfaces
from .options import textures

from .datatypes import List
from .grad.creation_ops import *

from .engine import states, materials, force_fields
from .engine.mesh import Mesh
from .engine.scene import Scene

from . import recorders

for name, member in gs_backend.__members__.items():
    globals()[name] = member
