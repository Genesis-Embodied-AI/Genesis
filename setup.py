"""Setup for fast-simplification."""

import builtins
from io import open as io_open
import os
import sys
import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


# Define macros for cython
macros = []
if os.name == "nt":  # windows
    extra_compile_args = ["/openmp", "/O2", "/w", "/GS"]
elif os.name == "posix":  # linux org mac os
    if sys.platform == "linux":
        extra_compile_args = ["-std=gnu++11", "-O3", "-w"]
    else:  # probably mac os
        extra_compile_args = ["-std=c++11", "-O3", "-w"]
else:
    raise OSError("Unsupported OS %s" % os.name)

# Check if 64-bit
if sys.maxsize > 2**32:
    macros.append(("IS64BITPLATFORM", None))


# for: the cc1plus: warning: command line option '-Wstrict-prototypes'
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # prevent numpy from thinking it is still in its setup process:
        try:
            del builtins.__NUMPY_SETUP__
        except AttributeError:
            pass
        import numpy

        self.include_dirs.append(numpy.get_include())

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        _build_ext.build_extensions(self)


setup(
    # Build cython modules
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            "genesis.ext.fast_simplification._simplify",
            ["genesis/ext/fast_simplification/_simplify.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=extra_compile_args,
            define_macros=macros,
        ),
        Extension(
            "genesis.ext.fast_simplification._replay",
            ["genesis/ext/fast_simplification/_replay.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=extra_compile_args,
            define_macros=macros,
        ),
    ],
)
