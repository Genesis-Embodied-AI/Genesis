"""Setup for fast-simplification."""

import builtins
from io import open as io_open
import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

filepath = os.path.dirname(__file__)

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


# Get version from version info
__version__ = None
version_file = os.path.join(filepath, "fast_simplification", "_version.py")
with io_open(version_file, mode="r") as fd:
    exec(fd.read())

# readme file
readme_file = os.path.join(filepath, "README.rst")


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
    name="fast_simplification",
    packages=["fast_simplification"],
    version=__version__,
    description="Wrapper around the Fast-Quadric-Mesh-Simplification library.",
    long_description=open(readme_file).read(),
    long_description_content_type="text/x-rst",
    author="Alex Kaszynski",
    author_email="akascap@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    url="https://github.com/pyvista/fast-simplification",
    python_requires=">=3.9",
    # Build cython modules
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            "fast_simplification._simplify",
            ["fast_simplification/_simplify.pyx"],
            language="c++",
            extra_compile_args=extra_compile_args,
            define_macros=macros,
        ),
        Extension(
            "fast_simplification._replay",
            ["fast_simplification/_replay.pyx"],
            language="c++",
            extra_compile_args=extra_compile_args,
            define_macros=macros,
        ),
    ],
    keywords="fast-simplification decimation",
    # install_requires=["numpy>=2.0"],
)
