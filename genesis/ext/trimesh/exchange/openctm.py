# Modified from the original OpenCTM python binding
# for inclusion in the `trimesh` package:
# https://github.com/mikedh/trimesh
#
# To get shared library this binding imports, you can download
# and install it on Linux using this bash script:
#  https://github.com/mikedh/trimesh/blob/master/docker/builds/openctm.bash
# ------------------------------------------------------------------------------
# Copyright (c) 2009-2010 Marcus Geelnard
#
# This software is provided 'as-is', without any express or implied
# warranty. In no event will the authors be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
#     1. The origin of this software must not be misrepresented; you must not
#     claim that you wrote the original software. If you use this software
#     in a product, an acknowledgment in the product documentation would be
#     appreciated but is not required.
#
#     2. Altered source versions must be plainly marked as such, and must not
#     be misrepresented as being the original software.
#
#     3. This notice may not be removed or altered from any source
#     distribution.
# ------------------------------------------------------------------------------

import os
import ctypes
import ctypes.util

import numpy as np

_ctm_loaders = {}

try:

    # try to find the shared library
    _ctm_lib_name = ctypes.util.find_library("openctm")
    if os.name == "nt":
        _ctm_loader = ctypes.WinDLL
    else:
        _ctm_loader = ctypes.CDLL
    if _ctm_lib_name is None or len(_ctm_lib_name) == 0:
        raise ImportError("libopenctm library not found!")
except BaseException as E:
    from ..exceptions import ExceptionWrapper

    _ctm_lib_name = None
    _ctm_loader = ExceptionWrapper(E)


def load_ctm(file_obj, file_type=None, **kwargs):
    """
    Load OpenCTM files from a file object.

    Parameters
    ----------
    file_obj : file object
      Open file-like object with CTM data.

    Returns
    ----------
    loaded : dict
     Keyword arguments for the Trimesh constructor
    """
    # actually load the library here
    _ctm_lib = _ctm_loader(_ctm_lib_name)

    # Types
    CTMfloat = ctypes.c_float
    CTMint = ctypes.c_int32
    CTMuint = ctypes.c_uint32
    CTMcontext = ctypes.c_void_p
    CTMenum = ctypes.c_uint32

    # boolean
    CTM_TRUE = 1
    # CTM_FALSE = 0

    # CTMenum
    CTM_NONE = 0x0000
    CTM_IMPORT = 0x0101
    # CTM_EXPORT = 0x0102
    CTM_VERTEX_COUNT = 0x0301
    CTM_TRIANGLE_COUNT = 0x0302
    CTM_HAS_NORMALS = 0x0303
    CTM_INDICES = 0x0601
    CTM_VERTICES = 0x0602
    CTM_NORMALS = 0x0603

    # Functions
    ctmNewContext = _ctm_lib.ctmNewContext
    ctmNewContext.argtypes = [CTMenum]
    ctmNewContext.restype = CTMcontext
    ctmFreeContext = _ctm_lib.ctmFreeContext
    ctmFreeContext.argtypes = [CTMcontext]
    ctmGetError = _ctm_lib.ctmGetError
    ctmGetError.argtypes = [CTMcontext]
    ctmGetError.restype = CTMenum
    ctmErrorString = _ctm_lib.ctmErrorString
    ctmErrorString.argtypes = [CTMenum]
    ctmErrorString.restype = ctypes.c_char_p
    ctmGetInteger = _ctm_lib.ctmGetInteger
    ctmGetInteger.argtypes = [CTMcontext, CTMenum]
    ctmGetInteger.restype = CTMint
    ctmGetFloat = _ctm_lib.ctmGetFloat
    ctmGetFloat.argtypes = [CTMcontext, CTMenum]
    ctmGetFloat.restype = CTMfloat
    ctmGetIntegerArray = _ctm_lib.ctmGetIntegerArray
    ctmGetIntegerArray.argtypes = [CTMcontext, CTMenum]
    ctmGetIntegerArray.restype = ctypes.POINTER(CTMuint)
    ctmGetFloatArray = _ctm_lib.ctmGetFloatArray
    ctmGetFloatArray.argtypes = [CTMcontext, CTMenum]
    ctmGetFloatArray.restype = ctypes.POINTER(CTMfloat)
    ctmLoad = _ctm_lib.ctmLoad
    ctmLoad.argtypes = [CTMcontext, ctypes.c_char_p]
    ctmSave = _ctm_lib.ctmSave
    ctmSave.argtypes = [CTMcontext, ctypes.c_char_p]

    ctm = ctmNewContext(CTM_IMPORT)

    # !!load file from name
    # this should be replaced with something that
    # actually uses the file object data to support streams
    name = str(file_obj.name).encode("utf-8")
    ctmLoad(ctm, name)

    err = ctmGetError(ctm)
    if err != CTM_NONE:
        raise IOError("Error loading file: " + str(ctmErrorString(err)))

    # get vertices
    vertex_count = ctmGetInteger(ctm, CTM_VERTEX_COUNT)
    vertex_ctm = ctmGetFloatArray(ctm, CTM_VERTICES)
    # use fromiter to avoid loop
    vertices = np.fromiter(vertex_ctm, dtype=np.float64, count=vertex_count * 3).reshape((-1, 3))
    # get faces
    face_count = ctmGetInteger(ctm, CTM_TRIANGLE_COUNT)
    face_ctm = ctmGetIntegerArray(ctm, CTM_INDICES)
    faces = np.fromiter(face_ctm, dtype=np.int64, count=face_count * 3).reshape((-1, 3))

    # create kwargs for trimesh constructor
    result = {"vertices": vertices, "faces": faces}

    # get face normals if available
    if ctmGetInteger(ctm, CTM_HAS_NORMALS) == CTM_TRUE:
        normals_ctm = ctmGetFloatArray(ctm, CTM_NORMALS)
        normals = np.fromiter(normals_ctm, dtype=np.float64, count=face_count * 3).reshape((-1, 3))
        result["face_normals"] = normals

    # free context
    ctmFreeContext(ctm)

    return result


if _ctm_lib_name is not None:
    # we have a library so add load_ctm
    _ctm_loaders = {"ctm": load_ctm}
