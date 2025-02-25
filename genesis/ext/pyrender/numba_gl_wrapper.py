import numpy as np
from numba import *
from numba import types
from numba.extending import (
    models,
    register_model,
    make_attribute_wrapper,
    typeof_impl,
    as_numba_type,
    unbox,
    NativeValue,
)
from numba.core import cgutils
from contextlib import ExitStack
import OpenGL.GL as GL
from OpenGL._bytes import as_8_bit
from OpenGL.GL import GLint, GLuint, GLvoidp, GLvoid, GLfloat, GLsizei, GLboolean, GLenum, GLsizeiptr, GLintptr

import genesis as gs


class GLWrapper:
    def __init__(self):
        self.gl_funcs = {}
        self._wrapper_type = None
        self._wrapper_instance = None

    def load_func(self, func_name, *signature):
        try:
            dll = GL.platform.PLATFORM.GL
            func_ptr = GL.platform.ctypesloader.buildFunction(
                GL.platform.PLATFORM.functionTypeFor(dll)(*signature),
                func_name,
                dll,
            )
        except AttributeError:
            pointer = GL.platform.PLATFORM.getExtensionProcedure(as_8_bit(func_name))
            func_ptr = GL.platform.PLATFORM.functionTypeFor(dll)(*signature)(pointer)
        self.gl_funcs[func_name] = func_ptr

    @property
    def wrapper_type(self):
        if self._wrapper_type is None:
            self.build_wrapper()
        return self._wrapper_type

    @property
    def wrapper_instance(self):
        if self._wrapper_instance is None:
            self.build_wrapper()
        return self._wrapper_instance

    def build_wrapper(self):
        load_func = self.load_func
        for name, signature in (
            ("glGetUniformLocation", (GLint, GLuint, GLvoidp)),
            ("glUniformMatrix4fv", (GLvoid, GLint, GLsizei, GLboolean, GLvoidp)),
            ("glUniform1i", (GLvoid, GLint, GLint)),
            ("glUniform1f", (GLvoid, GLint, GLfloat)),
            ("glUniform2f", (GLvoid, GLint, GLfloat, GLfloat)),
            ("glUniform3fv", (GLvoid, GLint, GLsizei, GLvoidp)),
            ("glUniform4fv", (GLvoid, GLint, GLsizei, GLvoidp)),
            ("glBindVertexArray", (GLvoid, GLuint)),
            ("glActiveTexture", (GLvoid, GLenum)),
            ("glBindTexture", (GLvoid, GLenum, GLuint)),
            ("glEnable", (GLvoid, GLenum)),
            ("glDisable", (GLvoid, GLenum)),
            ("glBlendFunc", (GLvoid, GLenum, GLenum)),
            ("glPolygonMode", (GLvoid, GLenum, GLenum)),
            ("glCullFace", (GLvoid, GLenum)),
            ("glDrawElementsInstanced", (GLvoid, GLenum, GLsizei, GLenum, GLvoidp, GLsizei)),
            ("glDrawArraysInstanced", (GLvoid, GLenum, GLint, GLsizei, GLsizei)),
            ("glDrawElementsInstancedBaseInstance", (GLvoid, GLenum, GLsizei, GLenum, GLvoidp, GLsizei, GLuint)),
            ("glDrawArraysInstancedBaseInstance", (GLvoid, GLenum, GLint, GLsizei, GLsizei, GLuint)),
            ("glUseProgram", (GLvoid, GLuint)),
            ("glFlush", (GLvoid,)),
            ("glReadPixels", (GLvoid, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, GLvoidp)),
            ("glBindBuffer", (GLvoid, GLenum, GLuint)),
            ("glBufferData", (GLvoid, GLenum, GLsizeiptr, GLvoidp, GLenum)),
            ("glBufferSubData", (GLvoid, GLenum, GLintptr, GLsizeiptr, GLvoidp)),
        ):
            try:
                load_func(name, *signature)
            except AttributeError:
                # OpenGL function not available, probably because the installed version does not support it (too old).
                # Moving to the next one without raising an exception since it is not blocking at this point.
                gs.logger.info(f"OpenGL function '{name}' not available on this machine.")

        funcs = self.gl_funcs
        func_types = {}
        for func_name in funcs:
            func_types[func_name] = typeof(funcs[func_name])

        class GLFunc:
            def __init__(self):
                for func_name in funcs:
                    setattr(self, func_name, funcs[func_name])

        class GLFuncType(types.Type):
            def __init__(self):
                super().__init__("GLFunc")

            def __eq__(self, other):
                return hasattr(other, "name") and other.name == "GLFunc"

            def __hash__(self):
                return hash("GLFuncType")

        glfunc_type = GLFuncType()

        @typeof_impl.register(GLFunc)
        def typeof_index(val, c):
            return glfunc_type

        as_numba_type.register(GLFunc, glfunc_type)

        @register_model(GLFuncType)
        class GLFuncModel(models.StructModel):
            def __init__(self, dmm, fe_type):
                members = list(func_types.items())
                super().__init__(dmm, fe_type, members)

        @unbox(GLFuncType)
        def unbox_interval(typ, obj, c):
            is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
            gl_func = cgutils.create_struct_proxy(typ)(c.context, c.builder)

            with ExitStack() as stack:
                for func_name in funcs:
                    func_obj = c.pyapi.object_getattr_string(obj, func_name)
                    with cgutils.early_exit_if_null(c.builder, stack, func_obj):
                        c.builder.store(cgutils.true_bit, is_error_ptr)
                    func_native = c.unbox(func_types[func_name], func_obj)
                    c.pyapi.decref(func_obj)
                    with cgutils.early_exit_if(c.builder, stack, func_native.is_error):
                        c.builder.store(cgutils.true_bit, is_error_ptr)

                    setattr(gl_func, func_name, func_native.value)

            return NativeValue(gl_func._getvalue(), is_error=c.builder.load(is_error_ptr))

        for func_name in funcs:
            make_attribute_wrapper(GLFuncType, func_name, func_name)

        self._wrapper_type = glfunc_type
        self._wrapper_instance = GLFunc()
