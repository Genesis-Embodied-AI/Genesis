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
from OpenGL.GL import GLint, GLuint, GLvoidp, GLvoid, GLfloat, GLsizei, GLboolean, GLenum, GLsizeiptr, GLintptr


class GLWrapper:
    def __init__(self):
        self.gl_funcs = {}

        load_func = self.load_func
        load_func("glGetUniformLocation", GLint, GLuint, GLvoidp)
        load_func("glUniformMatrix4fv", GLvoid, GLint, GLsizei, GLboolean, GLvoidp)
        load_func("glUniform1i", GLvoid, GLint, GLint)
        load_func("glUniform1f", GLvoid, GLint, GLfloat)
        load_func("glUniform2f", GLvoid, GLint, GLfloat, GLfloat)
        load_func("glUniform3fv", GLvoid, GLint, GLsizei, GLvoidp)
        load_func("glUniform4fv", GLvoid, GLint, GLsizei, GLvoidp)
        load_func("glBindVertexArray", GLvoid, GLuint)
        load_func("glActiveTexture", GLvoid, GLenum)
        load_func("glBindTexture", GLvoid, GLenum, GLuint)
        load_func("glEnable", GLvoid, GLenum)
        load_func("glDisable", GLvoid, GLenum)
        load_func("glBlendFunc", GLvoid, GLenum, GLenum)
        load_func("glPolygonMode", GLvoid, GLenum, GLenum)
        load_func("glCullFace", GLvoid, GLenum)
        load_func("glDrawElementsInstanced", GLvoid, GLenum, GLsizei, GLenum, GLvoidp, GLsizei)
        load_func("glDrawArraysInstanced", GLvoid, GLenum, GLint, GLsizei, GLsizei)
        load_func("glUseProgram", GLvoid, GLuint)
        load_func("glFlush", GLvoid)
        load_func("glReadPixels", GLvoid, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, GLvoidp)
        load_func("glBindBuffer", GLvoid, GLenum, GLuint)
        load_func("glBufferData", GLvoid, GLenum, GLsizeiptr, GLvoidp, GLenum)
        load_func("glBufferSubData", GLvoid, GLenum, GLintptr, GLsizeiptr, GLvoidp)

        self.build_wrapper()

    def load_func(self, func_name, *signature):
        dll = GL.platform.PLATFORM.GL
        func_ptr = GL.platform.ctypesloader.buildFunction(
            GL.platform.PLATFORM.functionTypeFor(dll)(*signature),
            func_name,
            dll,
        )
        self.gl_funcs[func_name] = func_ptr

    def build_wrapper(self):
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

        self.wrapper_type = glfunc_type
        self.wrapper_instance = GLFunc()
