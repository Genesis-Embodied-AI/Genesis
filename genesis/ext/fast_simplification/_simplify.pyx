# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


import numpy as np

cimport numpy as np
from libc.stdint cimport int64_t
from libcpp cimport bool


cdef extern from "wrapper.h" namespace "Simplify":
    void load_arrays_int32(const int, const int, double*, int*)
    void load_arrays_int64(const int, const int, double*, int64_t*)
    void simplify_mesh(int, double aggressiveness, bool verbose)
    void simplify_mesh_lossless(bool)
    void get_points(double*)
    void get_triangles(int*)
    void get_collapses(int*)
    int get_faces_int32(int*)
    int get_faces_int32_no_padding(int*)
    int get_faces_int64(int64_t*)
    void write_obj(const char*)
    void load_obj(const char*, bool)
    int n_points()
    int n_triangles()
    int n_collapses()
    int load_triangles_from_vtk(const int, int*)
    void load_points(const int, double*)



def load_int32(int n_points, int n_faces, double [:, ::1] points, int [:, ::1] faces):
    load_arrays_int32(n_points, n_faces, &points[0, 0], &faces[0, 0])


def load_int64(
        int n_points, int n_faces, double [:, ::1] points, int64_t [:, ::1] faces
):
    load_arrays_int64(n_points, n_faces, &points[0, 0], &faces[0, 0])


def simplify(int target_count, double aggressiveness=7, bool verbose=False):
    simplify_mesh(target_count, aggressiveness, verbose)

def simplify_lossless(bool verbose=False):
    simplify_mesh_lossless(verbose)


def save_obj(filename):
    py_byte_string = filename.encode('UTF-8')
    cdef char* c_filename = py_byte_string
    write_obj(c_filename)


def read(filename):
    py_byte_string = filename.encode('UTF-8')
    cdef char* c_filename = py_byte_string
    load_obj(c_filename, False)


def return_points():
    cdef double [:, ::1] points = np.empty((n_points(), 3), np.float64)
    get_points(&points[0, 0])
    return np.array(points)


def return_triangles():
    cdef int [:, ::1] triangles = np.empty((n_triangles(), 3), np.int32)
    get_triangles(&triangles[0, 0])
    return np.array(triangles)

def return_collapses():
    cdef int [:, ::1] collapses = np.empty((n_collapses(), 2), np.int32)
    get_collapses(&collapses[0, 0])
    return np.array(collapses)


def return_faces_int32_no_padding():
    """VTK formatted faces"""
    cdef int [::1] faces = np.empty(n_triangles()*3, np.int32)
    n_tri = get_faces_int32_no_padding(&faces[0])
    return np.array(faces[:n_tri*3])


def return_faces_int32():
    """VTK formatted faces"""
    cdef int [::1] faces = np.empty(n_triangles()*4, np.int32)
    n_tri = get_faces_int32(&faces[0])
    return np.array(faces[:n_tri*4])


def return_faces_int64():
    """VTK formatted faces"""
    cdef int64_t [::1] faces = np.empty(n_triangles()*4, np.int64)
    n_tri = get_faces_int64(&faces[0])
    return np.array(faces[:n_tri*4])


def load_from_vtk(int n_points, double [:, ::1] points, int [::1] faces, int n_faces):
    result = load_triangles_from_vtk(n_faces, &faces[0])
    if result:
        raise ValueError(
            "Input mesh ``mesh`` must consist of only triangles.\n"
            "Run ``.triangulate()`` to convert to an all triangle mesh."
        )
    load_points(n_points, &points[0, 0])
