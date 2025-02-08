# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


import numpy as np

cimport numpy as np
from libc.stdint cimport int64_t
from libcpp cimport bool


cdef extern from "wrapper_replay.h" namespace "Replay":
    void load_arrays_int32(const int, const int, const int, float*, int*, int*)
    void load_arrays_int64(const int, const int, const int,  float*, int64_t*, int*)
    void replay_simplification()
    void get_points(float*)
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
    void load_points(const int, float*)
    void load_collapses(const int, int*)

def load_int32(int n_points, int n_faces, int n_collapses, float [:, ::1] points, int [:, ::1] faces, int [:, ::1] collapses):
    load_arrays_int32(n_points, n_faces, n_collapses, &points[0, 0], &faces[0, 0], &collapses[0, 0])


def load_int64(
        int n_points, int n_faces, int n_collapses, float [:, ::1] points, int64_t [:, ::1] faces, int [:, ::1] collapses
):
    load_arrays_int64(n_points, n_faces, n_collapses, &points[0, 0], &faces[0, 0], &collapses[0, 0])


# def simplify(int target_count, double aggressiveness=7, bool verbose=False):
#     simplify_mesh(target_count, aggressiveness, verbose)

def replay():
    replay_simplification()


def save_obj(filename):
    py_byte_string = filename.encode('UTF-8')
    cdef char* c_filename = py_byte_string
    write_obj(c_filename)


def read(filename):
    py_byte_string = filename.encode('UTF-8')
    cdef char* c_filename = py_byte_string
    load_obj(c_filename, False)


def return_points():
    cdef float [:, ::1] points = np.empty((n_points(), 3), np.float32)
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


def load_from_vtk(int n_points, float [:, ::1] points, int [::1] faces, int n_faces):
    result = load_triangles_from_vtk(n_faces, &faces[0])
    if result:
        raise ValueError(
            "Input mesh ``mesh`` must consist of only triangles.\n"
            "Run ``.triangulate()`` to convert to an all triangle mesh."
        )
    load_points(n_points, &points[0, 0])


def compute_indice_mapping(int[:, :] collapses, int n_points):

    ''' Compute the mapping from original indices to new indices after collapsing
        edges

        (pure python implementation with numpy)
    '''

    # start with identity mapping
    indice_mapping = np.arange(n_points, dtype=int)

    # First round of mapping
    origin_indices = collapses[:, 1]
    indice_mapping[origin_indices] = collapses[:, 0]
    previous = np.zeros(len(indice_mapping))
    while not np.array_equal(previous, indice_mapping):
        previous = indice_mapping.copy()
        indice_mapping[origin_indices] = indice_mapping[
            indice_mapping[origin_indices]
        ]

    keep = np.setdiff1d(
        np.arange(n_points), collapses[:, 1]
    )  # Indices of the points that must be kept after decimation

    cdef int i = 0
    cdef int j = 0

    cdef int[:] application = np.zeros(n_points, dtype=np.int32)
    for i in range(n_points):
        if j == len(keep):
            break
        if i == keep[j]:
            application[i] = j
            j += 1

    indice_mapping = np.array(application)[indice_mapping]

    return indice_mapping


def clean_triangles_and_edges(int[:, :] mapped_triangles, bool clean_edges=False):
    """Return the edges and triangles of a mesh from mapped triangles

    Args:
        mapped_triangles (np.ndarray): Mapped triangles
        clean_edges (bool, optional): If True, remove duplicated edges.

    Returns:
        np.ndarray: Edges
        np.ndarray: Triangles
    """

    cdef int i, j, k, l
    cdef int n_edges = 0
    cdef int n_triangles = 0
    cdef int N = len(mapped_triangles)
    cdef int[:, :] edges_with_rep = np.zeros((N, 2), dtype=np.int32)
    cdef int[:, :] triangles = np.zeros((N, 3), dtype=np.int32)

    for i in range(N):
        j = mapped_triangles[i, 0]
        k = mapped_triangles[i, 1]
        l = mapped_triangles[i, 2]

        if j != k and j != l and k != l:
            triangles[n_triangles, 0] = j
            triangles[n_triangles, 1] = k
            triangles[n_triangles, 2] = l
            n_triangles += 1

        elif j != k:
            # j, k = np.sort([j, k])
            edges_with_rep[n_edges, 0] = j
            edges_with_rep[n_edges, 1] = k
            n_edges += 1

        elif j != l:
            # j, l = np.sort([j, l])
            edges_with_rep[n_edges, 0] = j
            edges_with_rep[n_edges, 1] = l
            n_edges += 1

        elif l != k:
            # l, k = np.sort([j, k])
            edges_with_rep[n_edges, 0] = l
            edges_with_rep[n_edges, 1] = k
            n_edges += 1

    if not clean_edges:

        return np.asarray(edges_with_rep)[:n_edges, :], np.asarray(triangles)[:n_triangles, :]


    cdef int[:, :] edges = np.zeros((n_edges, 2), dtype=np.int32)


    # Lexicographic sort
    cdef int[:] order = np.lexsort((np.asarray(edges_with_rep[:n_edges, 1]), np.asarray(edges_with_rep[:n_edges, 0])))
    # Remove duplicates
    cdef int n_keep_edges = 1
    edges[0, :] = edges_with_rep[order[0], :]
    print(f"n_edges : {n_edges}")
    for i in range(1, n_edges):
        if (edges_with_rep[order[i], 0] != edges_with_rep[order[i - 1], 0]) or (edges_with_rep[order[i], 1] != edges_with_rep[order[i - 1], 1]):
            edges[n_keep_edges, :] = edges_with_rep[order[i], :]
            n_keep_edges += 1


    return np.asarray(edges)[:n_keep_edges, :], np.asarray(triangles)[:n_triangles, :]
