// wrap simplify header file for integration with cython
#include "Replay.h"

namespace Replay{

  // load collapses
  void load_collapses(const int n_coll, int* coll){
    collapses.clear();
    for (int ii = 0; ii < n_coll; ii ++){
      std::vector<int> c;
      c.push_back(coll[0 + 2*ii]);
      c.push_back(coll[1 + 2*ii]);
      collapses.push_back(c);
    }
  }

  // load points
  void load_points(const int n_points, float* points){
    vertices.clear();
    // load vertices
    for (int ii = 0; ii < n_points; ii ++){
      Vertex v;
      v.p.x = points[0 + 3*ii];
      v.p.y = points[1 + 3*ii];
      v.p.z = points[2 + 3*ii];
      vertices.push_back(v);
    }
  }

  // load triangles
  void load_triangles(const int n_tri, int* faces){
    triangles.clear();
    for (int ii = 0; ii < n_tri; ii ++){
      Triangle t;
      t.attr = 0;
      t.material = -1;
      t.v[0] = faces[0 + 3*ii];
      t.v[1] = faces[1 + 3*ii];
      t.v[2] = faces[2 + 3*ii];
      triangles.push_back(t);
    }
  }

  // load triangles
  void load_triangles_int64(const int n_tri, int64_t* faces){
    triangles.clear();
    for (int ii = 0; ii < n_tri; ii ++){
      Triangle t;
      t.attr = 0;
      t.material = -1;
      t.v[0] = faces[0 + 3*ii];
      t.v[1] = faces[1 + 3*ii];
      t.v[2] = faces[2 + 3*ii];
      triangles.push_back(t);
    }
  }

  // load triangles from vtk and deal with padding
  int load_triangles_from_vtk(const int n_tri, int* faces){
    triangles.clear();
    for (int ii = 0; ii < n_tri; ii ++){
      Triangle t;
      t.attr = 0;
      t.material = -1;
      if (faces[4*ii] != 3){
        return 1;
      }
      t.v[0] = faces[1 + 4*ii];
      t.v[1] = faces[2 + 4*ii];
      t.v[2] = faces[3 + 4*ii];
      triangles.push_back(t);
    }
    return 0;
  }

  void load_arrays_int32(const int n_points, const int n_tri, const int n_coll,
                         float* points, int* faces, int* collapses){
    load_points(n_points, points);
    load_triangles(n_tri, faces);
    load_collapses(n_coll, collapses);
  }

  void load_arrays_int64(const int n_points, const int n_tri, const int n_coll,
                         float* points, int64_t* faces, int* collapses){
    load_points(n_points, points);
    load_triangles_int64(n_tri, faces);
    load_collapses(n_coll, collapses);
  }

  int n_points(){
    return vertices.size();
  }

  int n_triangles(){
    return triangles.size();
  }

  int n_collapses(){
    return collapses.size();
  }

  // load triangles
  void load_triangles(const int n_tri, int64_t* faces){
    triangles.clear();
    for (int ii = 0; ii < n_tri; ii ++){
      Triangle t;
      t.attr = 0;
      t.material = -1;
      t.v[0] = faces[0 + 3*ii];
      t.v[1] = faces[1 + 3*ii];
      t.v[2] = faces[2 + 3*ii];
      triangles.push_back(t);
    }
  }

  // populate a contiguous array with the points in the vertices vector
  void get_points(float* points){

    // load vertices
    int n_points = vertices.size();
    for (int ii = 0; ii < n_points; ii ++){
      points[0 + 3*ii] = vertices[ii].p.x;
      points[1 + 3*ii] = vertices[ii].p.y;
      points[2 + 3*ii] = vertices[ii].p.z;
    }
  }

  // populate a contiguous array with the points in the vertices vector
  void get_triangles(int* tri){

    // load vertices
    int n_tri = triangles.size();
    for (int ii = 0; ii < n_tri; ii ++){
      tri[0 + 3*ii] = triangles[ii].v[0];
      tri[1 + 3*ii] = triangles[ii].v[1];
      tri[2 + 3*ii] = triangles[ii].v[2];
    }
  }

  void get_collapses(int* coll){

    // load vertices
    int n_collapse = collapses.size();
    for (int ii = 0; ii < n_collapse; ii ++){
      coll[0 + 2*ii] = collapses.at(ii).at(0);
      coll[1 + 2*ii] = collapses.at(ii).at(1);
    }
  }

  // populate a contiguous array with the points in the vertices vector
  int get_faces_int32(int32_t* tri){

    // load vertices
    int n_tri = triangles.size();
    int jj = 0;
    for (int ii = 0; ii < n_tri; ii ++){
      if (!triangles[ii].deleted){
        tri[0 + 3*jj] = 3;
        tri[1 + 3*jj] = triangles[ii].v[0];
        tri[2 + 3*jj] = triangles[ii].v[1];
        tri[3 + 3*jj] = triangles[ii].v[2];
        jj += 1;
      }
    }
    return jj;
  }

  // populate a contiguous array with the points in the vertices
  // vector without the vtk padding
  int get_faces_int32_no_padding(int32_t* tri){

    // load vertices
    int n_tri = triangles.size();
    int jj = 0;
    for (int ii = 0; ii < n_tri; ii ++){
      if (!triangles[ii].deleted){
        tri[0 + 3*jj] = triangles[ii].v[0];
        tri[1 + 3*jj] = triangles[ii].v[1];
        tri[2 + 3*jj] = triangles[ii].v[2];
        jj += 1;
      }
    }
    return jj;
  }

  // populate a contiguous array with the points in the vertices vector
  int get_faces_int64(int64_t* tri){

    // load vertices
    int n_tri = triangles.size();
    int jj = 0;
    for (int ii = 0; ii < n_tri; ii ++){
      if (!triangles[ii].deleted){
        tri[0 + 4*jj] = 3;
        tri[1 + 4*jj] = triangles[ii].v[0];
        tri[2 + 4*jj] = triangles[ii].v[1];
        tri[3 + 4*jj] = triangles[ii].v[2];
        jj += 1;
      }
    }
    return jj;
  }
}
