Python Fast-Quadric-Mesh-Simplification Wrapper
===============================================
This is a python wrapping of the `Fast-Quadric-Mesh-Simplification Library
<https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification/>`_. Having
arrived at the same problem as the original author, but needing a Python
library, this project seeks to extend the work of the original library while
adding integration to Python and the `PyVista
<https://github.com/pyvista/pyvista>`_ project.

For the full documentation visit: https://pyvista.github.io/fast-simplification/

.. image:: https://github.com/pyvista/fast-simplification/raw/main/doc/images/simplify_demo.png

Installation
------------
Fast Simplification can be installed from PyPI using pip on Python >= 3.7::

  pip install fast-simplification

See the `Contributing <https://github.com/pyvista/fast-simplification#contributing>`_ for more details regarding development or if the installation through pip doesn't work out.

Basic Usage
-----------
The basic interface is quite straightforward and can work directly
with arrays of points and triangles:

.. code:: python

    points = [[ 0.5, -0.5, 0.0],
              [ 0.0, -0.5, 0.0],
              [-0.5, -0.5, 0.0],
              [ 0.5,  0.0, 0.0],
              [ 0.0,  0.0, 0.0],
              [-0.5,  0.0, 0.0],
              [ 0.5,  0.5, 0.0],
              [ 0.0,  0.5, 0.0],
              [-0.5,  0.5, 0.0]]

    faces = [[0, 1, 3],
             [4, 3, 1],
             [1, 2, 4],
             [5, 4, 2],
             [3, 4, 6],
             [7, 6, 4],
             [4, 5, 7],
             [8, 7, 5]]

    points_out, faces_out = fast_simplification.simplify(points, faces, 0.5)


Advanced Usage
--------------
This library supports direct integration with VTK through PyVista to
provide a simplistic interface to the library. As this library
provides a 4-5x improvement to the VTK decimation algorithms.

.. code:: python

   >>> from pyvista import examples
   >>> mesh = examples.download_nefertiti()
   >>> out = fast_simplification.simplify_mesh(mesh, target_reduction=0.9)

   Compare with built-in VTK/PyVista methods:

   >>> fas_sim = fast_simplification.simplify_mesh(mesh, target_reduction=0.9)
   >>> dec_std = mesh.decimate(0.9)  # vtkQuadricDecimation
   >>> dec_pro = mesh.decimate_pro(0.9)  # vtkDecimatePro

   >>> pv.set_plot_theme('document')
   >>> pl = pv.Plotter(shape=(2, 2), window_size=(1000, 1000))
   >>> pl.add_text('Original', 'upper_right', color='w')
   >>> pl.add_mesh(mesh, show_edges=True)
   >>> pl.camera_position = cpos

   >>> pl.subplot(0, 1)
   >>> pl.add_text(
   ...    'Fast-Quadric-Mesh-Simplification\n~2.2 seconds', 'upper_right', color='w'
   ... )
   >>> pl.add_mesh(fas_sim, show_edges=True)
   >>> pl.camera_position = cpos

   >>> pl.subplot(1, 0)
   >>> pl.add_mesh(dec_std, show_edges=True)
   >>> pl.add_text(
   ...    'vtkQuadricDecimation\n~9.5 seconds', 'upper_right', color='w'
   ... )
   >>> pl.camera_position = cpos

   >>> pl.subplot(1, 1)
   >>> pl.add_mesh(dec_pro, show_edges=True)
   >>> pl.add_text(
   ...    'vtkDecimatePro\n11.4~ seconds', 'upper_right', color='w'
   ... )
   >>> pl.camera_position = cpos
   >>> pl.show()


Comparison to other libraries
-----------------------------
The `pyfqmr <https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction>`_
library wraps the same header file as this library and has similar capabilities.
In this library, the decision was made to write the Cython layer on top of an
additional C++ layer rather than directly interfacing with wrapper from Cython.
This results in a mild performance improvement.

Reusing the example above:

.. code:: python

   Set up a timing function.

   >>> import pyfqmr
   >>> vertices = mesh.points
   >>> faces = mesh.faces.reshape(-1, 4)[:, 1:]
   >>> def time_pyfqmr():
   ...     mesh_simplifier = pyfqmr.Simplify()
   ...     mesh_simplifier.setMesh(vertices, faces)
   ...     mesh_simplifier.simplify_mesh(
   ...         target_count=out.n_faces, aggressiveness=7, verbose=0
   ...     )
   ...     vertices_out, faces_out, normals_out = mesh_simplifier.getMesh()
   ...     return vertices_out, faces_out, normals_out

Now, time it and compare with the non-VTK API of this library:

.. code:: python

   >>> timeit time_pyfqmr()
   2.75 s ± 5.35 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   >>> timeit vout, fout = fast_simplification.simplify(vertices, faces, 0.9)
   2.05 s ± 3.18 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

Additionally, the ``fast-simplification`` library has direct plugins
to the ``pyvista`` library, making it easy to read and write meshes:

.. code:: python

   >>> import pyvista
   >>> import fast_simplification
   >>> mesh = pyvista.read('my_mesh.stl')
   >>> simple = fast_simplification.simplify_mesh(mesh)
   >>> simple.save('my_simple_mesh.stl')

Since both libraries are based on the same core C++ code, feel free to
use whichever gives you the best performance and interoperability.

Replay decimation functionality
-------------------------------
This library also provides an interface to keep track of the successive
collapses that occur during the decimation process and to replay the
decimation process. This can be useful for different applications, such
as:

* applying the same decimation to a collection of meshes that share the
  same topology
* computing a correspondence map between the vertices of the original
  mesh and the vertices of the decimated mesh, to transfer field data from
  one to the other for example
* replaying the decimation process with a smaller target reduction than
  the original one, faster than decimating the original mesh with the
  smaller target reduction

To use this functionality, you need to set the ``return_collapses``
parameter to ``True`` when calling ``simplify``. This will return the
successive collapses of the decimation process in addition to points
and faces.

.. code:: python

   >>> import fast_simplification
   >>> import pyvista
   >>> mesh = pyvista.Sphere()
   >>> points, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
   >>> points_out, faces_out, collapses = fast_simplification.simplify(points, faces, 0.9, return_collapses=True)

Now you can call ``replay_simplification`` to replay the decimation process
and obtain the mapping between the vertices of the original mesh and the
vertices of the decimated mesh.

.. code:: python

   >>> points_out, faces_out, indice_mapping = fast_simplification.replay_simplification(points, faces, collapses)
   >>> i = 3
   >>> print(f'Vertex {i} of the original mesh is mapped to {indice_mapping[i]} of the decimated mesh')

You can also use the ``replay_simplification`` function to replay the
decimation process with a smaller target reduction than the original one.
This is faster than decimating the original mesh with the smaller target
reduction. To do so, you need to pass a subset of the collapses to the
``replay_simplification`` function. For example, to replay the decimation
process with a target reduction of 50% the initial rate, you can run:

.. code:: python

   >>> import numpy as np
   >>> collapses_half = collapses[:int(0.5 * len(collapses))]
   >>> points_out, faces_out, indice_mapping = fast_simplification.replay_simplification(points, faces, collapses_half)

If you have a collection of meshes that share the same topology, you can
apply the same decimation to all of them by calling ``replay_simplification``
with the same collapses for each mesh. This ensure that the decimated meshes
will share the same topology.

.. code:: python

   >>> import numpy as np
   >>> # Assume that you have a collection of meshes stored in a list meshes
   >>> _, _, collapses = fast_simplification.simplify(meshes[0].points, meshes[0].faces,
   ...                                                0.9, return_collapses=True)
   >>> decimated_meshes = []
   >>> for mesh in meshes:
   ...     points_out, faces_out, _ = fast_simplification.replay_simplification(mesh.points, mesh.faces, collapses)
   ...     decimated_meshes.append(pyvista.PolyData(points_out, faces_out))

Contributing
------------
Contribute to this repository by forking this repository and installing in
development mode with::

  git clone https://github.com/<USERNAME>/fast-simplification
  pip install -e .
  pip install -r requirements_test.txt

You can then add your feature or commit your bug fix and then run your unit
testing with::

  pytest

Unit testing will automatically enforce minimum code coverage standards.

Next, to ensure your code meets minimum code styling standards, run::

  pip install pre-commit
  pre-commit run --all-files

Finally, `create a pull request`_ from your fork and I'll be sure to review it.

.. _create a pull request: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
