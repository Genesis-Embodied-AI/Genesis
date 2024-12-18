from . import interfaces


def convex_decomposition(mesh, **kwargs):
    """
    Compute an approximate convex decomposition of a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to be decomposed into convex parts

    Returns
    -------
    mesh_args : list
      List of **kwargs for Trimeshes that are nearly
      convex and approximate the original.
    """
    # decompositions require testVHACD
    if interfaces.vhacd.exists:
        return interfaces.vhacd.convex_decomposition(mesh, **kwargs)
    else:
        raise ValueError("convex compositions require testVHACD installed!")
