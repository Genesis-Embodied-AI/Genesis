import numpy as np
from ..util import is_ccw  # NOQA


def concatenate(paths):
    """
    Concatenate multiple paths into a single path.

    Parameters
    -------------
    paths : (n,) Path
      Path objects to concatenate

    Returns
    -------------
    concat : Path, Path2D, or Path3D
      Concatenated result
    """
    # if only one path object just return copy
    if len(paths) == 1:
        return paths[0].copy()

    # upgrade to 3D if we have mixed 2D and 3D paths
    dimensions = set(i.vertices.shape[1] for i in paths)
    if len(dimensions) > 1:
        paths = [i.to_3D() if hasattr(i, "to_3D") else i for i in paths]

    # length of vertex arrays
    vert_len = np.array([len(i.vertices) for i in paths])
    # how much to offset each paths vertex indices by
    offsets = np.append(0.0, np.cumsum(vert_len))[:-1].astype(np.int64)

    # resulting entities
    entities = []
    # resulting vertices
    vertices = []
    # resulting metadata
    metadata = {}
    for path, offset in zip(paths, offsets):
        # update metadata
        metadata.update(path.metadata)
        # copy vertices, we will stack later
        vertices.append(path.vertices.copy())
        # copy entity then reindex points
        for entity in path.entities:
            # cleanly copy the entity into a new object
            copied = entity.copy()
            # offset the indexes
            copied.points += offset
            entities.append(copied)
    # generate the single new concatenated path
    # use input types so we don't have circular imports
    concat = type(path)(metadata=metadata, entities=entities, vertices=np.vstack(vertices))
    return concat
