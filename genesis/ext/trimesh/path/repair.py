"""
repair.py
--------------

Try to fix problems with closed regions.
"""

from . import segments
from .. import util

import numpy as np
from scipy.spatial import cKDTree


def fill_gaps(path, distance=0.025):
    """
    For 3D line segments defined by two points, turn
    them in to an origin defined as the closest point along
    the line to the zero origin as well as a direction vector
    and start and end parameter.

    Parameters
    ------------
    segments : (n, 2, 3) float
       Line segments defined by start and end points

    Returns
    --------------
    origins : (n, 3) float
       Point on line closest to [0, 0, 0]
    vectors : (n, 3) float
       Unit line directions
    parameters : (n, 2) float
       Start and end distance pairs for each line
    """

    # find any vertex without degree 2 (connected to two things)
    broken = np.array([k for k, d in dict(path.vertex_graph.degree()).items() if d != 2])

    # if all vertices have correct connectivity, exit
    if len(broken) == 0:
        return

    # first find broken vertices with distance
    tree = cKDTree(path.vertices[broken])
    pairs = tree.query_pairs(r=distance, output_type="ndarray")

    connect_seg = []
    if len(pairs) > 0:
        end_points = {tuple(sorted(e.end_points)) for e in path.entities}
        pair_set = {tuple(i) for i in np.sort(broken[pairs], axis=1)}

        # we don't want to connect entities to themselves so do a set
        # difference
        mask = np.array(list(pair_set.difference(end_points)))

        if len(mask) > 0:
            connect_seg = path.vertices[mask]

    # a set of values we can query intersections with quickly
    broken_set = set(broken)
    # query end points set vs path.dangling to avoid having
    # to compute every single path and discrete curve
    dangle = [i for i, e in enumerate(path.entities) if len(broken_set.intersection(e.end_points)) > 0]

    segs = []
    # mask for which entities to keep
    keep = np.ones(len(path.entities), dtype=bool)
    # save a reference to the line class to avoid circular import
    line_class = None

    for entity_index in dangle:
        # only consider line entities
        if path.entities[entity_index].__class__.__name__ != "Line":
            continue

        if line_class is None:
            line_class = path.entities[entity_index].__class__

        # get discrete version of entity
        points = path.entities[entity_index].discrete(path.vertices)
        # turn connected curve into segments
        seg_idx = util.stack_lines(np.arange(len(points)))
        # append the segments to our collection
        segs.append(points[seg_idx])
        # remove this entity and replace with segments
        keep[entity_index] = False

    # combine segments with connection segments
    all_segs = util.vstack_empty((util.vstack_empty(segs), connect_seg))

    # go home early
    if len(all_segs) == 0:
        return

    # split segments at broken vertices so topology can happen
    split = segments.split(all_segs, path.vertices[broken])
    # merge duplicate segments
    final_seg = segments.unique(split)

    # add line segments in as line entities
    entities = []
    for i in range(len((final_seg))):
        entities.append(line_class(points=np.arange(2) + (i * 2) + len(path.vertices)))

    # replace entities with new entities
    path.entities = np.append(path.entities[keep], entities)
    path.vertices = np.vstack((path.vertices, np.vstack(final_seg)))
    path._cache.clear()
    path.process()
