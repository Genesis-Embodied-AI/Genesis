"""
packing.py
------------

Pack rectangular regions onto larger rectangular regions.
"""

import numpy as np

from ..util import allclose, bounds_tree
from ..constants import log, tol

# floating point zero
_TOL_ZERO = 1e-12


class RectangleBin:
    """
    An N-dimensional binary space partition tree for packing
    hyper-rectangles. Split logic is pure `numpy` but behaves
    similarly to `scipy.spatial.Rectangle`.

    Mostly useful for packing 2D textures and 3D boxes and
    has not been tested outside of 2 and 3 dimensions.

    Original article about using this for packing textures:
    http://www.blackpawn.com/texts/lightmaps/
    """

    def __init__(self, bounds):
        """
        Create a rectangular bin.

        Parameters
        ------------
        bounds : (2, dimension *) float
          Bounds array are `[mins, maxes]`
        """
        # this is a *binary* tree so regardless of the dimensionality
        # of the rectangles each node has exactly two children
        self.child = []
        # is this node occupied.
        self.occupied = False
        # assume bounds are a list
        self.bounds = np.array(bounds, dtype=np.float64)

    @property
    def extents(self):
        """
        Bounding box size.

        Returns
        ----------
        extents : (dimension,) float
          Edge lengths of bounding box
        """
        bounds = self.bounds
        return bounds[1] - bounds[0]

    def insert(self, size, rotate=True):
        """
        Insert a rectangle into the bin.

        Parameters
        -------------
        size : (dimension,) float
          Size of rectangle to insert/

        Returns
        ----------
        inserted : (2,) float or None
          Position of insertion in the tree or None
          if the insertion was unsuccessful.
        """
        for child in self.child:
            # try inserting into child cells
            attempt = child.insert(size=size, rotate=rotate)
            if attempt is not None:
                return attempt

        # can't insert into occupied cells
        if self.occupied:
            return None

        # shortcut for our bounds
        bounds = self.bounds.copy()
        extents = bounds[1] - bounds[0]

        if rotate:
            # we are allowed to rotate the rectangle
            for roll in range(len(size)):
                size_test = extents - _roll(size, roll)
                fits = (size_test > -_TOL_ZERO).all()
                if fits:
                    size = _roll(size, roll)
                    break
            # we tried rotating and none of the directions fit
            if not fits:
                return None
        else:
            # compare the bin size to the insertion candidate size
            # manually compute extents here to avoid function call
            size_test = extents - size
            if (size_test < -_TOL_ZERO).any():
                return None

        # since the cell is big enough for the current rectangle, either it
        # is going to be inserted here, or the cell is going to be split
        # either way the cell is now occupied.
        self.occupied = True

        # this means the inserted rectangle fits perfectly
        # since we already checked to see if it was negative
        # no abs is needed
        if (size_test < _TOL_ZERO).all():
            return bounds

        # pick the axis to split along
        axis = size_test.argmax()
        # split hyper-rectangle along axis
        # note that split is *absolute* distance not offset
        # so we have to add the current min to the size
        splits = np.vstack((bounds, bounds))
        splits[1:3, axis] = bounds[0][axis] + size[axis]

        # assign two children
        self.child[:] = RectangleBin(splits[:2]), RectangleBin(splits[2:])

        # insert the requested item into the first child
        return self.child[0].insert(size, rotate=rotate)


def _roll(a, count):
    """
    A speedup for `numpy.roll` that only works
    on flat arrays and is fast on 2D and 3D and
    reverts to `numpy.roll` for other cases.

    Parameters
    -----------
    a : (n,) any
      Array to roll
    count : int
      Number of places to shift array

    Returns
    ---------
    rolled : (n,) any
      Input array shifted by requested amount

    """
    # a lookup table for roll in 2 and 3 dimensions
    lookup = [[[0, 1], [1, 0]], [[0, 1, 2], [2, 0, 1], [1, 2, 0]]]
    try:
        # roll the array using advanced indexing and a lookup table
        return a[lookup[len(a) - 2][count]]
    except IndexError:
        # failing that return the results using concat
        return np.concatenate([a[-count:], a[:-count]])


def rectangles_single(extents, size=None, shuffle=False, rotate=True):
    """
    Execute a single insertion order of smaller rectangles onto
    a larger rectangle using a binary space partition tree.

    Parameters
    ----------
    extents : (n, dimension) float
      The size of the hyper-rectangles to pack.
    size : None or (dim,) float
      Maximum size of container to pack onto.
      If not passed it will re-root the tree when items
      larger than any available node are inserted.
    shuffle : bool
      Whether or not to shuffle the insert order of the
      smaller rectangles, as the final packing density depends
      on insertion order.
    rotate : bool
      If True, allow integer-roll rotation.

    Returns
    ---------
    bounds : (m, 2, dim) float
      Axis aligned resulting bounds in space
    transforms : (m, dim + 1, dim + 1) float
      Homogeneous transformation including rotation.
    consume : (n,) bool
      Which of the original rectangles were packed,
      i.e. `consume.sum() == m`
    """

    extents = np.asanyarray(extents, dtype=np.float64)
    dimension = extents.shape[1]
    # the return arrays
    offset = np.zeros((len(extents), 2, dimension))
    consume = np.zeros(len(extents), dtype=bool)
    # start by ordering them by maximum length
    order = np.argsort(extents.max(axis=1))[::-1]

    if shuffle:
        # reorder with permutations
        order = np.random.permutation(order)

    if size is None:
        # if no bounds are passed start it with the size of a large
        # rectangle exactly which will require re-rooting for
        # subsequent insertions
        root_bounds = [[0.0] * dimension, extents[np.ptp(extents, axis=1).argmax()]]
    else:
        # restrict the bounds to passed size and disallow re-rooting
        root_bounds = [[0.0] * dimension, size]

    # the current root node to insert each rectangle
    root = RectangleBin(bounds=root_bounds)

    for index in order:
        # the current rectangle to be inserted
        rectangle = extents[index]
        # try to insert the hyper-rectangle into children
        inserted = root.insert(rectangle, rotate=rotate)

        if inserted is None and size is None:
            # we failed to insert into children
            # so we need to create a new parent
            # get the size of the current root node
            bounds = root.bounds
            # current extents
            current = np.ptp(bounds, axis=0)

            # pick the direction which has the least hyper-volume.
            best = np.inf
            for roll in range(len(current)):
                stack = np.array([current, _roll(rectangle, roll)])
                # we are going to combine two hyper-rect
                # so we have `dim` choices on ways to split
                # choose the split that minimizes the new hyper-volume
                # the new AABB is going to be the `max` of the lengths
                # on every dim except one which will be the `sum`
                ch = np.tile(stack.max(axis=0), (len(current), 1))
                np.fill_diagonal(ch, stack.sum(axis=0))

                # choose the new AABB by which one minimizes hyper-volume
                choice_prod = np.prod(ch, axis=1)
                if choice_prod.min() < best:
                    choices = ch
                    choices_idx = choice_prod.argmin()
                    best = choice_prod[choices_idx]
                if not rotate:
                    break

            # we now know the full extent of the AABB
            new_max = bounds[0] + choices[choices_idx]

            # offset the new bounding box corner
            new_min = bounds[0].copy()
            new_min[choices_idx] += current[choices_idx]

            # original bounds may be stretched
            new_ori_max = np.vstack((bounds[1], new_max)).max(axis=0)
            new_ori_max[choices_idx] = bounds[1][choices_idx]

            assert (new_ori_max >= bounds[1]).all()

            # the bounds containing the original sheet
            bounds_ori = np.array([bounds[0], new_ori_max])
            # the bounds containing the location to insert
            # the new rectangle
            bounds_ins = np.array([new_min, new_max])

            # generate the new root node
            new_root = RectangleBin([bounds[0], new_max])
            # this node has children so it is occupied
            new_root.occupied = True
            # create a bin for both bounds
            new_root.child = [RectangleBin(bounds_ori), RectangleBin(bounds_ins)]

            # insert the original sheet into the new tree
            root_offset = new_root.child[0].insert(np.ptp(bounds, axis=0), rotate=rotate)
            # we sized the cells so original tree would fit
            assert root_offset is not None

            # existing inserts need to be moved
            if not allclose(root_offset[0][0], 0.0):
                offset[consume] += root_offset[0][0]

            # insert the child that didn't fit before into the other child
            child = new_root.child[1].insert(rectangle, rotate=rotate)
            # since we re-sized the cells to fit insertion should always work
            assert child is not None

            offset[index] = child
            consume[index] = True
            # subsume the existing tree into a new root
            root = new_root

        elif inserted is not None:
            # we successfully inserted
            offset[index] = inserted
            consume[index] = True

    if tol.strict:
        # in tests make sure we've never returned overlapping bounds
        assert not bounds_overlap(offset[consume])

    return offset[consume], consume


def paths(paths, **kwargs):
    """
    Pack a list of Path2D objects into a rectangle.

    Parameters
    ------------
    paths: (n,) Path2D
      Geometry to be packed

    Returns
    ------------
    packed : trimesh.path.Path2D
      All paths packed into a single path object.
    transforms : (m, 3, 3) float
      Homogeneous transforms to move paths from their
      original position to the new one.
    consume : (n,) bool
      Which of the original paths were inserted,
      i.e. `consume.sum() == m`
    """
    from .util import concatenate

    # pack using exterior polygon which will have the
    # oriented bounding box calculated before packing
    packable = []
    original = []
    for index, path in enumerate(paths):
        quantity = path.metadata.get("quantity", 1)
        original.extend([index] * quantity)
        packable.extend([path.polygons_closed[path.root[0]]] * quantity)

    # pack the polygons using rectangular bin packing
    transforms, consume = polygons(polygons=packable, **kwargs)

    positioned = []
    for index, matrix in zip(np.nonzero(consume)[0], transforms):
        current = paths[original[index]].copy()
        current.apply_transform(matrix)
        positioned.append(current)

    # append all packed paths into a single Path object
    packed = concatenate(positioned)

    return packed, transforms, consume


def polygons(polygons, **kwargs):
    """
    Pack polygons into a rectangle by taking each Polygon's OBB
    and then packing that as a rectangle.

    Parameters
    ------------
    polygons : (n,) shapely.geometry.Polygon
      Source geometry
    **kwargs : dict
      Passed through to `packing.rectangles`.

    Returns
    -------------
    transforms : (m, 3, 3) float
      Homogeonous transforms from original frame to
      packed frame.
    consume : (n,) bool
      Which of the original polygons was packed,
      i.e. `consume.sum() == m`
    """

    from .polygons import polygons_obb, polygon_bounds

    # find the oriented bounding box of the polygons
    obb, extents = polygons_obb(polygons)

    # run packing for a number of iterations
    bounds, consume = rectangles(extents=extents, **kwargs)

    log.debug("%i/%i parts were packed successfully", consume.sum(), len(polygons))

    # transformations to packed positions
    roll = roll_transform(bounds=bounds, extents=extents[consume])

    transforms = np.array([np.dot(b, a) for a, b in zip(obb[consume], roll)])

    if tol.strict:
        # original bounds should not overlap
        assert not bounds_overlap(bounds)
        # confirm transfor
        check_bound = np.array(
            [polygon_bounds(polygons[index], matrix=m) for index, m in zip(np.nonzero(consume)[0], transforms)]
        )
        assert not bounds_overlap(check_bound)

    return transforms, consume


def rectangles(extents, size=None, density_escape=0.99, spacing=0.0, iterations=50, rotate=True, quanta=None):
    """
    Run multiple iterations of rectangle packing, this is the
    core function for all rectangular packing.

    Parameters
    ------------
    extents : (n, dimension) float
      Size of hyper-rectangle to be packed
    size : None or (dimension,) float
      Size of sheet to pack onto. If not passed tree will be allowed
      to create new volume-minimizing parent nodes.
    density_escape : float
      Exit early if rectangular density is above this threshold.
    spacing : float
      Distance to allow between rect
    iterations : int
      Number of iterations to run
    rotate : bool
      Allow right angle rotations or not.
    quanta : None or float
      Discrete "snap" interval.

    Returns
    ---------
    bounds :  (m, 2, dimension) float
      Axis aligned bounding boxes of inserted hyper-rectangle.
    inserted : (n,) bool
      Which of the original rect were packed.
    """
    # copy extents and make sure they are floats
    extents = np.array(extents, dtype=np.float64)
    dim = extents.shape[1]
    # add on any requested spacing
    extents += spacing * 2.0

    # hyper-volume: area in 2D, volume in 3D, party in 4D
    area = np.prod(extents, axis=1)
    # best density percentage in 0.0 - 1.0
    best_density = 0.0
    # how many rect were inserted
    best_count = 0

    for i in range(iterations):
        # run a single insertion order
        # don't shuffle the first run, shuffle subsequent runs
        bounds, insert = rectangles_single(extents=extents, size=size, shuffle=(i != 0))

        count = insert.sum()
        extents_all = np.ptp(bounds.reshape((-1, dim)), axis=0)

        if quanta is not None:
            # compute the density using an upsized quanta
            extents = np.ceil(extents_all / quanta) * quanta

        # calculate the packing density
        density = area[insert].sum() / np.prod(extents_all)

        # compare this packing density against our best
        if density > best_density or count > best_count:
            best_density = density
            best_count = count
            # save the result
            result = [bounds, insert]
            # exit early if everything is inserted and
            # we have exceeded our target density
            if density > density_escape and insert.all():
                break

    if spacing > 1e-12:
        # shrink the bounds by spacing
        result[0] += [[[spacing], [-spacing]]]

    log.debug("packed with density {:0.5f}".format(best_density))

    return result


def images(images, power_resize=False):
    """
    Pack a list of images and return result and offsets.

    Parameters
    ------------
    images : (n,) PIL.Image
      Images to be packed
    power_resize : bool
      Should the result image be upsized to the nearest
      power of two? Not every GPU supports materials that
      aren't a power of two size.

    Returns
    -----------
    packed : PIL.Image
      Multiple images packed into result
    offsets : (n, 2) int
       Offsets for original image to pack
    """
    from PIL import Image

    # use the number of pixels as the rectangle size
    bounds, insert = rectangles(extents=[i.size for i in images], rotate=False)
    # really should have inserted all the rect
    assert insert.all()

    # offsets should be integer multiple of pizels
    offset = bounds[:, 0].round().astype(int)
    extents = np.ptp(bounds.reshape((-1, 2)), axis=0)
    size = extents.round().astype(int)
    if power_resize:
        # round up all dimensions to powers of 2
        size = (2 ** np.ceil(np.log2(size))).astype(np.int64)

    # create the image in the mode of the first image
    result = Image.new(images[0].mode, tuple(size))
    # paste each image into the result
    for img, off in zip(images, offset):
        # box is upper left corner
        corner = (off[0], size[1] - img.size[1] - off[1])
        result.paste(img, box=corner)

    return result, offset


def meshes(meshes, **kwargs):
    """
    Pack 3D meshes into a rectangular volume using box packing.

    Parameters
    ------------
    meshes : (n,) trimesh.Trimesh
      Input geometry to pack
    **kwargs : dict
      Passed to `packing.rectangles`

    Returns
    ------------
    placed : (m,) trimesh.Trimesh
      Meshes moved into the rectangular volume.
    transforms : (m, 4, 4) float
      Homogeneous transform moving mesh from original
      position to being packed in a rectangular volume.
    consume : (n,) bool
      Which of the original meshes were inserted,
      i.e. `consume.sum() == m`
    """
    # pack meshes relative to their oriented bounding boxes
    obbs = [i.bounding_box_oriented for i in meshes]
    obb_extent = np.array([i.primitive.extents for i in obbs])
    obb_transform = np.array([o.primitive.transform for o in obbs])

    # run packing
    bounds, consume = rectangles(obb_extent, **kwargs)

    # generate the transforms from an origin centered AABB
    # to the final placed and rotated AABB
    transforms = np.array(
        [
            np.dot(r, np.linalg.inv(o))
            for o, r in zip(obb_transform[consume], roll_transform(bounds=bounds, extents=obb_extent[consume]))
        ],
        dtype=np.float64,
    )

    # copy the meshes and move into position
    placed = [meshes[index].copy().apply_transform(T) for index, T in zip(np.nonzero(consume)[0], transforms)]

    return placed, transforms, consume


def visualize(extents, bounds):
    """
    Visualize a 3D box packing.

    Parameters
    ------------
    extents : (n, 3) float
      AABB size before packing.
    bounds : (n, 2, 3) float
      AABB location after packing.

    Returns
    ------------
    scene : trimesh.Scene
      Scene with boxes at requested locations.
    """
    from ..scene import Scene
    from ..creation import box
    from ..visual import random_color

    # use a roll transform to verify extents
    transforms = roll_transform(bounds=bounds, extents=extents)
    meshes = [box(extents=e) for e in extents]

    for m, matrix, check in zip(meshes, transforms, bounds):
        m.apply_transform(matrix)
        assert np.allclose(m.bounds, check)
        m.visual.face_colors = random_color()
    return Scene(meshes)


def roll_transform(bounds, extents):
    """
    Packing returns rotations with integer "roll" which
    needs to be converted into a homogeneous rotation matrix.

    Currently supports `dimension=2` and `dimension=3`.

    Parameters
    --------------
    bounds : (n, 2, dimension) float
      Axis aligned bounding boxes of packed position
    extents : (n, dimension) float
      Original pre-rolled extents will be used
      to determine rotation to move to `bounds`.

    Returns
    ----------
    transforms : (n, dimension + 1, dimension + 1) float
      Homogeneous transformation to move cuboid at the origin
      into the position determined by `bounds`.
    """
    if len(bounds) != len(extents):
        raise ValueError("`bounds` must match `extents`")
    if len(extents) == 0:
        return []

    # find the size of the AABB of the passed bounds
    passed = np.ptp(bounds, axis=1)
    # zeroth index is 2D, `1` is 3D
    dimension = passed.shape[1]

    # store the resulting transformation matrices
    result = np.tile(np.eye(dimension + 1), (len(bounds), 1, 1))

    # a lookup table for rotations for rolling cuboiods
    # as `lookup[dimension - 2][roll]`
    # implemented for 2D and 3D
    lookup = [
        np.array([np.eye(3), np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])]),
        np.array(
            [
                np.eye(4),
                [[-0.0, -0.0, -1.0, -0.0], [-1.0, -0.0, -0.0, -0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                [[-0.0, -1.0, -0.0, -0.0], [0.0, 0.0, 1.0, 0.0], [-1.0, -0.0, -0.0, -0.0], [0.0, 0.0, 0.0, 1.0]],
            ]
        ),
    ]

    # rectangular rotation involves rolling
    for roll in range(extents.shape[1]):
        # find all the passed bounding boxes represented by
        # rolling the original extents by this amount
        rolled = np.roll(extents, roll, axis=1)
        # check to see if the rolled original extents
        # match the requested bounding box
        ok = np.ptp((passed - rolled), axis=1) < _TOL_ZERO
        if not ok.any():
            continue

        # the base rotation for this
        mat = lookup[dimension - 2][roll]
        # the lower corner of the AABB plus the rolled extent
        offset = np.tile(np.eye(dimension + 1), (ok.sum(), 1, 1))
        offset[:, :dimension, dimension] = bounds[:, 0][ok] + rolled[ok] / 2.0
        result[ok] = [np.dot(o, mat) for o in offset]

    if tol.strict:
        if dimension == 3:
            # make sure bounds match inputs
            from ..creation import box

            assert all(
                allclose(box(extents=e).apply_transform(m).bounds, b) for b, e, m in zip(bounds, extents, result)
            )
        elif dimension == 2:
            # in 2D check with a rectangle
            from .creation import rectangle

            assert all(
                allclose(rectangle(bounds=[-e / 2, e / 2]).apply_transform(m).bounds, b)
                for b, e, m in zip(bounds, extents, result)
            )
        else:
            raise ValueError("unsupported dimension")

    return result


def bounds_overlap(bounds, epsilon=1e-8):
    """
    Check to see if multiple axis-aligned bounding boxes
    contains overlaps using `rtree`.

    Parameters
    ------------
    bounds : (n, 2, dimension) float
      Axis aligned bounding boxes
    epsilon : float
      Amount to shrink AABB to avoid spurious floating
      point hits.

    Returns
    --------------
    overlap : bool
      True if any bound intersects any other bound.
    """
    # pad AABB by epsilon for deterministic intersections
    padded = np.array(bounds) + np.reshape([epsilon, -epsilon], (1, 2, 1))
    tree = bounds_tree(padded)
    # every returned AABB should not overlap with any other AABB
    return any(set(tree.intersection(current.ravel())) != {i} for i, current in enumerate(bounds))
