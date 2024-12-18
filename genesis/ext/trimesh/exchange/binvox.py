"""
Parsing functions for Binvox files.

https://www.patrickmin.com/binvox/binvox.html

Exporting meshes as binvox files requires the
`binvox` executable to be in your path.
"""

import os
import subprocess
import numpy as np
import collections

from .. import util
from ..base import Trimesh

# find the executable for binvox in PATH
binvox_encoder = util.which("binvox")
Binvox = collections.namedtuple("Binvox", ["rle_data", "shape", "translate", "scale"])


def parse_binvox_header(fp):
    """
    Read the header from a binvox file.
    Spec available:
    https://www.patrickmin.com/binvox/binvox.html

    Parameters
    ------------
    fp: file-object
      File like object with binvox file

    Returns
    ----------
    shape : tuple
      Shape of binvox according to binvox spec
    translate : tuple
      Translation
    scale : float
      Scale of voxels

    Raises
    ------------
    IOError
      If invalid binvox file.
    """

    line = fp.readline().strip()
    if hasattr(line, "decode"):
        binvox = b"#binvox"
        space = b" "
    else:
        binvox = "#binvox"
        space = " "
    if not line.startswith(binvox):
        raise IOError("Not a binvox file")
    shape = tuple(int(s) for s in fp.readline().strip().split(space)[1:])
    translate = tuple(float(s) for s in fp.readline().strip().split(space)[1:])
    scale = float(fp.readline().strip().split(space)[1])
    fp.readline()
    return shape, translate, scale


def parse_binvox(fp, writeable=False):
    """
    Read a binvox file, spec at
    https://www.patrickmin.com/binvox/binvox.html

    Parameters
    ------------
    fp: file-object
      File like object with binvox file

    Returns
    ----------
    binvox : namedtuple
      Containing data
    rle : numpy array
      Run length encoded data

    Raises
    ------------
    IOError
      If invalid binvox file
    """
    # get the header info
    shape, translate, scale = parse_binvox_header(fp)
    # get the rest of the file
    data = fp.read()
    # convert to numpy array
    rle_data = np.frombuffer(data, dtype=np.uint8)
    if writeable:
        rle_data = rle_data.copy()
    return Binvox(rle_data, shape, translate, scale)


_binvox_header = """#binvox 1
dim {sx} {sy} {sz}
translate {tx} {ty} {tz}
scale {scale}
data
"""


def binvox_header(shape, translate, scale):
    """
    Get a binvox header string.

    Parameters
    --------
    shape: length 3 iterable of ints denoting shape of voxel grid.
    translate: length 3 iterable of floats denoting translation.
    scale: num length of entire voxel grid.

    Returns
    --------
    string including "data\n" line.
    """
    sx, sy, sz = (int(s) for s in shape)
    tx, ty, tz = translate
    return _binvox_header.format(sx=sx, sy=sy, sz=sz, tx=tx, ty=ty, tz=tz, scale=scale)


def binvox_bytes(rle_data, shape, translate=(0, 0, 0), scale=1):
    """Get a binary representation of binvox data.

    Parameters
    --------
    rle_data : numpy array
      Run-length encoded numpy array.
    shape : (3,) int
      Shape of voxel grid.
    translate : (3,) float
      Translation of voxels
    scale : float
      Length of entire voxel grid.

    Returns
    --------
    data : bytes
      Suitable for writing to binary file
    """
    if rle_data.dtype != np.uint8:
        raise ValueError("rle_data.dtype must be np.uint8, got %s" % rle_data.dtype)

    header = binvox_header(shape, translate, scale).encode()
    return header + rle_data.tobytes()


def voxel_from_binvox(rle_data, shape, translate=None, scale=1.0, axis_order="xzy"):
    """
    Factory for building from data associated with binvox files.

    Parameters
    ---------
    rle_data : numpy
      Run-length-encoded of flat voxel
      values, or a `trimesh.rle.RunLengthEncoding` object.
      See `trimesh.rle` documentation for description of encoding
    shape : (3,) int
      Shape of voxel grid.
    translate : (3,) float
      Translation of voxels
    scale : float
      Length of entire voxel grid.
    encoded_axes : iterable
      With values in ('x', 'y', 'z', 0, 1, 2),
      where x => 0, y => 1, z => 2
      denoting the order of axes in the encoded data. binvox by
      default saves in xzy order, but using `xyz` (or (0, 1, 2)) will
      be faster in some circumstances.

    Returns
    ---------
    result : VoxelGrid
      Loaded voxels
    """
    # shape must be uniform else scale is ambiguous
    from ..voxel import encoding as enc
    from ..voxel.base import VoxelGrid

    from .. import transformations

    if isinstance(rle_data, enc.RunLengthEncoding):
        encoding = rle_data
    else:
        encoding = enc.RunLengthEncoding(rle_data, dtype=bool)

    # translate = np.asanyarray(translate) * scale)
    # translate = [0, 0, 0]
    transform = transformations.scale_and_translate(scale=scale / (np.array(shape) - 1), translate=translate)

    if axis_order == "xzy":
        perm = (0, 2, 1)
        shape = tuple(shape[p] for p in perm)
        encoding = encoding.reshape(shape).transpose(perm)
    elif axis_order is None or axis_order == "xyz":
        encoding = encoding.reshape(shape)
    else:
        raise ValueError("Invalid axis_order '%s': must be None, 'xyz' or 'xzy'")

    assert encoding.shape == shape

    return VoxelGrid(encoding, transform)


def load_binvox(file_obj, resolver=None, axis_order="xzy", file_type=None):
    """
    Load trimesh `VoxelGrid` instance from file.

    Parameters
    -----------
    file_obj : file-like object
      Contains binvox data
    resolver : unused
    axis_order : str
      Order of axes in encoded data.
      Binvox default is 'xzy', but 'xyz' may be faster
      where this is not relevant.

    Returns
    ---------
    result : trimesh.voxel.VoxelGrid
      Loaded voxel data
    """
    if file_type is not None and file_type != "binvox":
        raise ValueError("file_type must be None or binvox, got %s" % file_type)
    data = parse_binvox(file_obj, writeable=True)
    return voxel_from_binvox(
        rle_data=data.rle_data, shape=data.shape, translate=data.translate, scale=data.scale, axis_order=axis_order
    )


def export_binvox(voxel, axis_order="xzy"):
    """
    Export `trimesh.voxel.VoxelGrid` instance to bytes

    Parameters
    ------------
    voxel : `trimesh.voxel.VoxelGrid`
      Assumes axis ordering of `xyz` and encodes
      in binvox default `xzy` ordering.
    axis_order : str
      Eements in ('x', 'y', 'z', 0, 1, 2), the order
      of axes to encode data (standard is 'xzy' for binvox). `voxel`
      data is assumed to be in order 'xyz'.

    Returns
    -----------
    result : bytes
      Representation according to binvox spec
    """
    translate = voxel.translation
    scale = voxel.scale * ((np.array(voxel.shape) - 1))
    (neg_scale,) = np.where(scale < 0)
    encoding = voxel.encoding.flip(neg_scale)
    scale = np.abs(scale)
    if not util.allclose(scale[0], scale[1:], 1e-6 * scale[0] + 1e-8):
        raise ValueError("Can only export binvox with uniform scale")
    scale = scale[0]
    if axis_order == "xzy":
        encoding = encoding.transpose((0, 2, 1))
    elif axis_order != "xyz":
        raise ValueError('Invalid axis_order: must be one of ("xyz", "xzy")')
    rle_data = encoding.flat.run_length_data(dtype=np.uint8)
    return binvox_bytes(rle_data, shape=voxel.shape, translate=translate, scale=scale)


class Binvoxer(object):
    """
    Interface for binvox CL tool.

    This class is responsible purely for making calls to the CL tool. It
    makes no attempt to integrate with the rest of trimesh at all.

    Constructor args configure command line options.

    `Binvoxer.__call__` operates on the path to a mode file.

    If using this interface in published works, please cite the references
    below.

    See CL tool website for further details.

    https://www.patrickmin.com/binvox/

    @article{nooruddin03,
        author = {Fakir S. Nooruddin and Greg Turk},
        title = {Simplification and Repair of Polygonal Models Using Volumetric
                 Techniques},
        journal = {IEEE Transactions on Visualization and Computer Graphics},
        volume = {9},
        number = {2},
        pages = {191--205},
        year = {2003}
    }

    @Misc{binvox,
        author = {Patrick Min},
        title =  {binvox},
        howpublished = {{\tt http://www.patrickmin.com/binvox} or
                        {\tt https://www.google.com/search?q=binvox}},
        year =  {2004 - 2019},
        note = {Accessed: yyyy-mm-dd}
    }
    """

    SUPPORTED_INPUT_TYPES = (
        "ug",
        "obj",
        "off",
        "dfx",
        "xgl",
        "pov",
        "brep",
        "ply",
        "jot",
    )

    SUPPORTED_OUTPUT_TYPES = (
        "binvox",
        "hips",
        "mira",
        "vtk",
        "raw",
        "schematic",
        "msh",
    )

    def __init__(
        self,
        dimension=32,
        file_type="binvox",
        z_buffer_carving=True,
        z_buffer_voting=True,
        dilated_carving=False,
        exact=True,
        bounding_box=None,
        remove_internal=False,
        center=False,
        rotate_x=0,
        rotate_z=0,
        wireframe=False,
        fit=False,
        block_id=None,
        use_material_block_id=False,
        use_offscreen_pbuffer=False,
        downsample_factor=None,
        downsample_threshold=None,
        verbose=False,
        binvox_path=None,
    ):
        """
        Configure the voxelizer.

        Parameters
        ------------
        dimension: voxel grid size (max 1024 when not using exact)
        file_type: str
          Output file type, supported types are:
            'binvox'
            'hips'
            'mira'
            'vtk'
            'raw'
            'schematic'
            'msh'
        z_buffer_carving : use z buffer based carving. At least one of
            `z_buffer_carving` and `z_buffer_voting` must be True.
        z_buffer_voting: use z-buffer based parity voting method.
        dilated_carving: stop carving 1 voxel before intersection.
        exact: any voxel with part of a triangle gets set. Does not use
            graphics card.
        bounding_box: 6-element float list/tuple of min, max values,
            (minx, miny, minz, maxx, maxy, maxz)
        remove_internal: remove internal voxels if True. Note there is some odd
            behaviour if boundary voxels are occupied.
        center: center model inside unit cube.
        rotate_x: number of 90 degree ccw rotations around x-axis before
            voxelizing.
        rotate_z: number of 90 degree cw rotations around z-axis before
            voxelizing.
        wireframe: also render the model in wireframe (helps with thin parts).
        fit: only write voxels in the voxel bounding box.
        block_id: when converting to schematic, use this as the block ID.
        use_matrial_block_id: when converting from obj to schematic, parse
            block ID from material spec "usemtl blockid_<id>" (ids 1-255 only).
        use_offscreen_pbuffer: use offscreen pbuffer instead of onscreen
            window.
        downsample_factor: downsample voxels by this factor in each dimension.
            Must be a power of 2 or None. If not None/1 and `core dumped`
            errors occur, try slightly adjusting dimensions.
        downsample_threshold: when downsampling, destination voxel is on if
            more than this number of voxels are on.
        verbose : bool
          If False, silences stdout/stderr from subprocess call.
        binvox_path : str
          Path to binvox executable. The default looks for an
          executable called `binvox` on your `PATH`.
        """
        if binvox_path is None:
            encoder = binvox_encoder
        else:
            encoder = binvox_path

        if encoder is None:
            raise IOError(
                " ".join(
                    [
                        "No `binvox_path` provided and no binvox executable found",
                        "on PATH, please go to https://www.patrickmin.com/binvox/ and",
                        "download the appropriate version.",
                    ]
                )
            )

        if dimension > 1024 and not exact:
            raise ValueError("Maximum dimension using exact is 1024, got %d" % dimension)
        if file_type not in Binvoxer.SUPPORTED_OUTPUT_TYPES:
            raise ValueError(
                "file_type %s not in set of supported output types %s"
                % (file_type, str(Binvoxer.SUPPORTED_OUTPUT_TYPES))
            )
        args = [encoder, "-d", str(dimension), "-t", file_type]
        if exact:
            args.append("-e")
        if z_buffer_carving:
            if z_buffer_voting:
                pass
            else:
                args.append("-c")
        elif z_buffer_voting:
            args.append("-v")
        else:
            raise ValueError("One of `z_buffer_carving` or `z_buffer_voting` must be True")
        if dilated_carving:
            args.append("-dc")

        # Additional parameters
        if bounding_box is not None:
            if len(bounding_box) != 6:
                raise ValueError("bounding_box must have 6 elements")
            args.append("-bb")
            args.extend(str(b) for b in bounding_box)
        if remove_internal:
            args.append("-ri")
        if center:
            args.append("-cb")
        args.extend(("-rotx",) * rotate_x)
        args.extend(("-rotz",) * rotate_z)
        if wireframe:
            args.append("-aw")
        if fit:
            args.append("-fit")
        if block_id is not None:
            args.extend(("-bi", block_id))
        if use_material_block_id:
            args.append("-mb")
        if use_offscreen_pbuffer:
            args.append("-pb")
        if downsample_factor is not None:
            times = np.log2(downsample_factor)
            if int(times) != times:
                raise ValueError("downsample_factor must be a power of 2, got %d" % downsample_factor)
            args.extend(("-down",) * int(times))
        if downsample_threshold is not None:
            args.extend(("-dmin", str(downsample_threshold)))
        args.append("PATH")
        self._args = args
        self._file_type = file_type

        self.verbose = verbose

    @property
    def file_type(self):
        return self._file_type

    def __call__(self, path, overwrite=False):
        """
        Create an voxel file in the same directory as model at `path`.

        Parameters
        ------------
        path: string path to model file. Supported types:
            'ug'
            'obj'
            'off'
            'dfx'
            'xgl'
            'pov'
            'brep'
            'ply'
            'jot' (polygongs only)
        overwrite: if False, checks the output path (head.file_type) is empty
            before running. If True and a file exists, raises an IOError.

        Returns
        ------------
        string path to voxel file. File type give by file_type in constructor.
        """
        head, ext = os.path.splitext(path)
        ext = ext[1:].lower()
        if ext not in Binvoxer.SUPPORTED_INPUT_TYPES:
            raise ValueError(
                "file_type %s not in set of supported input types %s" % (ext, str(Binvoxer.SUPPORTED_INPUT_TYPES))
            )
        out_path = "%s.%s" % (head, self._file_type)
        if os.path.isfile(out_path) and not overwrite:
            raise IOError("Attempted to voxelize object at existing path")
        self._args[-1] = path

        # generalizes to python2 and python3
        # will capture terminal output into variable rather than printing
        verbosity = subprocess.check_output(self._args)

        # if requested print ourselves
        if self.verbose:
            util.log.debug(verbosity)

        return out_path


def voxelize_mesh(mesh, binvoxer=None, export_type="off", **binvoxer_kwargs):
    """
    Interface for voxelizing Trimesh object via the binvox tool.

    Implementation simply saved the mesh in the specified export_type then
    runs the `Binvoxer.__call__` (using either the supplied `binvoxer` or
    creating one via `binvoxer_kwargs`)

    Parameters
    ------------
    mesh: Trimesh object to voxelize.
    binvoxer: optional Binvoxer instance.
    export_type: file type to export mesh as temporarily for Binvoxer to
        operate on.
    **binvoxer_kwargs: kwargs for creating a new Binvoxer instance. If binvoxer
        if provided, this must be empty.

    Returns
    ------------
    `VoxelGrid` object resulting.
    """
    if not isinstance(mesh, Trimesh):
        raise ValueError("mesh must be Trimesh instance, got %s" % str(mesh))
    if binvoxer is None:
        binvoxer = Binvoxer(**binvoxer_kwargs)
    elif len(binvoxer_kwargs) > 0:
        raise ValueError("Cannot provide binvoxer and binvoxer_kwargs")
    if binvoxer.file_type != "binvox":
        raise ValueError('Only "binvox" binvoxer `file_type` currently supported')
    with util.TemporaryDirectory() as folder:
        model_path = os.path.join(folder, "model.%s" % export_type)
        with open(model_path, "wb") as fp:
            mesh.export(fp, file_type=export_type)
        out_path = binvoxer(model_path)
        with open(out_path, "rb") as fp:
            out_model = load_binvox(fp)
    return out_model


_binvox_loaders = {"binvox": load_binvox}
