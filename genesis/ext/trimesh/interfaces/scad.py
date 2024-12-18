import os
import platform
from subprocess import CalledProcessError

from ..util import which
from .generic import MeshScript
from ..constants import log

# start the search with the user's PATH
_search_path = os.environ.get("PATH", "")
# add additional search locations on windows
if platform.system() == "Windows":
    # split existing path by delimiter
    _search_path = [i for i in _search_path.split(";") if len(i) > 0]
    _search_path.append(os.path.normpath(r"C:\Program Files\OpenSCAD"))
    _search_path.append(os.path.normpath(r"C:\Program Files (x86)\OpenSCAD"))
    _search_path = ";".join(_search_path)
    log.debug("searching for scad in: %s", _search_path)
# add mac-specific search locations
if platform.system() == "Darwin":
    _search_path = [i for i in _search_path.split(":") if len(i) > 0]
    _search_path.append("/Applications/OpenSCAD.app/Contents/MacOS")
    _search_path = ":".join(_search_path)
    log.debug("searching for scad in: %s", _search_path)
# try to find the SCAD executable by name
_scad_executable = which("openscad", path=_search_path)
if _scad_executable is None:
    _scad_executable = which("OpenSCAD", path=_search_path)
exists = _scad_executable is not None


def interface_scad(meshes, script, debug=False, **kwargs):
    """
    A way to interface with openSCAD which is itself an interface
    to the CGAL CSG bindings.
    CGAL is very stable if difficult to install/use, so this function provides a
    tempfile- happy solution for getting the basic CGAL CSG functionality.

    Parameters
    ---------
    meshes: list of Trimesh objects
    script: string of the script to send to scad.
            Trimesh objects can be referenced in the script as
            $mesh_0, $mesh_1, etc.
    """
    if not exists:
        raise ValueError("No SCAD available!")
    # OFF is a simple text format that references vertices by-index
    # making it slightly preferable to STL for this kind of exchange duty
    try:
        with MeshScript(meshes=meshes, script=script, debug=debug, exchange="off") as scad:
            result = scad.run(_scad_executable + " $SCRIPT -o $MESH_POST")
    except CalledProcessError as e:
        # Check if scad is complaining about an empty top level geometry.
        # If so, just return an empty Trimesh object.
        if "Current top level object is empty." in e.output.decode():
            from .. import Trimesh

            return Trimesh()
        else:
            raise

    return result


def boolean(meshes, operation="difference", debug=False, **kwargs):
    """
    Run an operation on a set of meshes
    """
    script = operation + "(){"
    for i in range(len(meshes)):
        script += 'import("$MESH_' + str(i) + '");'
    script += "}"
    return interface_scad(meshes, script, debug=debug, **kwargs)
