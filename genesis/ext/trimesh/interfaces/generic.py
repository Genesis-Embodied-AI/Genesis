import os
import platform
import subprocess

from string import Template
from tempfile import NamedTemporaryFile
from subprocess import check_output, CalledProcessError

from .. import exchange
from ..util import log


class MeshScript:

    def __init__(self, meshes, script, exchange="stl", debug=False, **kwargs):

        self.debug = debug
        self.kwargs = kwargs
        self.meshes = meshes
        self.script = script
        self.exchange = exchange

    def __enter__(self):
        # windows has problems with multiple programs using open files so we close
        # them at the end of the enter call, and delete them ourselves at exit
        # Blender sorts its objects alphabetically
        # so prefix the mesh number on the file name
        digit_count = len(str(len(self.meshes)))
        self.mesh_pre = [
            NamedTemporaryFile(
                suffix=".{}".format(self.exchange),
                prefix="{}_".format(str(i).zfill(digit_count)),
                mode="wb",
                delete=False,
            )
            for i in range(len(self.meshes))
        ]
        self.mesh_post = NamedTemporaryFile(suffix=".{}".format(self.exchange), mode="rb", delete=False)
        self.script_out = NamedTemporaryFile(mode="wb", delete=False)

        # export the meshes to a temporary STL container
        for mesh, file_obj in zip(self.meshes, self.mesh_pre):
            mesh.export(file_obj=file_obj.name)

        self.replacement = {"MESH_" + str(i): m.name for i, m in enumerate(self.mesh_pre)}
        self.replacement["MESH_PRE"] = str([i.name for i in self.mesh_pre])
        self.replacement["MESH_POST"] = self.mesh_post.name
        self.replacement["SCRIPT"] = self.script_out.name

        script_text = Template(self.script).substitute(self.replacement)
        if platform.system() == "Windows":
            script_text = script_text.replace("\\", "\\\\")
        self.script_out.write(script_text.encode("utf-8"))

        # close all temporary files
        self.script_out.close()
        self.mesh_post.close()
        for file_obj in self.mesh_pre:
            file_obj.close()
        return self

    def run(self, command):
        command_run = Template(command).substitute(self.replacement).split()
        # run the binary
        startupinfo = None
        if platform.system() == "Windows":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        if self.debug:
            log.info("executing: {}".format(" ".join(command_run)))

        try:
            output = check_output(command_run, stderr=subprocess.STDOUT, startupinfo=startupinfo)
        except CalledProcessError as e:
            # Log output if debug is enabled
            if self.debug:
                log.info(e.output.decode())
            raise

        if self.debug:
            log.info(output.decode())

        # bring the binaries result back as a set of Trimesh kwargs
        mesh_results = exchange.load.load_mesh(self.mesh_post.name, **self.kwargs)

        return mesh_results

    def __exit__(self, *args, **kwargs):
        if self.debug:
            log.info("MeshScript.debug: not deleting {}".format(self.script_out.name))
            return
        # delete all the temporary files by name
        # they are closed but their names are still available
        os.remove(self.script_out.name)
        for file_obj in self.mesh_pre:
            os.remove(file_obj.name)
        os.remove(self.mesh_post.name)
