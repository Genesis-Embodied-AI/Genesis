import os
import pickle as pkl
import subprocess
import sys
import tempfile

import igl
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import genesis as gs
from genesis.ext import trimesh

from . import geom as gu
from . import mesh as msu
from . import misc as miu

# misc operations for external binary
# ParticleMesherPy
try:
    LD_LIBRARY_PATH = os.path.join(miu.get_src_dir(), "ext/ParticleMesher/ParticleMesherPy")
    sys.path.append(LD_LIBRARY_PATH)
    cur_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{cur_ld_library_path}:{LD_LIBRARY_PATH}"
    import ParticleMesherPy

    is_ParticleMesherPy_available = True
except Exception as e:
    ParticleMesherPy_error_msg = f"{e.__class__.__name__}: {e}"
    is_ParticleMesherPy_available = False

# splashsurf
try:
    subprocess.run(["splashsurf"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    is_splashsurf_available = True
except Exception as e:
    splashsurf_error_msg = f"{e.__class__.__name__}: {e}"
    is_splashsurf_available = False


def n_particles_vol(p_size=0.01, volume=1.0):
    return max(1, round(volume / p_size**3))


def n_particles_3D(p_size=0.01, size=(1.0, 1.0, 1.0)):
    return max(1, round(size[0] / p_size)) * max(1, round(size[1] / p_size)) * max(1, round(size[2] / p_size))


def n_particles_1D(p_size=0.01, length=1.0):
    return max(1, round(length / p_size))


def nowhere_particles(n):
    positions = np.tile(gu.nowhere(), [n, 1])
    return positions


def trimesh_to_particles_simple(mesh, p_size, sampler):
    """
    Mesh to particles via `random` or `regular` sampler.
    """
    assert sampler in ("random", "regular")

    # compute file name via hashing for caching
    ptc_file_path = msu.get_ptc_path(mesh.vertices, mesh.faces, p_size, sampler)

    # loading pre-computed cache if available
    is_cached_loaded = False
    if os.path.exists(ptc_file_path):
        gs.logger.debug("Sampled particles file (`.ptc`) found in cache.")
        try:
            with open(ptc_file_path, "rb") as file:
                positions = pkl.load(file)
            is_cached_loaded = True
        except (EOFError, pkl.UnpicklingError):
            gs.logger.info("Ignoring corrupted cache.")

    if not is_cached_loaded:
        with gs.logger.timer(f"Sampling particles with ~<{sampler}>~ sampler and generating `.ptc` file:"):
            # sample a cube first
            box_size = mesh.bounds[1] - mesh.bounds[0]
            box_center = (mesh.bounds[1] + mesh.bounds[0]) / 2
            positions = _box_to_particles(p_size=p_size, pos=box_center, size=box_size, sampler=sampler)
            # reject out-of-boundary particles
            positions = positions[igl.signed_distance(positions, mesh.vertices, mesh.faces)[0] < 0]

            os.makedirs(os.path.dirname(ptc_file_path), exist_ok=True)
            with open(ptc_file_path, "wb") as file:
                pkl.dump(positions, file)

    return positions


def trimesh_to_particles_pbs(mesh, p_size, sampler, pos=(0, 0, 0)):
    """
    Physics-based particle sampler using the method proposed by Kugelstadt et al. [2021].
    References: https://splishsplash.readthedocs.io/en/latest/VolumeSampling.html
    If this sampler fails, it returns `None`.
    """
    assert "pbs" in sampler

    # compute file name via hashing for caching
    ptc_file_path = msu.get_ptc_path(mesh.vertices, mesh.faces, p_size, sampler)

    # loading pre-computed cache if available
    is_cached_loaded = False
    if os.path.exists(ptc_file_path):
        gs.logger.debug("Sampled particles file (`.ptc`) found in cache.")
        try:
            with open(ptc_file_path, "rb") as file:
                positions = pkl.load(file)
            is_cached_loaded = True
        except (EOFError, pkl.UnpicklingError):
            gs.logger.info("Ignoring corrupted cache.")

    if not is_cached_loaded:
        with gs.logger.timer(f"Sampling particles with ~<{sampler}>~ sampler and generating `.ptc` file:"):
            sdf_res = int(sampler.split("-")[-1])

            # We scale up a bit the particle size because this method tends to sample denser particles compared to `random` and `regular` samplers.
            scale = 1.104  # Magic number from Pingchuan Ma.
            particle_radius = p_size * scale / 2

            _, tmp_mesh_path = tempfile.mkstemp()
            _, tmp_vtk_path = tempfile.mkstemp()
            tmp_mesh_path += ".obj"
            tmp_vtk_path += ".vtk"
            mesh.export(tmp_mesh_path)

            # Sample particles
            steps = 100  # use bigger value leads to smoother surface
            command = (
                os.path.join(miu.get_src_dir(), "ext/VolumeSampling")
                + f" -i {tmp_mesh_path} -o {tmp_vtk_path} --no-cache --mode 4 -r {particle_radius:.6f} --res {sdf_res},{sdf_res},{sdf_res} --steps {steps}"
            )
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, _ = process.communicate()
            if not os.path.isfile(tmp_vtk_path):
                return None

            # Print sampler output
            stdout_str = stdout.decode("utf-8")
            gs.logger.debug(stdout_str)

            # Read the generated VTK file
            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(tmp_vtk_path)
            reader.Update()
            positions = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())

            # Clean up the intermediate files
            output_dir = os.path.join(miu.get_src_dir(), "ext/output")
            if os.path.exists(output_dir):
                process = subprocess.Popen(
                    f"rm -rf {output_dir}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                process.communicate()

            # Cache the generated positions
            os.makedirs(os.path.dirname(ptc_file_path), exist_ok=True)
            with open(ptc_file_path, "wb") as file:
                pkl.dump(positions, file)

    positions += np.array(pos)

    return positions


def _box_to_particles(p_size, pos, size, sampler):
    """
    Private function to sample particles from a box. This function only supports `random` and `regular` samplers.
    This is a private function that does not consider additional mesh offset or scale.
    """
    size = np.array(size)
    pos = np.array(pos)
    lower = pos - size / 2
    upper = pos + size / 2

    if sampler == "random":
        n_particles = n_particles_3D(p_size, size)
        positions = np.random.uniform(low=lower, high=upper, size=(n_particles, 3))

    elif sampler == "regular":
        n_x = n_particles_1D(p_size, size[0])
        n_y = n_particles_1D(p_size, size[1])
        n_z = n_particles_1D(p_size, size[2])
        p_lower = lower + 0.5 * p_size
        p_upper = upper - 0.5 * p_size
        x = np.linspace(p_lower[0], p_upper[0], n_x)
        y = np.linspace(p_lower[1], p_upper[1], n_y)
        z = np.linspace(p_lower[2], p_upper[2], n_z)
        positions = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1).reshape((-1, 3))

    else:
        gs.raise_exception(f"Unsupported sampler method: {sampler}.")

    return positions


def box_to_particles(p_size=0.01, pos=(0, 0, 0), size=(1, 1, 1), sampler="random"):
    if "pbs" in sampler:
        mesh = trimesh.creation.box(extents=size)

        positions = trimesh_to_particles_pbs(mesh, p_size, sampler, pos=pos)
        if positions is None:
            gs.logger.warning("`pbs` sampler failed. Falling back to `random` sampler.")
            sampler = "random"

    if sampler in ["random", "regular"]:
        positions = _box_to_particles(
            p_size=p_size,
            pos=pos,
            size=size,
            sampler=sampler,
        )

    return positions


def cylinder_to_particles(p_size=0.01, pos=(0, 0, 0), radius=0.5, height=1.0, sampler="random"):
    if "pbs" in sampler:
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
        positions = trimesh_to_particles_pbs(mesh, p_size, sampler, pos=pos)
        if positions is None:
            gs.logger.warning("`pbs` sampler failed. Falling back to `random` sampler.")
            sampler = "random"

    if sampler in ["random", "regular"]:
        # sample a cube first
        size = np.array([2 * radius, 2 * radius, height])
        positions = _box_to_particles(
            p_size=p_size,
            pos=pos,
            size=size,
            sampler=sampler,
        )
        # reject out-of-boundary particles
        positions_r = np.linalg.norm(positions[:, [0, 1]] - np.array(pos)[[0, 1]], axis=1)
        positions = positions[positions_r <= radius]

    return positions


def sphere_to_particles(p_size=0.01, pos=(0, 0, 0), radius=0.5, sampler="random"):
    if "pbs" in sampler:
        mesh = trimesh.creation.icosphere(radius=radius)
        positions = trimesh_to_particles_pbs(mesh, p_size, sampler, pos=pos)
        if positions is None:
            gs.logger.warning("`pbs` sampler failed. Falling back to `random` sampler.")
            sampler = "random"

    if sampler in ["random", "regular"]:
        # sample a cube first
        size = np.array([2 * radius, 2 * radius, 2 * radius])
        positions = _box_to_particles(
            p_size=p_size,
            pos=pos,
            size=size,
            sampler=sampler,
        )
        # reject out-of-boundary particles
        positions_r = np.linalg.norm(positions - np.array(pos), axis=1)
        positions = positions[positions_r <= radius]

    return positions


def shell_to_particles(p_size=0.01, pos=(0, 0, 0), inner_radius=0.5, outer_radius=0.7, sampler="random"):
    if "pbs" in sampler:
        mesh = trimesh.creation.icosphere(radius=outer_radius)
        positions = trimesh_to_particles_pbs(mesh, p_size, sampler, pos=pos)
        if positions is None:
            gs.logger.warning("`pbs` sampler failed. Falling back to `random` sampler.")
            sampler = "random"

    if sampler in ["random", "regular"]:
        # sample a cube first
        size = np.array([2 * outer_radius, 2 * outer_radius, 2 * outer_radius])
        positions = _box_to_particles(
            p_size=p_size,
            pos=pos,
            size=size,
            sampler=sampler,
        )
        # reject out-of-boundary particles
        positions_r = np.linalg.norm(positions - np.array(pos), axis=1)
        positions = positions[positions_r <= outer_radius]

    # reject inner particles
    positions_r = np.linalg.norm(positions - np.array(pos), axis=1)
    positions = positions[positions_r >= inner_radius]

    return positions


def particles_to_mesh(positions, radius, backend):
    if positions.shape[0] == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))

    if isinstance(radius, np.ndarray):
        radii = radius
        radius = np.min(radius)
        if "splashsurf" in backend:
            gs.logger.warning("Cannot use variable radius for splashsurf. Fall back to openvdb.")
            backend = "openvdb"
    else:
        radii = np.array([])

    if backend == "openvdb":
        radius_scale = 2.0
        if not is_ParticleMesherPy_available:
            gs.raise_exception(f"Failed to import ParticleMesher. {ParticleMesherPy_error_msg}")

        reconstructor = ParticleMesherPy.MeshConstructor(
            ParticleMesherPy.MeshConstructorConfig(
                particle_radius=radius * radius_scale,
                voxel_scale=1.0,
                isovalue=0.0,
                adaptivity=0.01,
            )
        )
        mesh = reconstructor.construct(positions=positions, radii=radii * radius_scale)
        gs.logger.debug(f"[ParticleMehser]: {mesh.info_msg}")
        vertices = mesh.vertices.reshape([-1, 3])
        faces = mesh.triangles.reshape([-1, 3])

        return trimesh.Trimesh(vertices, faces, process=False)

    elif "splashsurf" in backend:
        if not is_splashsurf_available:
            gs.raise_exception(f"Failed to import splashsurf. {splashsurf_error_msg}")

        _, tmp_xyz_path = tempfile.mkstemp()
        os.close(_)
        tmp_xyz_path += ".xyz"

        _, tmp_obj_path = tempfile.mkstemp()
        os.close(_)
        tmp_obj_path += ".obj"

        positions.astype(np.float32).tofile(tmp_xyz_path)

        # suggested value is 1.4-1.6. 1.0 seems more detailed?
        if len(backend.split("-")) >= 2:
            r = radius * float(backend.split("-")[1])
        else:
            r = radius * 1.0

        if "smooth" in backend:
            smooth_iter = int(backend.split("-")[-1])
            command = f"splashsurf reconstruct {tmp_xyz_path} -r={r:.5f} -c=0.8 -l=2.0 -t=0.6 --subdomain-grid=on -o {tmp_obj_path} --mesh-cleanup=on --mesh-smoothing-weights=on --mesh-smoothing-iters={smooth_iter} --normals=on --normals-smoothing-iters=10"
        else:
            command = f"splashsurf reconstruct {tmp_xyz_path} -r={r:.5f} -c=0.8 -l=2.0 -t=0.6 --subdomain-grid=on -o {tmp_obj_path}"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate()

        if not os.path.isfile(tmp_obj_path):
            gs.raise_exception("Surface reconstruction failed.")

        # gs.logger.debug(stdout.decode('utf-8'))

        mesh = trimesh.load_mesh(tmp_obj_path)
        gs.logger.debug(f"[splashsurf]: reconstruct vertices: {mesh.vertices.shape}, {mesh.faces.shape}")
        return mesh

    else:
        gs.raise_exception(f"Unsupported backend: {backend}.")


def init_foam_generator(
    object_id,
    particle_radius,
    time_step,
    gravity,
    lower_bound,
    upper_bound,
    spray_decay,
    foam_decay,
    bubble_decay,
    k_foam,
    foam_density,
):
    if not is_ParticleMesherPy_available:
        gs.raise_exception(f"Failed to import ParticleMesher. {ParticleMesherPy_error_msg}")

    min_ke = 0.1 * (particle_radius**3 * 6400) * (2.5**2)
    return ParticleMesherPy.FoamGenerator(
        config=ParticleMesherPy.FoamGeneratorConfig(
            particle_radius=particle_radius,
            voxel_scale=1.0,
            time_step=time_step,
            lower_bound=tuple(lower_bound),
            upper_bound=tuple(upper_bound),
            gravity=tuple(gravity),
            lim_ta=(0.05, 0.5),
            lim_wc=(0.05, 0.5),
            lim_ke=(min_ke, min_ke * 50),
            generate_neighbor_min=5,
            k_foam=k_foam,
            k_ad=0.90,
            spray_decay=spray_decay,
            foam_decay=foam_decay,
            bubble_decay=bubble_decay,
            foam_density=foam_density,
        ),
        object_id=object_id,
    )


def generate_foam_particles(generator, positions, velocities):
    foams = generator.generate_foams(
        positions=positions,
        velocities=velocities,
    )
    gs.logger.debug(f"[ParticleMesher]: {foams.info_msg}")
    return foams.positions.reshape([-1, 3])


def filter_surface(positions, radii, particle_radius, half_width=8.0, radius_scale=1.0):
    if not is_ParticleMesherPy_available:
        gs.raise_exception(f"Failed to import ParticleMesher. {ParticleMesherPy_error_msg}")

    splitter = ParticleMesherPy.SurfaceSplitter(
        ParticleMesherPy.SurfaceSplitterConfig(
            particle_radius=particle_radius * radius_scale,
            voxel_scale=0.25,
            support_scale=4.0,
            half_width=half_width,
            surface_neighbor_max=20,
        )
    )

    # surface_indices = splitter.split_surface_sdf(positions, radii * radius_scale)
    surface_indices = splitter.split_surface_count(positions)
    gs.logger.debug(
        f"[ParticleMesher]: {surface_indices.info_msg}\n"
        f"\tFrom {positions.shape[0]} to {np.sum(surface_indices.is_surface)}"
    )
    return surface_indices.is_surface
