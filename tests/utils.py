import platform
import io
import os
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from itertools import chain
from pathlib import Path
from typing import Literal, Sequence

import cpuinfo
import numpy as np
import mujoco
import torch
from huggingface_hub import snapshot_download
from PIL import Image, UnidentifiedImageError
from requests.exceptions import HTTPError

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import mjcf as mju
from genesis.utils.mesh import get_assets_dir
from genesis.utils.misc import tensor_to_array
from genesis.options.morphs import URDF_FORMAT, MJCF_FORMAT, MESH_FORMATS, GLTF_FORMATS, USD_FORMATS


REPOSITY_URL = "Genesis-Embodied-AI/Genesis"
DEFAULT_BRANCH_NAME = "main"

HUGGINGFACE_ASSETS_REVISION = "16e4eae0024312b84518f4b555dd630d6b34095a"
HUGGINGFACE_SNAPSHOT_REVISION = "15e836c732972cd8ddf57e43136986b34653b279"

MESH_EXTENSIONS = (".mtl", *MESH_FORMATS, *GLTF_FORMATS, *USD_FORMATS)
IMAGE_EXTENSIONS = (".png", ".jpg")


# Get repository "root" path (actually test dir is good enough)
TEST_DIR = os.path.dirname(__file__)


@dataclass
class MjSim:
    model: mujoco.MjModel
    data: mujoco.MjData


@cache
def get_hardware_fingerprint(include_gpu=True):
    # CPU info
    cpu_info = cpuinfo.get_cpu_info()
    infos = [
        cpu_info.get("brand_raw", cpu_info.get("hardware_raw")),
        cpu_info.get("arch"),
    ]

    # GPU info
    if include_gpu and torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        infos += [
            props.name,
            ".".join(map(str, (props.major, props.minor))),
            props.total_memory,
            props.multi_processor_count,  # Number of "streaming multiprocessors"
        ]

    return "-".join(map(str, filter(None, infos)))


@cache
def get_platform_fingerprint():
    # OS distribution info
    system = platform.system()
    dist_name = None
    if system == "Linux":
        try:
            dist_info = platform.freedesktop_os_release()
            dist_name = dist_info["ID"]
            dist_ver = dist_info["VERSION_ID"]
        except FileNotFoundError:
            pass
    elif system == "Darwin":
        dist_name = "MacOS"
        dist_ver, *_ = platform.mac_ver()
    if dist_name is None:
        dist_name = system
        dist_ver, *_ = platform.release().split(".", 1)  # Only extract major version.

    infos = [
        dist_name,
        dist_ver,  # Only extract major version.
    ]

    # Python info
    py_major, py_minor, py_patchlevel = platform.python_version_tuple()
    infos += [
        ".".join((py_major, py_minor)),  # Ignore patch-level version
    ]

    return "-".join(map(str, filter(None, infos)))


@cache
def get_git_commit_timestamp(ref="HEAD"):
    try:
        contrib_date = subprocess.check_output(
            ["git", "show", "-s", "--quiet", "--format=%ci", ref], cwd=TEST_DIR, encoding="utf-8"
        ).strip()
    except subprocess.CalledProcessError:
        # Commit not found, either because it does not exist or becaused fo shallow git clone
        return float("nan")

    try:
        date = datetime.fromisoformat(contrib_date)
    except ValueError:
        date = datetime.strptime(contrib_date, "%Y-%m-%d %H:%M:%S %z")
    timestamp = date.timestamp()

    return timestamp


@cache
def get_git_commit_info(ref="HEAD"):
    # Fetch current commit revision
    try:
        revision = subprocess.check_output(["git", "rev-parse", ref], cwd=TEST_DIR, encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        revision = f"{uuid.uuid4().hex}@UNKNOWN"
        timestamp = float("nan")
        return revision, timestamp

    # Fetch all remote branches containing the current commit
    try:
        branches = subprocess.check_output(
            ["git", "branch", "--remote", "--contains", ref], cwd=TEST_DIR, encoding="utf-8"
        ).splitlines()
    except subprocess.CalledProcessError:
        # Raise error if not found neither locally nor remotely
        branches = ()

    # Check if the current commit is contained by main branch
    remote_handle = "UNKNOWN"
    for branch in branches:
        try:
            remote_name, branch_name = branch.strip().split("/", 1)
        except ValueError:
            continue
        if branch_name != DEFAULT_BRANCH_NAME:
            continue
        remote_url = subprocess.check_output(
            ["git", "remote", "get-url", remote_name], cwd=TEST_DIR, encoding="utf-8"
        ).strip()
        if remote_url.startswith("https://github.com/"):
            remote_handle = remote_url[19:-4]
        elif remote_url.startswith("git@github.com:"):
            remote_handle = remote_url[15:-4]
        if remote_handle == REPOSITY_URL:
            is_commit_on_default_branch = True
            break
    else:
        is_commit_on_default_branch = False

    # Return the contribution date as timestamp if and only if the HEAD commit is contained on main branch
    if is_commit_on_default_branch:
        timestamp = get_git_commit_timestamp(ref)
        return revision, timestamp

    revision = f"{revision}@{remote_handle}"
    timestamp = float("nan")
    return revision, timestamp


def get_hf_dataset(
    pattern,
    repo_name: str = "assets",
    local_dir: str | None = None,
    num_retry: int = 4,
    retry_delay: float = 30.0,
    local_dir_use_symlinks: bool = True,
):
    assert num_retry >= 1

    if repo_name == "assets":
        revision = HUGGINGFACE_ASSETS_REVISION
    elif repo_name == "snapshots":
        revision = HUGGINGFACE_SNAPSHOT_REVISION
    else:
        raise ValueError(f"Unsupported repository '{repo_name}'")

    for i in range(num_retry):
        try:
            # Try downloading the assets
            asset_path = snapshot_download(
                repo_type="dataset",
                repo_id=f"Genesis-Intelligence/{repo_name}",
                revision=revision,
                allow_patterns=pattern,
                max_workers=1,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
            )

            # Make sure that download was successful
            has_files = False
            for path in Path(asset_path).glob(pattern):
                if not path.is_file():
                    continue

                ext = path.suffix.lower()
                if ext not in (URDF_FORMAT, MJCF_FORMAT, *IMAGE_EXTENSIONS, *MESH_EXTENSIONS):
                    continue

                has_files = True

                if path.stat().st_size == 0:
                    raise HTTPError(f"File '{path}' is empty.")

                if path.suffix.lower() in (URDF_FORMAT, MJCF_FORMAT):
                    try:
                        ET.parse(path)
                    except ET.ParseError as e:
                        raise HTTPError("Impossible to parse XML file.") from e
                elif path.suffix.lower() in IMAGE_EXTENSIONS:
                    try:
                        Image.open(path)
                    except UnidentifiedImageError as e:
                        raise HTTPError("Impossible to parse Image file.") from e
                elif path.suffix.lower() in MESH_EXTENSIONS:
                    # TODO: Validating mesh files is more tricky. Ignoring them for now.
                    pass

            if not has_files:
                raise HTTPError("No file downloaded.")
        except (HTTPError, FileNotFoundError):
            if i == num_retry - 1:
                raise
            print(f"Failed to download assets from HuggingFace dataset. Trying again in {retry_delay}s...")
            time.sleep(retry_delay)
        else:
            break

    return asset_path


def assert_allclose(actual, desired, *, atol=None, rtol=None, tol=None, err_msg=None):
    # Determine absolute and relative tolerance from input arguments
    assert (tol is not None) ^ (atol is not None or rtol is not None)
    if tol is not None:
        atol = tol
        rtol = tol
    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0

    # Convert input arguments as numpy arrays
    args = [actual, desired]
    for i, arg in enumerate(args):
        if isinstance(arg, (tuple, list)):
            arg = np.stack([tensor_to_array(val) for val in arg], axis=0)
        args[i] = tensor_to_array(arg)

    # Early return without checking anything is both arrays are empty (0D arrays have size 1).
    if all(e.size == 0 for e in args):
        return

    # Try to make sure both arrays have the exact same shape.
    # First, try to broadcast both matrices. Then it is does not work, squeeze them before trying again.
    try:
        args = np.broadcast_arrays(*args)
    except ValueError:
        args = np.broadcast_arrays(*map(np.squeeze, args))

    np.testing.assert_allclose(*args, atol=atol, rtol=rtol, err_msg=err_msg)


def assert_array_equal(actual, desired, *, err_msg=None):
    assert_allclose(actual, desired, atol=0.0, rtol=0.0, err_msg=err_msg)


def init_simulators(gs_sim, mj_sim=None, qpos=None, qvel=None):
    if mj_sim is not None:
        _, (_, _, mj_qs_idx, mj_dofs_idx, _, _) = _get_model_mappings(gs_sim, mj_sim)

    (gs_robot,) = gs_sim.entities

    gs_sim.scene.reset()
    if qpos is not None:
        gs_robot.set_qpos(qpos)
    if qvel is not None:
        gs_robot.set_dofs_velocity(qvel)
    # TODO: This should be moved in `set_state`, `set_qpos`, `set_dofs_position`, `set_dofs_velocity`
    gs_sim.rigid_solver.dofs_state.qf_constraint.fill(0.0)
    gs_sim.rigid_solver._func_forward_dynamics()
    gs_sim.rigid_solver._func_constraint_force()
    gs_sim.rigid_solver._func_update_acc()

    if gs_sim.scene.visualizer:
        gs_sim.scene.visualizer.update()

    if mj_sim is not None:
        mujoco.mj_resetData(mj_sim.model, mj_sim.data)
        mj_sim.data.qpos[mj_qs_idx] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[mj_dofs_idx] = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
        mujoco.mj_forward(mj_sim.model, mj_sim.data)


def _gs_search_by_joints_name(
    scene,
    joints_name: str | list[str],
    to: Literal["entity", "index"] = "index",
    is_local: bool = False,
    flatten: bool = True,
):
    if isinstance(joints_name, str):
        joints_name = [joints_name]

    for entity in scene.entities:
        try:
            gs_joints_idx = dict()
            gs_joints_qs_idx = dict()
            gs_joints_dofs_idx = dict()
            valid_joints_name = []
            for joint in entity.joints:
                valid_joints_name.append(joint.name)
                if joint.name in joints_name:
                    if to == "entity":
                        gs_joints_idx[joint.name] = joint
                        gs_joints_qs_idx[joint.name] = joint
                        gs_joints_dofs_idx[joint.name] = joint
                    elif to == "index":
                        gs_joints_idx[joint.name] = joint.idx_local if is_local else joint.idx
                        gs_joints_qs_idx[joint.name] = joint.qs_idx_local if is_local else joint.qs_idx
                        gs_joints_dofs_idx[joint.name] = joint.dofs_idx_local if is_local else joint.dofs_idx
                    else:
                        raise ValueError(f"Cannot recognize what ({to}) to extract for the search")

            missing_joints_name = set(joints_name) - gs_joints_idx.keys()
            if len(missing_joints_name) > 0:
                raise ValueError(
                    f"Cannot find joints `{missing_joints_name}`. Valid joints names are {valid_joints_name}"
                )

            if flatten:
                return (
                    list(gs_joints_idx.values()),
                    list(chain.from_iterable(gs_joints_qs_idx.values())),
                    list(chain.from_iterable(gs_joints_dofs_idx.values())),
                )
            return (gs_joints_idx, gs_joints_qs_idx, gs_joints_dofs_idx)
        except ValueError:
            pass
    else:
        raise ValueError(f"Fail to find joint indices for {joints_name}")


def _gs_search_by_links_name(
    scene,
    links_name: str | Sequence[str],
    to: Literal["entity", "index"] = "index",
    is_local: bool = False,
    flatten: bool = True,
):
    if isinstance(links_name, str):
        links_name = (links_name,)

    for entity in scene.entities:
        try:
            gs_links_idx = dict()
            valid_links_name = []
            for link in entity.links:
                valid_links_name.append(link.name)
                if link.name in links_name:
                    if to == "entity":
                        gs_links_idx[link.name] = link
                    elif to == "index":
                        gs_links_idx[link.name] = link.idx_local if is_local else link.idx
                    else:
                        raise ValueError(f"Cannot recognize what ({to}) to extract for the search")

            missing_links_name = set(links_name) - gs_links_idx.keys()
            if missing_links_name:
                raise ValueError(f"Cannot find links `{missing_links_name}`. Valid link names are {valid_links_name}")

            if flatten:
                return list(gs_links_idx.values())
            return gs_links_idx
        except ValueError:
            pass
    else:
        raise ValueError(f"Fail to find link indices for {links_name}")


def _get_model_mappings(
    gs_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    bodies_name: list[str] | None = None,
):
    if joints_name is None:
        joints_name = [
            joint.name for entity in gs_sim.entities for joint in entity.joints if joint.type != gs.JOINT_TYPE.FIXED
        ]
    if bodies_name is None:
        bodies_name = [
            body.name
            for entity in gs_sim.entities
            for body in entity.links
            if not (body.is_fixed and body.parent_idx < 0)
        ]

    motors_name: list[str] = []
    mj_joints_idx: list[int] = []
    mj_qs_idx: list[int] = []
    mj_dofs_idx: list[int] = []
    mj_geoms_idx: list[int] = []
    mj_motors_idx: list[int] = []
    for joint_name in joints_name:
        if joint_name:
            mj_joint = mj_sim.model.joint(joint_name)
        else:
            # Must rely on exhaustive search if the joint has empty name
            for j in range(mj_sim.model.njoint):
                mj_joint = mj_sim.model.joint(j)
                if mj_joint.name == "":
                    break
            else:
                raise ValueError(f"Invalid joint name '{joint_name}'.")
        mj_joints_idx.append(mj_joint.id)
        mj_type = mj_sim.model.jnt_type[mj_joint.id]
        if mj_type == mujoco.mjtJoint.mjJNT_HINGE:
            n_dofs, n_qs = 1, 1
        elif mj_type == mujoco.mjtJoint.mjJNT_SLIDE:
            n_dofs, n_qs = 1, 1
        elif mj_type == mujoco.mjtJoint.mjJNT_BALL:
            n_dofs, n_qs = 3, 4
        elif mj_type == mujoco.mjtJoint.mjJNT_FREE:
            n_dofs, n_qs = 6, 7
        else:
            raise ValueError(f"Invalid joint type '{mj_type}'.")
        mj_dof_start_j = mj_sim.model.jnt_dofadr[mj_joint.id]
        mj_dofs_idx += range(mj_dof_start_j, mj_dof_start_j + n_dofs)

        mj_q_start_j = mj_sim.model.jnt_qposadr[mj_joint.id]
        mj_qs_idx += range(mj_q_start_j, mj_q_start_j + n_qs)
        if (mj_joint.id == mj_sim.model.actuator_trnid[:, 0]).any():
            motors_name.append(joint_name)
            (motors_idx,) = np.nonzero(mj_joint.id == mj_sim.model.actuator_trnid[:, 0])
            # FIXME: only supporting 1DoF per actuator
            mj_motors_idx.append(motors_idx[0])

    mj_bodies_idx, mj_geoms_idx = [], []
    for body_name in bodies_name:
        mj_body = mj_sim.model.body(body_name)
        mj_bodies_idx.append(mj_body.id)
        for mj_geom_idx in range(mj_body.geomadr[0], mj_body.geomadr[0] + mj_body.geomnum[0]):
            mj_geom = mj_sim.model.geom(mj_geom_idx)
            if mj_geom.contype or mj_geom.conaffinity:
                mj_geoms_idx.append(mj_geom.id)

    (gs_joints_idx, gs_q_idx, gs_dofs_idx) = _gs_search_by_joints_name(gs_sim.scene, joints_name)
    (_, _, gs_motors_idx) = _gs_search_by_joints_name(gs_sim.scene, motors_name)

    gs_bodies_idx = _gs_search_by_links_name(gs_sim.scene, bodies_name)
    gs_geoms_idx: list[int] = []
    for gs_body_idx in gs_bodies_idx:
        link = gs_sim.rigid_solver.links[gs_body_idx]
        gs_geoms_idx += range(link.geom_start, link.geom_end)

    gs_maps = (gs_bodies_idx, gs_joints_idx, gs_q_idx, gs_dofs_idx, gs_geoms_idx, gs_motors_idx)
    mj_maps = (mj_bodies_idx, mj_joints_idx, mj_qs_idx, mj_dofs_idx, mj_geoms_idx, mj_motors_idx)
    return gs_maps, mj_maps


def build_mujoco_sim(
    xml_path, gs_solver, gs_integrator, merge_fixed_links, multi_contact, adjacent_collision, dof_damping, native_ccd
):
    if gs_solver == gs.constraint_solver.CG:
        mj_solver = mujoco.mjtSolver.mjSOL_CG
    elif gs_solver == gs.constraint_solver.Newton:
        mj_solver = mujoco.mjtSolver.mjSOL_NEWTON
    else:
        raise ValueError(f"Solver '{gs_solver}' not supported")
    if gs_integrator == gs.integrator.Euler:
        mj_integrator = mujoco.mjtIntegrator.mjINT_EULER
    elif gs_integrator == gs.integrator.implicitfast:
        mj_integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    else:
        raise ValueError(f"Integrator '{gs_integrator}' not supported")

    xml_path = os.path.join(get_assets_dir(), xml_path)
    model = mju.build_model(
        xml_path, discard_visual=True, default_armature=None, merge_fixed_links=merge_fixed_links, links_to_keep=()
    )

    model.opt.solver = mj_solver
    model.opt.integrator = mj_integrator
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
    model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    if native_ccd:
        model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    else:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    if multi_contact:
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
    else:
        model.opt.enableflags &= ~np.uint32(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    if adjacent_collision:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
    else:
        model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    data = mujoco.MjData(model)

    return MjSim(model, data)


def build_genesis_sim(
    xml_path,
    gs_solver,
    gs_integrator,
    merge_fixed_links,
    multi_contact,
    mujoco_compatibility,
    adjacent_collision,
    gjk_collision,
    show_viewer,
    mj_sim,
):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            res=(960, 640),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=mj_sim.model.opt.timestep,
            substeps=1,
            gravity=mj_sim.model.opt.gravity.tolist(),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs_integrator,
            constraint_solver=gs_solver,
            enable_mujoco_compatibility=mujoco_compatibility,
            box_box_detection=True,
            enable_self_collision=True,
            enable_adjacent_collision=adjacent_collision,
            enable_multi_contact=multi_contact,
            iterations=mj_sim.model.opt.iterations,
            tolerance=mj_sim.model.opt.tolerance,
            ls_iterations=mj_sim.model.opt.ls_iterations,
            ls_tolerance=mj_sim.model.opt.ls_tolerance,
            use_gjk_collision=gjk_collision,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    morph_kwargs = dict(
        file=xml_path,
        convexify=True,
        decompose_robot_error_threshold=float("inf"),
        default_armature=None,
    )
    if xml_path.endswith(".xml"):
        morph = gs.morphs.MJCF(**morph_kwargs)
    else:
        morph = gs.morphs.URDF(
            fixed=True,
            merge_fixed_links=merge_fixed_links,
            links_to_keep=(),
            **morph_kwargs,
        )
    gs_robot = scene.add_entity(
        morph,
        visualize_contact=True,
    )

    # Force matching Mujoco safety factor for constraint time constant.
    # Note that this time constant affects the penetration depth at rest.
    gs_sim = scene.sim
    gs_sim.rigid_solver._sol_default_timeconst = None
    gs_sim.rigid_solver._sol_min_timeconst = 2.0 * gs_sim._substep_dt

    # Force recomputation of invweights to make sure it works fine
    for link in scene.rigid_solver.links:
        link.invweight[:] = -1
    for joint in scene.rigid_solver.joints:
        joint.dofs_invweight[:] = -1

    scene.build()

    return gs_sim


def check_mujoco_model_consistency(
    gs_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    bodies_name: list[str] | None = None,
    *,
    tol: float,
):
    # Delay import to enable run benchmarks for old Genesis versions that do not have this method
    from genesis.engine.solvers.rigid.rigid_solver_decomp import _sanitize_sol_params

    # Get mapping between Mujoco and Genesis
    gs_maps, mj_maps = _get_model_mappings(gs_sim, mj_sim, joints_name, bodies_name)
    (gs_bodies_idx, gs_joints_idx, gs_q_idx, gs_dofs_idx, gs_geoms_idx, gs_motors_idx) = gs_maps
    (mj_bodies_idx, mj_joints_idx, mj_qs_idx, mj_dofs_idx, mj_geoms_idx, mj_motors_idx) = mj_maps

    # solver
    gs_gravity = gs_sim.rigid_solver.scene.gravity
    mj_gravity = mj_sim.model.opt.gravity
    assert_allclose(gs_gravity, mj_gravity, tol=tol)
    assert mj_sim.model.opt.timestep == gs_sim.rigid_solver.substep_dt
    assert mj_sim.model.opt.tolerance == gs_sim.rigid_solver._options.tolerance
    assert mj_sim.model.opt.iterations == gs_sim.rigid_solver._options.iterations
    assert not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    assert not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    assert not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    assert not (mj_sim.model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_FWDINV)

    mj_adj_collision = bool(mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    gs_adj_collision = gs_sim.rigid_solver._options.enable_adjacent_collision
    assert gs_adj_collision == mj_adj_collision

    gs_use_gjk_collision = gs_sim.rigid_solver._options.use_gjk_collision
    mj_use_gjk_collision = not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    assert gs_use_gjk_collision == mj_use_gjk_collision

    mj_solver = mujoco.mjtSolver(mj_sim.model.opt.solver)
    if mj_solver.name == "mjSOL_PGS":
        assert False
    elif mj_solver.name == "mjSOL_CG":
        assert gs_sim.rigid_solver._options.constraint_solver == gs.constraint_solver.CG
    elif mj_solver.name == "mjSOL_NEWTON":
        assert gs_sim.rigid_solver._options.constraint_solver == gs.constraint_solver.Newton
    else:
        assert False

    mj_integrator = mujoco.mjtIntegrator(mj_sim.model.opt.integrator)
    if mj_integrator.name == "mjINT_EULER":
        assert gs_sim.rigid_solver._options.integrator == gs.integrator.Euler
    elif mj_integrator.name == "mjINT_IMPLICIT":
        assert False
    elif mj_integrator.name == "mjINT_IMPLICITFAST":
        assert gs_sim.rigid_solver._options.integrator == gs.integrator.implicitfast
    else:
        assert False

    mj_cone = mujoco.mjtCone(mj_sim.model.opt.cone)
    if mj_cone.name == "mjCONE_ELLIPTIC":
        assert False
    elif mj_cone.name == "mjCONE_PYRAMIDAL":
        assert True
    else:
        assert False

    gs_roots_name = sorted(
        gs_sim.rigid_solver.links[i].name
        for i in set(gs_sim.rigid_solver.links_info.root_idx.to_numpy()[gs_bodies_idx])
    )
    mj_roots_name = sorted(mj_sim.model.body(i).name for i in set(mj_sim.model.body_rootid[mj_bodies_idx]))
    assert gs_roots_name == mj_roots_name

    # body
    for gs_i, mj_i in zip(gs_bodies_idx, mj_bodies_idx):
        gs_invweight_i = gs_sim.rigid_solver.links_info.invweight.to_numpy()[gs_i]
        mj_invweight_i = mj_sim.model.body(mj_i).invweight0
        assert_allclose(gs_invweight_i, mj_invweight_i, tol=tol)
        gs_inertia_i = gs_sim.rigid_solver.links_info.inertial_i.to_numpy()[gs_i, [0, 1, 2], [0, 1, 2]]
        mj_inertia_i = mj_sim.model.body(mj_i).inertia
        assert_allclose(gs_inertia_i, mj_inertia_i, tol=tol)
        gs_ipos_i = gs_sim.rigid_solver.links_info.inertial_pos.to_numpy()[gs_i]
        mj_ipos_i = mj_sim.model.body(mj_i).ipos
        assert_allclose(gs_ipos_i, mj_ipos_i, tol=tol)
        gs_iquat_i = gs_sim.rigid_solver.links_info.inertial_quat.to_numpy()[gs_i]
        mj_iquat_i = mj_sim.model.body(mj_i).iquat
        assert_allclose(gs_iquat_i, mj_iquat_i, tol=tol)
        gs_pos_i = gs_sim.rigid_solver.links_info.pos.to_numpy()[gs_i]
        mj_pos_i = mj_sim.model.body(mj_i).pos
        assert_allclose(gs_pos_i, mj_pos_i, tol=tol)
        gs_quat_i = gs_sim.rigid_solver.links_info.quat.to_numpy()[gs_i]
        mj_quat_i = mj_sim.model.body(mj_i).quat
        assert_allclose(gs_quat_i, mj_quat_i, tol=tol)
        gs_mass_i = gs_sim.rigid_solver.links_info.inertial_mass.to_numpy()[gs_i]
        mj_mass_i = mj_sim.model.body(mj_i).mass
        assert_allclose(gs_mass_i, mj_mass_i, tol=tol)

    # dof / joints
    gs_dof_damping = gs_sim.rigid_solver.dofs_info.damping.to_numpy()
    mj_dof_damping = mj_sim.model.dof_damping
    assert_allclose(gs_dof_damping[gs_dofs_idx], mj_dof_damping[mj_dofs_idx], tol=tol)

    gs_dof_armature = gs_sim.rigid_solver.dofs_info.armature.to_numpy()
    mj_dof_armature = mj_sim.model.dof_armature
    assert_allclose(gs_dof_armature[gs_dofs_idx], mj_dof_armature[mj_dofs_idx], tol=tol)

    # FIXME: 1 stiffness per joint in Mujoco, 1 stiffness per DoF in Genesis
    gs_dof_stiffness = gs_sim.rigid_solver.dofs_info.stiffness.to_numpy()
    mj_dof_stiffness = mj_sim.model.jnt_stiffness
    # assert_allclose(gs_dof_stiffness[gs_dofs_idx], mj_dof_stiffness[mj_joints_idx], tol=tol)

    gs_dof_invweight0 = gs_sim.rigid_solver.dofs_info.invweight.to_numpy()
    mj_dof_invweight0 = mj_sim.model.dof_invweight0
    assert_allclose(gs_dof_invweight0[gs_dofs_idx], mj_dof_invweight0[mj_dofs_idx], tol=tol)

    # TODO: Genesis does not support frictionloss contraint at dof level for now
    gs_joint_solparams = np.array([joint.sol_params.cpu() for entity in gs_sim.entities for joint in entity.joints])
    mj_joint_solparams = np.concatenate((mj_sim.model.jnt_solref, mj_sim.model.jnt_solimp), axis=-1)
    _sanitize_sol_params(
        mj_joint_solparams, gs_sim.rigid_solver._sol_min_timeconst, gs_sim.rigid_solver._sol_default_timeconst
    )
    assert_allclose(gs_joint_solparams[gs_joints_idx], mj_joint_solparams[mj_joints_idx], tol=tol)
    gs_geom_solparams = np.array([geom.sol_params.cpu() for entity in gs_sim.entities for geom in entity.geoms])
    mj_geom_solparams = np.concatenate((mj_sim.model.geom_solref, mj_sim.model.geom_solimp), axis=-1)
    _sanitize_sol_params(
        mj_geom_solparams, gs_sim.rigid_solver._sol_min_timeconst, gs_sim.rigid_solver._sol_default_timeconst
    )
    assert_allclose(gs_geom_solparams[gs_geoms_idx], mj_geom_solparams[mj_geoms_idx], tol=tol)
    # FIXME: Masking geometries and equality constraints is not supported for now
    gs_eq_solparams = np.array(
        [equality.sol_params.cpu() for entity in gs_sim.entities for equality in entity.equalities]
    ).reshape((-1, 7))
    mj_eq_solparams = np.concatenate((mj_sim.model.eq_solref, mj_sim.model.eq_solimp), axis=-1)
    _sanitize_sol_params(
        mj_eq_solparams, gs_sim.rigid_solver._sol_min_timeconst, gs_sim.rigid_solver._sol_default_timeconst
    )
    assert_allclose(gs_eq_solparams, mj_eq_solparams, tol=tol)

    assert_allclose(mj_sim.model.jnt_margin, 0, tol=tol)
    gs_joint_range = np.stack(
        [
            gs_sim.rigid_solver.dofs_info.limit[gs_sim.rigid_solver.joints_info.dof_start[i]].to_numpy()
            for i in gs_joints_idx
        ],
        axis=0,
    )
    mj_joint_range = mj_sim.model.jnt_range
    mj_joint_range[mj_sim.model.jnt_limited == 0, 0] = float("-inf")
    mj_joint_range[mj_sim.model.jnt_limited == 0, 1] = float("+inf")
    assert_allclose(gs_joint_range, mj_joint_range[mj_joints_idx], tol=tol)

    # actuator (position control)
    for v in mj_sim.model.actuator_dyntype:
        assert v == mujoco.mjtDyn.mjDYN_NONE
    for v in mj_sim.model.actuator_biastype:
        assert v in (mujoco.mjtBias.mjBIAS_AFFINE, mujoco.mjtBias.mjBIAS_NONE)

    # NOTE: not considering gear
    gs_kp = gs_sim.rigid_solver.dofs_info.kp.to_numpy()
    gs_kv = gs_sim.rigid_solver.dofs_info.kv.to_numpy()
    mj_kp = -mj_sim.model.actuator_biasprm[:, 1]
    mj_kv = -mj_sim.model.actuator_biasprm[:, 2]
    assert_allclose(gs_kp[gs_motors_idx], mj_kp[mj_motors_idx], tol=tol)
    assert_allclose(gs_kv[gs_motors_idx], mj_kv[mj_motors_idx], tol=tol)


def check_mujoco_data_consistency(
    gs_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    bodies_name: list[str] | None = None,
    *,
    qvel_prev: np.ndarray | None = None,
    tol: float,
):
    # Get mapping between Mujoco and Genesis
    gs_maps, mj_maps = _get_model_mappings(gs_sim, mj_sim, joints_name, bodies_name)
    (gs_bodies_idx, _, gs_q_idx, gs_dofs_idx, _, _) = gs_maps
    (mj_bodies_idx, _, mj_qs_idx, mj_dofs_idx, _, _) = mj_maps

    # crb
    gs_crb_inertial = gs_sim.rigid_solver.links_state.crb_inertial.to_numpy()[:, 0].reshape([-1, 9])[
        :, [0, 4, 8, 1, 2, 5]
    ]
    mj_crb_inertial = mj_sim.data.crb[:, :6]  # upper-triangular part
    assert_allclose(gs_crb_inertial[gs_bodies_idx], mj_crb_inertial[mj_bodies_idx], tol=tol)
    gs_crb_pos = gs_sim.rigid_solver.links_state.crb_pos.to_numpy()[:, 0]
    mj_crb_pos = mj_sim.data.crb[:, 6:9]
    assert_allclose(gs_crb_pos[gs_bodies_idx], mj_crb_pos[mj_bodies_idx], tol=tol)
    gs_crb_mass = gs_sim.rigid_solver.links_state.crb_mass.to_numpy()[:, 0]
    mj_crb_mass = mj_sim.data.crb[:, 9]
    assert_allclose(gs_crb_mass[gs_bodies_idx], mj_crb_mass[mj_bodies_idx], tol=tol)

    gs_mass_mat = gs_sim.rigid_solver.mass_mat.to_numpy()[:, :, 0]
    mj_mass_mat = np.zeros((mj_sim.model.nv, mj_sim.model.nv))
    mujoco.mj_fullM(mj_sim.model, mj_mass_mat, mj_sim.data.qM)
    assert_allclose(gs_mass_mat[gs_dofs_idx][:, gs_dofs_idx], mj_mass_mat[mj_dofs_idx][:, mj_dofs_idx], tol=tol)

    gs_meaninertia = gs_sim.rigid_solver.meaninertia.to_numpy()[0]
    mj_meaninertia = mj_sim.model.stat.meaninertia
    assert_allclose(gs_meaninertia, mj_meaninertia, tol=tol)

    # Pre-constraint so-called bias forces in configuration space
    gs_qfrc_bias = gs_sim.rigid_solver.dofs_state.qf_bias.to_numpy()[:, 0]
    mj_qfrc_bias = mj_sim.data.qfrc_bias
    assert_allclose(gs_qfrc_bias, mj_qfrc_bias[mj_dofs_idx], tol=tol)
    gs_qfrc_passive = gs_sim.rigid_solver.dofs_state.qf_passive.to_numpy()[:, 0]
    mj_qfrc_passive = mj_sim.data.qfrc_passive
    assert_allclose(gs_qfrc_passive, mj_qfrc_passive[mj_dofs_idx], tol=tol)
    gs_qfrc_actuator = gs_sim.rigid_solver.dofs_state.qf_applied.to_numpy()[:, 0]
    mj_qfrc_actuator = mj_sim.data.qfrc_actuator
    assert_allclose(gs_qfrc_actuator, mj_qfrc_actuator[mj_dofs_idx], tol=tol)

    gs_n_contacts = gs_sim.rigid_solver.collider._collider_state.n_contacts.to_numpy()[0]
    mj_n_contacts = mj_sim.data.ncon
    assert gs_n_contacts == mj_n_contacts
    gs_n_constraints = gs_sim.rigid_solver.constraint_solver.n_constraints.to_numpy()[0]
    mj_n_constraints = mj_sim.data.nefc
    assert gs_n_constraints == mj_n_constraints

    if gs_n_constraints:
        gs_contact_pos = gs_sim.rigid_solver.collider._collider_state.contact_data.pos.to_numpy()[:gs_n_contacts, 0]
        mj_contact_pos = mj_sim.data.contact.pos
        # Sort based on the axis with the largest variation
        max_var_axis = 0
        if gs_n_contacts > 1:
            max_var = -1
            for axis in range(3):
                sorted_contact_pos = np.sort(mj_contact_pos[:, axis])
                var = np.min(sorted_contact_pos[1:] - sorted_contact_pos[:-1])
                if var > max_var:
                    max_var_axis = axis
                    max_var = var
        gs_sidx = np.argsort(gs_contact_pos[:, max_var_axis])
        mj_sidx = np.argsort(mj_contact_pos[:, max_var_axis])
        assert_allclose(gs_contact_pos[gs_sidx], mj_contact_pos[mj_sidx], tol=tol)
        gs_contact_normal = gs_sim.rigid_solver.collider._collider_state.contact_data.normal.to_numpy()[
            :gs_n_contacts, 0
        ]
        mj_contact_normal = -mj_sim.data.contact.frame[:, :3]
        assert_allclose(gs_contact_normal[gs_sidx], mj_contact_normal[mj_sidx], tol=tol)
        gs_penetration = gs_sim.rigid_solver.collider._collider_state.contact_data.penetration.to_numpy()[
            :gs_n_contacts, 0
        ]
        mj_penetration = -mj_sim.data.contact.dist
        assert_allclose(gs_penetration[gs_sidx], mj_penetration[mj_sidx], tol=tol)

        # FIXME: It is not always possible to reshape Mujoco jacobian because joint bound constraints are computed in
        # "sparse" dof space, unlike contact constraints.
        error = None
        gs_jac = gs_sim.rigid_solver.constraint_solver.jac.to_numpy()[:gs_n_constraints, :, 0]
        mj_jac = mj_sim.data.efc_J.reshape([mj_n_constraints, -1])
        gs_efc_D = gs_sim.rigid_solver.constraint_solver.efc_D.to_numpy()[:gs_n_constraints, 0]
        mj_efc_D = mj_sim.data.efc_D
        gs_efc_aref = gs_sim.rigid_solver.constraint_solver.aref.to_numpy()[:gs_n_constraints, 0]
        mj_efc_aref = mj_sim.data.efc_aref
        for gs_sidx, mj_sidx in (
            (np.argsort(gs_jac.sum(axis=1)), np.argsort(mj_jac.sum(axis=1))),
            (np.argsort(gs_efc_aref), np.argsort(mj_efc_aref)),
        ):
            try:
                gs_jac_nz_mask = (np.abs(gs_jac[gs_sidx]) > 0.0).all(axis=0)
                gs_jac_nz = gs_jac[gs_sidx][:, np.array(gs_dofs_idx)[gs_jac_nz_mask[gs_dofs_idx]]]
                mj_jac_nz_mask = np.zeros_like(gs_jac_nz_mask, dtype=np.bool_)
                mj_jac_nz_mask[mj_dofs_idx] = gs_jac_nz_mask[gs_dofs_idx]
                if mj_jac.shape[-1] == len(mj_dofs_idx):
                    mj_jac_nz = mj_jac[mj_sidx][:, np.array(mj_dofs_idx)[mj_jac_nz_mask[mj_dofs_idx]]]
                else:
                    mj_jac_nz = mj_jac[mj_sidx]

                assert_allclose(gs_jac_nz, mj_jac_nz, tol=tol)
                assert_allclose(gs_efc_D[gs_sidx], mj_efc_D[mj_sidx], tol=tol)
                assert_allclose(gs_efc_aref[gs_sidx], mj_efc_aref[mj_sidx], tol=tol)
                break
            except AssertionError as e:
                error = e
        else:
            assert error is not None
            raise error

        gs_efc_force = gs_sim.rigid_solver.constraint_solver.efc_force.to_numpy()[:gs_n_constraints, 0]
        mj_efc_force = mj_sim.data.efc_force
        assert_allclose(gs_efc_force[gs_sidx], mj_efc_force[mj_sidx], tol=tol)

    if gs_n_constraints:
        mj_iter = mj_sim.data.solver_niter[0] - 1
        if gs_n_constraints and mj_iter >= 0:
            gs_scale = 1.0 / (gs_meaninertia * max(1, gs_sim.rigid_solver.n_dofs))
            gs_gradient = gs_scale * np.linalg.norm(
                gs_sim.rigid_solver.constraint_solver.grad.to_numpy()[: gs_sim.rigid_solver.n_dofs, 0]
            )
            mj_gradient = mj_sim.data.solver.gradient[mj_iter]
            assert_allclose(gs_gradient, mj_gradient, tol=tol)
            gs_improvement = gs_scale * (
                gs_sim.rigid_solver.constraint_solver.prev_cost[0] - gs_sim.rigid_solver.constraint_solver.cost[0]
            )
            mj_improvement = mj_sim.data.solver.improvement[mj_iter]
            # FIXME: This is too challenging to match because of compounding of errors
            # assert_allclose(gs_improvement, mj_improvement, tol=tol)

        if qvel_prev is not None:
            gs_efc_vel = gs_jac @ qvel_prev
            mj_efc_vel = mj_sim.data.efc_vel
            assert_allclose(gs_efc_vel[gs_sidx], mj_efc_vel[mj_sidx], tol=tol)

    gs_qfrc_constraint = gs_sim.rigid_solver.dofs_state.qf_constraint.to_numpy()[:, 0]
    mj_qfrc_constraint = mj_sim.data.qfrc_constraint
    assert_allclose(gs_qfrc_constraint[gs_dofs_idx], mj_qfrc_constraint[mj_dofs_idx], tol=tol)

    gs_qfrc_all = gs_sim.rigid_solver.dofs_state.force.to_numpy()[:, 0]
    mj_qfrc_all = mj_sim.data.qfrc_smooth + mj_sim.data.qfrc_constraint
    assert_allclose(gs_qfrc_all[gs_dofs_idx], mj_qfrc_all[mj_dofs_idx], tol=tol)

    gs_qfrc_smooth = gs_sim.rigid_solver.dofs_state.qf_smooth.to_numpy()[:, 0]
    mj_qfrc_smooth = mj_sim.data.qfrc_smooth
    assert_allclose(gs_qfrc_smooth[gs_dofs_idx], mj_qfrc_smooth[mj_dofs_idx], tol=tol)

    gs_qacc_smooth = gs_sim.rigid_solver.dofs_state.acc_smooth.to_numpy()[:, 0]
    mj_qacc_smooth = mj_sim.data.qacc_smooth
    assert_allclose(gs_qacc_smooth[gs_dofs_idx], mj_qacc_smooth[mj_dofs_idx], tol=tol)

    # Acceleration pre- VS post-implicit damping
    # gs_qacc_post = gs_sim.rigid_solver.dofs_state.acc.to_numpy()[:, 0]
    if gs_n_constraints:
        gs_qacc_pre = gs_sim.rigid_solver.constraint_solver.qacc.to_numpy()[:, 0]
    else:
        gs_qacc_pre = gs_qacc_smooth
    mj_qacc_pre = mj_sim.data.qacc
    assert_allclose(gs_qacc_pre[gs_dofs_idx], mj_qacc_pre[mj_dofs_idx], tol=tol)

    gs_qvel = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
    mj_qvel = mj_sim.data.qvel
    assert_allclose(gs_qvel[gs_dofs_idx], mj_qvel[mj_dofs_idx], tol=tol)
    gs_qpos = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    mj_qpos = mj_sim.data.qpos
    assert_allclose(gs_qpos[gs_q_idx], mj_qpos[mj_qs_idx], tol=tol)

    # ------------------------------------------------------------------------

    gs_com = gs_sim.rigid_solver.links_state.root_COM.to_numpy()[:, 0]
    gs_root_idx = np.unique(gs_sim.rigid_solver.links_info.root_idx.to_numpy()[gs_bodies_idx])
    mj_com = mj_sim.data.subtree_com
    mj_root_idx = np.unique(mj_sim.model.body_rootid[mj_bodies_idx])
    assert_allclose(gs_com[gs_root_idx], mj_com[mj_root_idx], tol=tol)

    gs_xipos = gs_sim.rigid_solver.links_state.i_pos.to_numpy()[:, 0]
    mj_xipos = mj_sim.data.xipos - mj_sim.data.subtree_com[mj_sim.model.body_rootid]
    assert_allclose(gs_xipos[gs_bodies_idx], mj_xipos[mj_bodies_idx], tol=tol)

    gs_xpos = gs_sim.rigid_solver.links_state.pos.to_numpy()[:, 0]
    mj_xpos = mj_sim.data.xpos
    assert_allclose(gs_xpos[gs_bodies_idx], mj_xpos[mj_bodies_idx], tol=tol)

    gs_xquat = gs_sim.rigid_solver.links_state.quat.to_numpy()[:, 0]
    gs_xmat = gu.quat_to_R(gs_xquat).reshape([-1, 9])
    mj_xmat = mj_sim.data.xmat
    assert_allclose(gs_xmat[gs_bodies_idx], mj_xmat[mj_bodies_idx], tol=tol)

    gs_cd_vel = gs_sim.rigid_solver.links_state.cd_vel.to_numpy()[:, 0]
    mj_cd_vel = mj_sim.data.cvel[:, 3:]
    assert_allclose(gs_cd_vel[gs_bodies_idx], mj_cd_vel[mj_bodies_idx], tol=tol)
    gs_cd_ang = gs_sim.rigid_solver.links_state.cd_ang.to_numpy()[:, 0]
    mj_cd_ang = mj_sim.data.cvel[:, :3]
    assert_allclose(gs_cd_ang[gs_bodies_idx], mj_cd_ang[mj_bodies_idx], tol=tol)

    gs_cdof_vel = gs_sim.rigid_solver.dofs_state.cdof_vel.to_numpy()[:, 0]
    mj_cdof_vel = mj_sim.data.cdof[:, 3:]
    assert_allclose(gs_cdof_vel[gs_dofs_idx], mj_cdof_vel[mj_dofs_idx], tol=tol)
    gs_cdof_ang = gs_sim.rigid_solver.dofs_state.cdof_ang.to_numpy()[:, 0]
    mj_cdof_ang = mj_sim.data.cdof[:, :3]
    assert_allclose(gs_cdof_ang[gs_dofs_idx], mj_cdof_ang[mj_dofs_idx], tol=tol)

    mj_cdof_dot_ang = mj_sim.data.cdof_dot[:, :3]
    gs_cdof_dot_ang = gs_sim.rigid_solver.dofs_state.cdofd_ang.to_numpy()[:, 0]
    assert_allclose(gs_cdof_dot_ang[gs_dofs_idx], mj_cdof_dot_ang[mj_dofs_idx], tol=tol)

    mj_cdof_dot_vel = mj_sim.data.cdof_dot[:, 3:]
    gs_cdof_dot_vel = gs_sim.rigid_solver.dofs_state.cdofd_vel.to_numpy()[:, 0]
    assert_allclose(gs_cdof_dot_vel[gs_dofs_idx], mj_cdof_dot_vel[mj_dofs_idx], tol=tol)

    # cinr
    gs_cinr_inertial = gs_sim.rigid_solver.links_state.cinr_inertial.to_numpy()[:, 0].reshape([-1, 9])[
        :, [0, 4, 8, 1, 2, 5]
    ]
    mj_cinr_inertial = mj_sim.data.cinert[:, :6]  # upper-triangular part
    assert_allclose(gs_cinr_inertial[gs_bodies_idx], mj_cinr_inertial[mj_bodies_idx], tol=tol)
    gs_cinr_pos = gs_sim.rigid_solver.links_state.cinr_pos.to_numpy()[:, 0]
    mj_cinr_pos = mj_sim.data.cinert[:, 6:9]
    assert_allclose(gs_cinr_pos[gs_bodies_idx], mj_cinr_pos[mj_bodies_idx], tol=tol)
    gs_cinr_mass = gs_sim.rigid_solver.links_state.cinr_mass.to_numpy()[:, 0]
    mj_cinr_mass = mj_sim.data.cinert[:, 9]
    assert_allclose(gs_cinr_mass[gs_bodies_idx], mj_cinr_mass[mj_bodies_idx], tol=tol)


def simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos=None, qvel=None, *, tol, num_steps):
    # Get mapping between Mujoco and Genesis
    _, (_, _, mj_qs_idx, mj_dofs_idx, _, _) = _get_model_mappings(gs_sim, mj_sim)

    # Make sure that "static" model information are matching
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)

    # Initialize the simulation
    init_simulators(gs_sim, mj_sim, qpos, qvel)

    # Run the simulation for a few steps
    qvel_prev = None

    for i in range(num_steps):
        # Make sure that all "dynamic" quantities are matching before stepping
        check_mujoco_data_consistency(gs_sim, mj_sim, qvel_prev=qvel_prev, tol=tol)

        # Keep Mujoco and Genesis simulation in sync to avoid drift over time
        mj_sim.data.qpos[mj_qs_idx] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[mj_dofs_idx] = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
        mj_sim.data.qacc_warmstart[mj_dofs_idx] = gs_sim.rigid_solver.constraint_solver.qacc_ws.to_numpy()[:, 0]
        mj_sim.data.qacc_smooth[mj_dofs_idx] = gs_sim.rigid_solver.dofs_state.acc_smooth.to_numpy()[:, 0]

        # Backup current velocity
        qvel_prev = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

        # Do a single simulation step (eventually with substeps for Genesis)
        mujoco.mj_step(mj_sim.model, mj_sim.data)
        gs_sim.scene.step()
        # if gs_sim.scene.visualizer:
        #     gs_sim.scene.visualizer.update()


def rgb_array_to_png_bytes(rgb_arr: np.ndarray) -> bytes:
    img = Image.fromarray(rgb_arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
