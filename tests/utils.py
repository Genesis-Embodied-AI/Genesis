import base64
import io
import json
import numbers
import os
import platform
import re
import subprocess
import tempfile
import time
import uuid
import webbrowser
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import cache
from itertools import chain
from pathlib import Path
from types import GeneratorType
from typing import Literal, NamedTuple, Sequence

import cpuinfo
import igl
import mujoco
import pytest
import numpy as np
import torch
from httpcore import TimeoutException as HTTPTimeoutException
from httpx import HTTPError as HTTPXError
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image, UnidentifiedImageError
from requests.exceptions import HTTPError

import genesis as gs
import genesis.utils.geom as gu
from genesis.options.morphs import GLTF_FORMATS, MESH_FORMATS, MJCF_FORMAT, URDF_FORMAT, USD_FORMATS
from genesis.utils import mjcf as mju
from genesis.utils.mesh import get_assets_dir
from genesis.utils.misc import tensor_to_array

REPOSITY_URL = "Genesis-Embodied-AI/Genesis"
DEFAULT_BRANCH_NAME = "main"

HUGGINGFACE_ASSETS_REVISION = "990a727788f11e34ad006c69bf769303b20cb11c"
HUGGINGFACE_SNAPSHOT_REVISION = "d8e1eac38320a5507ea5b851d18431b1aba836dc"

MESH_EXTENSIONS = (".mtl", *MESH_FORMATS, *GLTF_FORMATS, *USD_FORMATS)
IMAGE_EXTENSIONS = (".png", ".jpg")

IMG_STD_ERR_THR = 1.0
IMG_NUM_ERR_THR = 0.001
IMG_BLUR_KERNEL_SIZE = 1  # Size of the blur kernel (must be odd)


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
        next(filter(None, map(cpu_info.get, ("brand_raw", "hardware_raw", "vendor_id_raw")))),
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
        try:
            remote_handle = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote_url).group(1)
        except AttributeError:
            pass
        if remote_handle == REPOSITY_URL:
            is_commit_on_default_branch = True
            break
    else:
        is_commit_on_default_branch = False
    revision = f"{revision}@{remote_handle}"

    # Return the contribution date as timestamp if and only if the HEAD commit is on main branch
    if is_commit_on_default_branch:
        timestamp = get_git_commit_timestamp(ref)
    else:
        timestamp = float("nan")

    return revision, timestamp


def get_hf_dataset(
    pattern,
    repo_name: str = "assets",
    local_dir: str | None = None,
    num_retry: int = 4,
    retry_delay: float = 30.0,
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
        except (HTTPTimeoutException, HTTPXError, HTTPError, LocalEntryNotFoundError, FileNotFoundError, RuntimeError):
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
        if isinstance(arg, (GeneratorType, map)):
            arg = tuple(arg)
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
    except ValueError as e:
        try:
            args = np.broadcast_arrays(*map(np.squeeze, args))
        except ValueError:
            raise e

    np.testing.assert_allclose(*args, atol=atol, rtol=rtol, err_msg=err_msg)


def assert_equal(actual, desired, *, err_msg=None):
    assert_allclose(actual, desired, atol=0.0, rtol=0.0, err_msg=err_msg)


def assert_pixel_match(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    err_msg: str = "Images do not match",
    verbose: bool = True,
    std_err_threshold: float = IMG_STD_ERR_THR,
    ratio_err_threshold: float = IMG_NUM_ERR_THR,
    blurred_kernel_size: int = IMG_BLUR_KERNEL_SIZE,
) -> None:
    """Assert two RGB image arrays match.

    The images match unless the per-channel standard deviation of their blurred difference exceeds
    ``std_err_threshold`` AND the number of differing pixels exceeds ``ratio_err_threshold`` of the total size.
    This tolerates the few-pixel jitter that software renderers produce on any platform while still catching a
    real difference. On mismatch, raise ``AssertionError``; unless ``verbose`` is False, also print the error
    metrics and a base64-encoded PNG of the per-pixel delta (so the failing frame can be recovered from CI logs).
    """
    img_a = np.atleast_3d(np.asarray(img_a)).astype(np.float32)
    img_b = np.atleast_3d(np.asarray(img_b)).astype(np.float32)
    if img_a.shape != img_b.shape:
        raise AssertionError(f"{err_msg} (shape {img_a.shape} != {img_b.shape})")

    # Blur both images with a normalized box kernel to smooth anti-aliasing edges before comparing.
    blurred = []
    for img_arr in (img_a, img_b):
        if blurred_kernel_size == 1:
            blurred.append(img_arr)
            continue
        kernel = np.ones((blurred_kernel_size, blurred_kernel_size), dtype=np.float32) / (blurred_kernel_size**2)
        pad_size = blurred_kernel_size // 2
        h, w = img_arr.shape[:2]
        padded = np.pad(img_arr, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="edge")
        blurred_arr = np.zeros_like(img_arr, dtype=np.float32)
        for c in range(img_arr.shape[-1]):
            for i in range(h):
                for j in range(w):
                    blurred_arr[i, j, c] = np.sum(
                        padded[i : i + blurred_kernel_size, j : j + blurred_kernel_size, c] * kernel
                    )
        blurred.append(blurred_arr)

    img_err = np.minimum(np.abs(blurred[1] - blurred[0]), 255).astype(np.uint8)
    std_err = float(np.max(np.std(img_err.reshape((-1, img_err.shape[-1])), axis=0)))
    ratio_err = int((np.abs(img_err) > np.finfo(np.float32).eps).sum())
    if not (std_err > std_err_threshold and ratio_err > ratio_err_threshold * img_err.size):
        return

    if verbose:
        print(
            f"Image mismatch [std_err={std_err:.2f} (thr={std_err_threshold:.2f}), "
            f"ratio_err={ratio_err} (thr={ratio_err_threshold * img_err.size})]:"
        )
        raw_bytes = io.BytesIO()
        img_delta = np.minimum(np.abs(img_b - img_a), 255).astype(np.uint8)
        img_obj = Image.fromarray(img_delta.squeeze(-1) if img_delta.shape[-1] == 1 else img_delta)
        img_obj.save(raw_bytes, "PNG")
        raw_bytes.seek(0)
        print(base64.b64encode(raw_bytes.read()))
    raise AssertionError(err_msg)


def init_simulators(gs_sim, mj_sim=None, qpos=None, qvel=None):
    if mj_sim is not None:
        _, (_, _, mj_qs_idx, mj_dofs_idx, _, _) = _get_model_mappings(gs_sim, mj_sim)

    (gs_robot,) = gs_sim.entities

    gs_sim.scene.reset()
    if qpos is not None:
        gs_robot.set_qpos(qpos)
    if qvel is not None:
        gs_robot.set_dofs_velocity(qvel)

    gs_sim.rigid_solver.dyn_state.dofs.qf_constraint.fill(0.0)
    gs_sim.rigid_solver._func_forward_dynamics()
    gs_sim.rigid_solver._func_constraint_force()
    gs_sim.rigid_solver._func_update_acc()

    if gs_sim.scene.visualizer:
        gs_sim.scene.visualizer.update()

    if mj_sim is not None:
        mujoco.mj_resetData(mj_sim.model, mj_sim.data)
        mj_sim.data.qpos[mj_qs_idx] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[mj_dofs_idx] = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]
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
            try:
                mj_joint = mj_sim.model.joint(joint_name)
            except KeyError:
                for entity in gs_sim.entities:
                    for joint in entity.joints:
                        if joint.name == joint_name:
                            mj_joint = mj_sim.model.joint(joint.idx)
                            break
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

    gs_joints_idx, gs_q_idx, gs_dofs_idx = _gs_search_by_joints_name(gs_sim.scene, joints_name)
    _, _, gs_motors_idx = _gs_search_by_joints_name(gs_sim.scene, motors_name)

    gs_bodies_idx = _gs_search_by_links_name(gs_sim.scene, bodies_name)
    gs_geoms_idx: list[int] = []
    for gs_body_idx in gs_bodies_idx:
        link = gs_sim.rigid_solver.links[gs_body_idx]
        gs_geoms_idx += range(link.geom_start, link.geom_end)

    gs_maps = (gs_bodies_idx, gs_joints_idx, gs_q_idx, gs_dofs_idx, gs_geoms_idx, gs_motors_idx)
    mj_maps = (mj_bodies_idx, mj_joints_idx, mj_qs_idx, mj_dofs_idx, mj_geoms_idx, mj_motors_idx)
    return gs_maps, mj_maps


def build_mujoco_sim(
    xml_path,
    gs_solver,
    gs_integrator,
    merge_fixed_links,
    multi_contact,
    adjacent_collision,
    native_ccd,
    *,
    friction_cone,
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

    file = os.path.join(get_assets_dir(), xml_path)
    if not os.path.exists(file):
        asset_path = get_hf_dataset(pattern=xml_path)
        file = os.path.join(asset_path, xml_path)

    model = mju.build_model(
        file, discard_visual=True, default_armature=None, merge_fixed_links=merge_fixed_links, links_to_keep=()
    )

    model.opt.solver = mj_solver
    model.opt.integrator = mj_integrator
    if friction_cone == gs.friction_cone.elliptic:
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    else:
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
    *,
    friction_cone,
):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            res=(960, 640),
        ),
        sim_options=gs.options.SimOptions(
            dt=mj_sim.model.opt.timestep,
            substeps=1,
            gravity=mj_sim.model.opt.gravity,
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs_integrator,
            constraint_solver=gs_solver,
            enable_mujoco_compatibility=mujoco_compatibility,
            friction_cone=friction_cone,
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

    file = os.path.join(get_assets_dir(), xml_path)
    if not os.path.exists(file):
        asset_path = get_hf_dataset(pattern=xml_path)
        file = os.path.join(asset_path, xml_path)

    morph_kwargs = dict(
        file=file,
        convexify=True,
        decompose_robot_error_threshold=float("inf"),
        align=False,
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
    scene.add_entity(
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
    from genesis.engine.solvers.rigid.rigid_solver import _sanitize_sol_params

    # Get mapping between Mujoco and Genesis
    gs_maps, mj_maps = _get_model_mappings(gs_sim, mj_sim, joints_name, bodies_name)
    gs_bodies_idx, gs_joints_idx, gs_q_idx, gs_dofs_idx, gs_geoms_idx, gs_motors_idx = gs_maps
    mj_bodies_idx, mj_joints_idx, mj_qs_idx, mj_dofs_idx, mj_geoms_idx, mj_motors_idx = mj_maps

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
        assert gs_sim.rigid_solver._options.friction_cone == gs.friction_cone.elliptic
        assert_allclose(gs_sim.rigid_solver._options.impratio, mj_sim.model.opt.impratio, tol=tol)
    elif mj_cone.name == "mjCONE_PYRAMIDAL":
        assert gs_sim.rigid_solver._options.friction_cone == gs.friction_cone.pyramidal
    else:
        assert False

    gs_roots_name = sorted(
        gs_sim.rigid_solver.links[i].name
        for i in set(gs_sim.rigid_solver.dyn_info.links.root_idx.to_numpy()[gs_bodies_idx])
    )
    mj_roots_name = sorted(mj_sim.model.body(i).name for i in set(mj_sim.model.body_rootid[mj_bodies_idx]))
    assert gs_roots_name == mj_roots_name

    # body
    for gs_i, mj_i in zip(gs_bodies_idx, mj_bodies_idx):
        gs_invweight_i = gs_sim.rigid_solver.dyn_info.links.invweight.to_numpy()[gs_i]
        mj_invweight_i = mj_sim.model.body(mj_i).invweight0
        try:
            assert_allclose(gs_invweight_i, mj_invweight_i, tol=tol)
        except AssertionError:
            if tuple(int(x) for x in mujoco.__version__.split(".")[:2]) < (3, 5):
                pytest.skip(
                    "MuJoCo < 3.5 lacks the degenerate invweight fix. "
                    "See https://github.com/google-deepmind/mujoco/commit/1cda1e7a"
                )
            raise
        gs_inertia_i = gs_sim.rigid_solver.dyn_info.links.inertial_i.to_numpy()[gs_i, [0, 1, 2], [0, 1, 2]]
        mj_inertia_i = mj_sim.model.body(mj_i).inertia
        assert_allclose(gs_inertia_i, mj_inertia_i, tol=tol)
        gs_ipos_i = gs_sim.rigid_solver.dyn_info.links.inertial_pos.to_numpy()[gs_i]
        mj_ipos_i = mj_sim.model.body(mj_i).ipos
        assert_allclose(gs_ipos_i, mj_ipos_i, tol=tol)
        gs_iquat_i = gs_sim.rigid_solver.dyn_info.links.inertial_quat.to_numpy()[gs_i]
        mj_iquat_i = mj_sim.model.body(mj_i).iquat
        assert_allclose(gs_iquat_i, mj_iquat_i, tol=tol)
        gs_pos_i = gs_sim.rigid_solver.dyn_info.links.pos.to_numpy()[gs_i]
        mj_pos_i = mj_sim.model.body(mj_i).pos
        assert_allclose(gs_pos_i, mj_pos_i, tol=tol)
        gs_quat_i = gs_sim.rigid_solver.dyn_info.links.quat.to_numpy()[gs_i]
        mj_quat_i = mj_sim.model.body(mj_i).quat
        assert_allclose(gs_quat_i, mj_quat_i, tol=tol)
        gs_mass_i = gs_sim.rigid_solver.dyn_info.links.inertial_mass.to_numpy()[gs_i]
        mj_mass_i = mj_sim.model.body(mj_i).mass
        assert_allclose(gs_mass_i, mj_mass_i, tol=tol)

    # dof / joints
    gs_dof_damping = gs_sim.rigid_solver.dyn_info.dofs.damping.to_numpy()
    mj_dof_damping = mj_sim.model.dof_damping
    assert_allclose(gs_dof_damping[gs_dofs_idx], mj_dof_damping[mj_dofs_idx], tol=tol)

    gs_dof_armature = gs_sim.rigid_solver.dyn_info.dofs.armature.to_numpy()
    mj_dof_armature = mj_sim.model.dof_armature
    assert_allclose(gs_dof_armature[gs_dofs_idx], mj_dof_armature[mj_dofs_idx], tol=tol)

    # TODO: 1 stiffness per joint in Mujoco, 1 stiffness per DoF in Genesis
    gs_dof_stiffness = gs_sim.rigid_solver.dyn_info.dofs.stiffness.to_numpy()
    mj_dof_stiffness = mj_sim.model.jnt_stiffness
    if all(joint.n_dofs == 1 for joint in gs_sim.rigid_solver.joints):
        assert_allclose(gs_dof_stiffness[gs_dofs_idx], mj_dof_stiffness[mj_joints_idx], tol=tol)

    gs_dof_invweight0 = gs_sim.rigid_solver.dyn_info.dofs.invweight.to_numpy()
    mj_dof_invweight0 = mj_sim.model.dof_invweight0
    assert_allclose(gs_dof_invweight0[gs_dofs_idx], mj_dof_invweight0[mj_dofs_idx], tol=tol)

    gs_dof_dof_frictionloss = gs_sim.rigid_solver.dyn_info.dofs.frictionloss.to_numpy()
    mj_dof_dof_frictionloss = mj_sim.model.dof_frictionloss
    assert_allclose(gs_dof_dof_frictionloss[gs_dofs_idx], mj_dof_dof_frictionloss[mj_dofs_idx], tol=tol)

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
            gs_sim.rigid_solver.dyn_info.dofs.limit[gs_sim.rigid_solver.dyn_info.joints.dof_start[i]].to_numpy()
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

    # NOTE: not considering gear for biasprm (only relevant for AFFINE actuators where gear=1 in practice).
    gs_act_gain = gs_sim.rigid_solver.dyn_info.dofs.act_gain.to_numpy()
    gs_act_bias = gs_sim.rigid_solver.dyn_info.dofs.act_bias.to_numpy()
    mj_gear = mj_sim.model.actuator_gear[:, 0]
    mj_gainprm = mj_sim.model.actuator_gainprm[:, 0] * mj_gear
    mj_biasprm = mj_sim.model.actuator_biasprm[:, :3] * mj_gear[:, None]
    assert_allclose(gs_act_gain[gs_motors_idx], mj_gainprm[mj_motors_idx], tol=tol)
    assert_allclose(gs_act_bias[gs_motors_idx], mj_biasprm[mj_motors_idx], tol=tol)


def check_mujoco_data_consistency(
    gs_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    bodies_name: list[str] | None = None,
    *,
    qvel_prev: np.ndarray | None = None,
    tol: float,
    ignore_constraints: bool = False,
):
    # Get mapping between Mujoco and Genesis
    gs_maps, mj_maps = _get_model_mappings(gs_sim, mj_sim, joints_name, bodies_name)
    gs_bodies_idx, _, gs_q_idx, gs_dofs_idx, _, _ = gs_maps
    mj_bodies_idx, _, mj_qs_idx, mj_dofs_idx, _, _ = mj_maps

    # crb
    gs_crb_inertial = gs_sim.rigid_solver.dyn_state.links.crb_inertial.to_numpy()[:, 0].reshape([-1, 9])[
        :, [0, 4, 8, 1, 2, 5]
    ]
    mj_crb_inertial = mj_sim.data.crb[:, :6]  # upper-triangular part
    assert_allclose(gs_crb_inertial[gs_bodies_idx], mj_crb_inertial[mj_bodies_idx], tol=tol)
    gs_crb_pos = gs_sim.rigid_solver.dyn_state.links.crb_pos.to_numpy()[:, 0]
    mj_crb_pos = mj_sim.data.crb[:, 6:9]
    assert_allclose(gs_crb_pos[gs_bodies_idx], mj_crb_pos[mj_bodies_idx], tol=tol)
    gs_crb_mass = gs_sim.rigid_solver.dyn_state.links.crb_mass.to_numpy()[:, 0]
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
    gs_qfrc_bias = gs_sim.rigid_solver.dyn_state.dofs.qf_bias.to_numpy()[:, 0]
    mj_qfrc_bias = mj_sim.data.qfrc_bias
    assert_allclose(gs_qfrc_bias, mj_qfrc_bias[mj_dofs_idx], tol=tol)
    gs_qfrc_passive = gs_sim.rigid_solver.dyn_state.dofs.qf_passive.to_numpy()[:, 0]
    mj_qfrc_passive = mj_sim.data.qfrc_passive
    assert_allclose(gs_qfrc_passive, mj_qfrc_passive[mj_dofs_idx], tol=tol)
    gs_qfrc_actuator = gs_sim.rigid_solver.dyn_state.dofs.qf_applied.to_numpy()[:, 0]
    mj_qfrc_actuator = mj_sim.data.qfrc_actuator
    assert_allclose(gs_qfrc_actuator, mj_qfrc_actuator[mj_dofs_idx], tol=tol)

    gs_n_contacts = gs_sim.rigid_solver.collider._collider_state.n_contacts.to_numpy()[0]
    mj_n_contacts = mj_sim.data.ncon
    assert gs_n_contacts == mj_n_contacts
    gs_n_constraints = gs_sim.rigid_solver.constraint_solver.n_constraints.to_numpy()[0]
    mj_n_constraints = mj_sim.data.nefc
    assert gs_n_constraints == mj_n_constraints

    if gs_n_constraints and not ignore_constraints:
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

            # Note that 'constraint_solver.active' refers to whether the quadratic part of a constraint is active,
            # unlike Mujoco that defines 'nactive' as the number of active constraints regardless of its type.
            # In practice, this only makes a difference if frictionloss is enabled.
            gs_nactive = sum(gs_sim.rigid_solver.constraint_solver.active.to_numpy()[:gs_n_constraints, 0])
            mj_native = mj_sim.data.solver.nactive[mj_iter]
            if not (gs_sim.rigid_solver.dyn_info.dofs.frictionloss.to_numpy() > gs.EPS).any():
                assert mj_native == gs_nactive

            # FIXME: For some reason, mujoco is sometimes (seemingful) wrongly reporting 0...
            if mj_improvement > gs.EPS:
                # Must relax tolerance because of compounding of errors.
                assert_allclose(gs_improvement, mj_improvement, tol=tol * 1e2)

        if qvel_prev is not None:
            gs_efc_vel = gs_jac @ qvel_prev
            mj_efc_vel = mj_sim.data.efc_vel
            assert_allclose(gs_efc_vel[gs_sidx], mj_efc_vel[mj_sidx], tol=tol)

    gs_qfrc_constraint = gs_sim.rigid_solver.dyn_state.dofs.qf_constraint.to_numpy()[:, 0]
    mj_qfrc_constraint = mj_sim.data.qfrc_constraint
    assert_allclose(gs_qfrc_constraint[gs_dofs_idx], mj_qfrc_constraint[mj_dofs_idx], tol=tol)

    gs_qfrc_all = gs_sim.rigid_solver.dyn_state.dofs.force.to_numpy()[:, 0]
    mj_qfrc_all = mj_sim.data.qfrc_smooth + mj_sim.data.qfrc_constraint
    assert_allclose(gs_qfrc_all[gs_dofs_idx], mj_qfrc_all[mj_dofs_idx], tol=tol)

    gs_qfrc_smooth = gs_sim.rigid_solver.dyn_state.dofs.qf_smooth.to_numpy()[:, 0]
    mj_qfrc_smooth = mj_sim.data.qfrc_smooth
    assert_allclose(gs_qfrc_smooth[gs_dofs_idx], mj_qfrc_smooth[mj_dofs_idx], tol=tol)

    gs_qacc_smooth = gs_sim.rigid_solver.dyn_state.dofs.acc_smooth.to_numpy()[:, 0]
    mj_qacc_smooth = mj_sim.data.qacc_smooth
    assert_allclose(gs_qacc_smooth[gs_dofs_idx], mj_qacc_smooth[mj_dofs_idx], tol=tol)

    # Acceleration pre- VS post-implicit damping gs_qacc_post = gs_sim.rigid_solver.dyn_state.dofs.acc.to_numpy()[:, 0]
    if gs_n_constraints:
        gs_qacc_pre = gs_sim.rigid_solver.constraint_solver.qacc.to_numpy()[:, 0]
    else:
        gs_qacc_pre = gs_qacc_smooth
    mj_qacc_pre = mj_sim.data.qacc
    assert_allclose(gs_qacc_pre[gs_dofs_idx], mj_qacc_pre[mj_dofs_idx], tol=tol)

    gs_qvel = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]
    mj_qvel = mj_sim.data.qvel
    assert_allclose(gs_qvel[gs_dofs_idx], mj_qvel[mj_dofs_idx], tol=tol)
    gs_qpos = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    mj_qpos = mj_sim.data.qpos
    assert_allclose(gs_qpos[gs_q_idx], mj_qpos[mj_qs_idx], tol=tol)

    # ------------------------------------------------------------------------

    gs_com = gs_sim.rigid_solver.dyn_state.links.root_COM.to_numpy()[:, 0]
    gs_root_idx = np.unique(gs_sim.rigid_solver.dyn_info.links.root_idx.to_numpy()[gs_bodies_idx])
    mj_com = mj_sim.data.subtree_com
    mj_root_idx = np.unique(mj_sim.model.body_rootid[mj_bodies_idx])
    assert_allclose(gs_com[gs_root_idx], mj_com[mj_root_idx], tol=tol)

    gs_xipos = gs_sim.rigid_solver.dyn_state.links.i_pos.to_numpy()[:, 0]
    mj_xipos = mj_sim.data.xipos - mj_sim.data.subtree_com[mj_sim.model.body_rootid]
    assert_allclose(gs_xipos[gs_bodies_idx], mj_xipos[mj_bodies_idx], tol=tol)

    gs_xpos = gs_sim.rigid_solver.dyn_state.links.pos.to_numpy()[:, 0]
    mj_xpos = mj_sim.data.xpos
    assert_allclose(gs_xpos[gs_bodies_idx], mj_xpos[mj_bodies_idx], tol=tol)

    gs_xquat = gs_sim.rigid_solver.dyn_state.links.quat.to_numpy()[:, 0]
    gs_xmat = gu.quat_to_R(gs_xquat).reshape([-1, 9])
    mj_xmat = mj_sim.data.xmat
    assert_allclose(gs_xmat[gs_bodies_idx], mj_xmat[mj_bodies_idx], tol=tol)

    gs_cd_vel = gs_sim.rigid_solver.dyn_state.links.cd_vel.to_numpy()[:, 0]
    mj_cd_vel = mj_sim.data.cvel[:, 3:]
    assert_allclose(gs_cd_vel[gs_bodies_idx], mj_cd_vel[mj_bodies_idx], tol=tol)
    gs_cd_ang = gs_sim.rigid_solver.dyn_state.links.cd_ang.to_numpy()[:, 0]
    mj_cd_ang = mj_sim.data.cvel[:, :3]
    assert_allclose(gs_cd_ang[gs_bodies_idx], mj_cd_ang[mj_bodies_idx], tol=tol)

    gs_cdof_vel = gs_sim.rigid_solver.dyn_state.dofs.cdof_vel.to_numpy()[:, 0]
    mj_cdof_vel = mj_sim.data.cdof[:, 3:]
    assert_allclose(gs_cdof_vel[gs_dofs_idx], mj_cdof_vel[mj_dofs_idx], tol=tol)
    gs_cdof_ang = gs_sim.rigid_solver.dyn_state.dofs.cdof_ang.to_numpy()[:, 0]
    mj_cdof_ang = mj_sim.data.cdof[:, :3]
    assert_allclose(gs_cdof_ang[gs_dofs_idx], mj_cdof_ang[mj_dofs_idx], tol=tol)

    mj_cdof_dot_ang = mj_sim.data.cdof_dot[:, :3]
    gs_cdof_dot_ang = gs_sim.rigid_solver.dyn_state.dofs.cdofd_ang.to_numpy()[:, 0]
    assert_allclose(gs_cdof_dot_ang[gs_dofs_idx], mj_cdof_dot_ang[mj_dofs_idx], tol=tol)

    mj_cdof_dot_vel = mj_sim.data.cdof_dot[:, 3:]
    gs_cdof_dot_vel = gs_sim.rigid_solver.dyn_state.dofs.cdofd_vel.to_numpy()[:, 0]
    assert_allclose(gs_cdof_dot_vel[gs_dofs_idx], mj_cdof_dot_vel[mj_dofs_idx], tol=tol)

    # cinr
    gs_cinr_inertial = gs_sim.rigid_solver.dyn_state.links.cinr_inertial.to_numpy()[:, 0].reshape([-1, 9])[
        :, [0, 4, 8, 1, 2, 5]
    ]
    mj_cinr_inertial = mj_sim.data.cinert[:, :6]  # upper-triangular part
    assert_allclose(gs_cinr_inertial[gs_bodies_idx], mj_cinr_inertial[mj_bodies_idx], tol=tol)
    gs_cinr_pos = gs_sim.rigid_solver.dyn_state.links.cinr_pos.to_numpy()[:, 0]
    mj_cinr_pos = mj_sim.data.cinert[:, 6:9]
    assert_allclose(gs_cinr_pos[gs_bodies_idx], mj_cinr_pos[mj_bodies_idx], tol=tol)
    gs_cinr_mass = gs_sim.rigid_solver.dyn_state.links.cinr_mass.to_numpy()[:, 0]
    mj_cinr_mass = mj_sim.data.cinert[:, 9]
    assert_allclose(gs_cinr_mass[gs_bodies_idx], mj_cinr_mass[mj_bodies_idx], tol=tol)


def simulate_and_check_mujoco_consistency(
    gs_sim, mj_sim, qpos=None, qvel=None, *, tol, num_steps, ignore_constraints=False
):
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
        check_mujoco_data_consistency(
            gs_sim, mj_sim, qvel_prev=qvel_prev, tol=tol, ignore_constraints=ignore_constraints
        )

        # Keep Mujoco and Genesis simulation in sync to avoid drift over time
        mj_sim.data.qpos[mj_qs_idx] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[mj_dofs_idx] = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]
        mj_sim.data.qacc_warmstart[mj_dofs_idx] = gs_sim.rigid_solver.constraint_solver.qacc_ws.to_numpy()[:, 0]
        mj_sim.data.qacc_smooth[mj_dofs_idx] = gs_sim.rigid_solver.dyn_state.dofs.acc_smooth.to_numpy()[:, 0]

        # Backup current velocity
        qvel_prev = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]

        # Do a single simulation step (eventually with substeps for Genesis)
        mujoco.mj_step(mj_sim.model, mj_sim.data)
        gs_sim.scene.step()
        # if gs_sim.scene.visualizer:
        #     gs_sim.scene.visualizer.update()


def rgb_array_to_png_bytes(rgb_arr: np.ndarray | torch.Tensor) -> bytes:
    img = Image.fromarray(tensor_to_array(rgb_arr))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class Crossing(NamedTuple):
    """A genuinely-interpenetrating pair of links among those passed to `get_genuine_interpenetration`."""

    link_a: int
    link_b: int
    depth: float  # separating translation for a crossing, deepest incursion for jammed/contact pairs (metres)


def get_genuine_interpenetration(links, cross_tol=1e-3, n_dir=40, n_bisect=9, is_exact=True):
    """Measure the deepest genuine interpenetration over all link pairs among `links` (each a list of
    `(verts, faces)` collision geoms), as the penetration-depth ground truth a collision algorithm is
    expected to resolve.

    Pairs overlapping within `cross_tol` are contacts reporting their incursion; a crossing pair reads, by
    structure:

    - FULL CONTAINMENT (every vert of one link inside the other): the separation - a swallowed body has no
      free surface to heal toward, so only the extraction resolves (an engulfed sphere reads R + r).
    - PARTIAL ENCLOSURE (a quarter or more of one link's verts buried): min(burial incursion, separation) -
      an oversized apple jammed in a cup reads the wall burial, and a hyper-inflated one enclosing the whole
      cup reads the cheaper translation that slides it off.
    - DENT (the breach touches a single face of each wall): the separation when the breach normal is
      coherent (a directional press retracts by its press depth - the unsigned incursion would escape
      through the far face of a thin wall), min(incursion, separation) when it cancels (a sphere seated all
      around a donut hole reads its 2 mm seat; unseating it is no resolution).
    - PIERCE (some wall carries penetrating verts on both of its faces): the separation, which retracts the
      press or backs the rod out. When separating costs an order of magnitude more than the incursion - a
      bore-fit body poking through its container's wall, where the separation is extraction-scale - the
      resolution is instead the push-back heal: the material protrusion past the pierced wall, marched from
      the antipodal-pair verts along the breach normal (the mean normal of the intruder's penetrating verts,
      coherent only for one-faced breaches), floored by the incursion and capped by the separation.

    Separation means zero material overlap, not unlinked: chain-linked donuts pressed together separate by
    the press depth. Insideness is vertex-sampled via generalized winding numbers (exact on the overlapping
    closed components of convex decompositions, where pseudonormal signed distances break), so each mesh
    must be tessellated finer than its partner's thinnest dimension. Searches run `n_dir` Fibonacci
    directions, bracketed on a geometric ladder and bisected `n_bisect` times.

    Returns `(max_depth, crossings)`: `max_depth` is the largest depth over ALL overlapping pairs, and
    `crossings` lists the pairs deeper than `cross_tol`, deepest first.

    `is_exact` gates the insideness of the ground-truth `overlap` on the exact `igl.winding_number`; set it
    False to use the faster tree approximation, whose platform-dependent error can shift `overlap` on
    borderline geometry (see `inside_of`).
    """
    # Geoms may arrive as torch tensors (verts) or numpy arrays (faces); igl needs float64 verts and int64 faces.
    links = [
        [(tensor_to_array(verts, dtype=np.float64), tensor_to_array(faces, dtype=np.int64)) for verts, faces in geoms]
        for geoms in links
    ]

    # Broad-phase AABB per link, from the full-resolution geoms. A link with no collision geom (e.g. a free-joint
    # base link) gets an inverted AABB so it is rejected against every other link, keeping the link indexing aligned
    # with the caller's list so the returned crossings stay valid indices.
    links_aabb = [
        (
            (np.concatenate([verts for verts, _ in geoms]).min(0), np.concatenate([verts for verts, _ in geoms]).max(0))
            if geoms
            else (np.full(3, np.inf), np.full(3, -np.inf))
        )
        for geoms in links
    ]

    # One concatenated full-resolution mesh per link.
    merged = []
    for geoms in links:
        verts_all, faces_all, offset = [], [], 0
        for verts, faces in geoms:
            verts_all.append(verts)
            faces_all.append(faces + offset)
            offset += len(verts)
        merged.append(
            (np.concatenate(verts_all), np.concatenate(faces_all))
            if verts_all
            else (np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))
        )

    def inside_of(points, verts_other, faces_other, lo_other, hi_other, is_exact=False):
        # Winding-number insideness with an AABB prefilter: a point outside the partner's box can't be inside.
        # `igl.fast_winding_number` is a tree approximation whose error is platform-dependent (BLAS order),
        # not float noise, so it can flip a vertex's insideness across platforms. `is_exact` uses the exact
        # `igl.winding_number` for the one-shot calls that set `overlap`'s ground truth; the hot `separated_at`
        # search keeps the fast one, where a wrong result only costs an extra bisection step.
        is_inside = np.zeros(len(points), dtype=bool)
        is_in_box = ((points >= lo_other) & (points <= hi_other)).all(1)
        if is_in_box.any():
            winding_number = igl.winding_number if is_exact else igl.fast_winding_number
            is_inside[is_in_box] = np.abs(winding_number(verts_other, faces_other, points[is_in_box])) > 0.5
        return is_inside

    # Fibonacci direction sphere and the shift grid (each direction times each probe distance).
    golden = 0.5 * (1.0 + 5.0**0.5)
    i_dir = np.arange(n_dir)
    z_dir = 1.0 - 2.0 * (i_dir + 0.5) / n_dir
    r_dir = np.sqrt(1.0 - z_dir * z_dir)
    azimuth = 2.0 * np.pi * i_dir / golden
    dirs = np.stack([r_dir * np.cos(azimuth), r_dir * np.sin(azimuth), z_dir], axis=1)
    probe = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 4e-2, 8e-2, 1.2e-1])

    max_depth = 0.0
    crossings = []
    for i_la, geoms_a in enumerate(links):
        lo_a, hi_a = links_aabb[i_la]
        for i_lb in range(i_la + 1, len(links)):
            # Broad-phase reject via AABB.
            lo_b, hi_b = links_aabb[i_lb]
            if (lo_a > hi_b).any() or (lo_b > hi_a).any():
                continue
            va, fa = merged[i_la]
            vb, fb = merged[i_lb]

            # Incursion magnitudes from unsigned distances gated by winding-number insideness: the
            # pseudonormal sign of igl.signed_distance is unreliable on the overlapping closed components of
            # convex decompositions, while the generalized winding number stays exact.
            is_inside_a0 = inside_of(va, vb, fb, lo_b, hi_b, is_exact=is_exact)
            is_inside_b0 = inside_of(vb, va, fa, lo_a, hi_a, is_exact=is_exact)
            dist_a0, faces_near_a = igl.signed_distance(va, vb, fb)[:2]
            dist_b0, faces_near_b = igl.signed_distance(vb, va, fa)[:2]
            dist_a0 = np.abs(dist_a0)
            dist_b0 = np.abs(dist_b0)
            depth_a0 = np.where(is_inside_a0, dist_a0, 0.0)
            depth_b0 = np.where(is_inside_b0, dist_b0, 0.0)
            overlap = max(depth_a0.max(), depth_b0.max())
            if overlap <= cross_tol:
                # Contact (or cavity containment touching lightly): report the overlap directly.
                max_depth = max(max_depth, overlap)
                continue

            # Dent-vs-pierce classification: a pierced wall carries penetrating verts on BOTH of its
            # faces - a close pair with near-opposite outward normals whose offset is aligned with them
            # (opposite-normal verts offset sideways sit across a breach rim or a hole, not through a
            # wall). Pair distance is capped: beyond ~12 mm two opposite-normal verts belong to opposite
            # sides of a body, not one wall. Near-touching verts count as penetrating here, since a
            # crossing exactly as deep as the wall leaves the far-face verts ON the partner's surface.
            r_pair = min(3.0 * overlap, 12e-3) + 1e-3

            def antipodal_pairs(pen_pts, pen_normals):
                pairs = [np.empty((0, 2), dtype=np.int64)]
                for i_lo in range(0, len(pen_pts), 256):
                    chunk = slice(i_lo, i_lo + 256)
                    diff = pen_pts[None] - pen_pts[chunk, None]
                    dist2 = (diff**2).sum(-1)
                    dots = pen_normals[chunk] @ pen_normals.T
                    is_aligned = np.abs(np.einsum("knj,kj->kn", diff, pen_normals[chunk])) > 0.7 * np.sqrt(dist2)
                    i_v, j_v = np.nonzero((dist2 <= r_pair**2) & (dots < -0.5) & is_aligned)
                    pairs.append(np.stack([i_v + i_lo, j_v], axis=1))
                return np.concatenate(pairs)

            normals_a = igl.per_vertex_normals(va, fa)
            normals_b = igl.per_vertex_normals(vb, fb)
            normals_faces_a = np.cross(va[fa[:, 1]] - va[fa[:, 0]], va[fa[:, 2]] - va[fa[:, 0]])
            normals_faces_a /= np.maximum(np.linalg.norm(normals_faces_a, axis=1)[:, None], 1e-30)
            normals_faces_b = np.cross(vb[fb[:, 1]] - vb[fb[:, 0]], vb[fb[:, 2]] - vb[fb[:, 0]])
            normals_faces_b /= np.maximum(np.linalg.norm(normals_faces_b, axis=1)[:, None], 1e-30)
            is_pen_a = is_inside_a0 | (dist_a0 <= 0.6e-3)
            is_pen_b = is_inside_b0 | (dist_b0 <= 0.6e-3)
            pairs_a = antipodal_pairs(va[is_pen_a], normals_a[is_pen_a])
            pairs_b = antipodal_pairs(vb[is_pen_b], normals_b[is_pen_b])
            has_pierce = bool(len(pairs_a)) or bool(len(pairs_b))

            def separated_at(offsets):
                # A shift separates the pair when no vert of one link samples inside the other. Deeply
                # overlapping shifts are decided by a strided 1/8 vert subsample first (any hit proves
                # overlap); only the undecided shifts pay the full queries, so the result is exact.
                offsets = np.atleast_2d(offsets)
                is_separated = np.ones(len(offsets), dtype=bool)
                for verts_query, sign, verts_other, faces_other, lo_other, hi_other in (
                    (va[::8], 1.0, vb, fb, lo_b, hi_b),
                    (vb[::8], -1.0, va, fa, lo_a, hi_a),
                    (va, 1.0, vb, fb, lo_b, hi_b),
                    (vb, -1.0, va, fa, lo_a, hi_a),
                ):
                    idx = np.flatnonzero(is_separated)
                    if not len(idx):
                        break
                    points = (verts_query[None] + sign * offsets[idx, None]).reshape(-1, 3)
                    is_inside = inside_of(points, verts_other, faces_other, lo_other, hi_other)
                    is_separated[idx[is_inside.reshape(len(idx), -1).any(1)]] = False
                return is_separated

            # Per-side informed escape directions: a side's coherent mean penetrating normal (a one-faced
            # press or bulge) and the local wall axes of its crossings - the normals of the partner faces
            # nearest its antipodal-pair verts, i.e. the faces of the crossed walls themselves. Wall axes
            # carry a pierce whose own vert normals cancel (a stem's ring verts point radially around the
            # stem, and its antipodal-pair axes span the stem's cross-section - both perpendicular to the
            # through-wall direction), and being face-based they survive sparse vert sampling. The axes are
            # clustered (sign-invariant, folded onto each cluster's dominant eigenvector) because one side
            # may cross several walls at once, and each crossing owns its axis and its witness verts - a
            # dominant unrelated contact would otherwise steer the push-back parallel to the pierced wall.
            # Appended to the Fibonacci grid the directions also remove the direction-grid tilt from the
            # separation.
            dirs_informed_sides = []
            is_press_sides = []
            press_probes = []
            clusters_sides = []
            has_press_normal = False
            for normals_pen, verts_pen, faces_near_pen, normals_faces_other, pairs_side, depths_pen in (
                (normals_a[is_inside_a0], va[is_pen_a], faces_near_a[is_pen_a], normals_faces_b, pairs_a, depth_a0),
                (normals_b[is_inside_b0], vb[is_pen_b], faces_near_b[is_pen_b], normals_faces_a, pairs_b, depth_b0),
            ):
                side_dirs = []
                side_is_press = []
                side_clusters = []
                press_probes.append(None)
                if len(normals_pen):
                    normal_mean = normals_pen.mean(0)
                    norm = np.linalg.norm(normal_mean)
                    if norm > 0.5:
                        side_dirs.append(np.outer([1.0, -1.0], normal_mean / norm))
                        side_is_press += [True, True]
                        has_press_normal = True
                        press_probes[-1] = (verts_pen[depths_pen[depths_pen > 0.0].argmax()], normal_mean / norm)
                if len(pairs_side):
                    verts_idx = np.unique(pairs_side)
                    axes_pairs = normals_faces_other[faces_near_pen[verts_idx]]
                    remaining = np.ones(len(axes_pairs), dtype=bool)
                    for _ in range(3):
                        if remaining.sum() < 4:
                            break
                        axes_rem = axes_pairs[remaining]
                        axis_dominant = np.linalg.eigh((axes_rem[:, :, None] * axes_rem[:, None, :]).mean(0))[1][:, -1]
                        in_cluster = remaining & (np.abs(axes_pairs @ axis_dominant) > 0.8)
                        if in_cluster.sum() >= 4:
                            axes_cluster = axes_pairs[in_cluster].copy()
                            axes_cluster[axes_cluster @ axis_dominant < 0.0] *= -1.0
                            axis_mean = axes_cluster.mean(0)
                            axis_mean /= np.linalg.norm(axis_mean)
                            side_dirs.append(np.outer([1.0, -1.0], axis_mean))
                            side_is_press += [False, False]
                            side_clusters.append((axis_mean, verts_pen[verts_idx[in_cluster]]))
                        remaining &= ~in_cluster
                dirs_informed_sides.append(np.concatenate(side_dirs) if side_dirs else np.empty((0, 3)))
                is_press_sides.append(np.array(side_is_press, dtype=bool))
                clusters_sides.append(side_clusters)
            dirs_search = np.concatenate([dirs, *dirs_informed_sides])
            is_press_dir = np.concatenate([np.zeros(len(dirs), dtype=bool), *is_press_sides])

            # Smallest separating translation: bracket each direction's transition on the geometric ladder,
            # then bisect in lockstep; np.inf when no direction separates within the ladder.
            i_first = np.full(len(dirs_search), -1)
            is_active = np.ones(len(dirs_search), dtype=bool)
            for i_r in range(len(probe)):
                if not is_active.any():
                    break
                idx = np.flatnonzero(is_active)
                is_separated = separated_at(dirs_search[idx] * probe[i_r])
                i_first[idx[is_separated]] = i_r
                is_active[idx[is_separated]] = False
            seps_dir = np.full(len(dirs_search), np.inf)
            idx = np.flatnonzero(i_first >= 0)
            if len(idx):
                lo = np.where(i_first[idx] > 0, probe[np.maximum(i_first[idx] - 1, 0)], 0.0)
                hi = probe[i_first[idx]]
                for _ in range(n_bisect):
                    mid = 0.5 * (lo + hi)
                    is_separated = separated_at(dirs_search[idx] * mid[:, None])
                    lo = np.where(is_separated, lo, mid)
                    hi = np.where(is_separated, mid, hi)
                seps_dir[idx] = hi
            separation = seps_dir.min() if len(seps_dir) else np.inf
            seps_press_sides = []
            i_dir = len(dirs)
            for is_press_side in is_press_sides:
                n_side = len(is_press_side)
                seps_side = seps_dir[i_dir : i_dir + n_side][is_press_side]
                seps_press_sides.append(seps_side.min() if len(seps_side) else np.inf)
                i_dir += n_side

            if is_inside_a0.all() or is_inside_b0.all():
                # Full containment: a body with every vert inside the partner has no free surface to heal
                # toward - burial is meaningless and only the extraction resolves (an engulfed sphere reads
                # R + r, however deep it is buried).
                depth = separation if np.isfinite(separation) else overlap
            elif max(is_inside_a0.mean(), is_inside_b0.mean()) >= 0.25:
                # Partial enclosure: unseating or extracting resolves nothing a solver would report - the
                # depth is the burial incursion, unless plainly separating is cheaper.
                depth = min(overlap, separation)
            elif not has_pierce:
                # Dent with a coherent breach normal: a directional press, resolved by retracting along
                # that normal - the separation RESTRICTED to the press directions, not the global minimum
                # (for a presser inside its container the global minimum is the extraction through the
                # mouth, which resolves nothing a solver would report). The retraction is only a press
                # resolution while it fits the wall it presses: a press that produced no pierce cannot run
                # deeper than the local wall thickness, so a larger directional separation means the
                # retraction is blocked (a body wedged inside its container escaping diagonally through
                # the mouth) and the depth is the incursion, as for an incoherent all-around seat.
                depth_press = np.inf
                s_probe = np.arange(1, 121) * 1e-3
                for press_probe, sep_press, (verts_other, faces_other, lo_other, hi_other) in zip(
                    press_probes,
                    seps_press_sides,
                    ((vb, fb, lo_b, hi_b), (va, fa, lo_a, hi_a)),
                ):
                    if press_probe is None or not np.isfinite(sep_press):
                        continue
                    vert_deep, normal_press = press_probe
                    thickness = 0.0
                    for sign in (1.0, -1.0):
                        pts = vert_deep[None] + sign * normal_press * s_probe[:, None]
                        is_inside_probe = inside_of(pts, verts_other, faces_other, lo_other, hi_other)
                        thickness += s_probe[-1] if is_inside_probe.all() else s_probe[np.argmin(is_inside_probe)]
                    if sep_press <= thickness + 2e-3:
                        depth_press = min(depth_press, sep_press)
                depth = depth_press if np.isfinite(depth_press) else min(overlap, separation)
            elif separation <= 10.0 * overlap:
                # Ordinary pierce: presses, rods, beams - the separation IS the resolution (it retracts the
                # press or backs the rod out; the breach normals of fully-crossed walls cancel anyway).
                depth = separation
            else:
                # Extraction-scale containment (separating costs an order of magnitude more than the
                # incursion: a bore-fit body poking through its container's wall): the resolution is the
                # push-back heal - the retreat along the wall axis that brings all of the crossing link's
                # material back through the pierced wall entirely. Projection onto the axis gives it
                # exactly: the retreat for a sign is the span from the wall's far face (the cluster's
                # extreme projection against that sign, so walls stacked along one axis - a beam through
                # both tube walls - are cleared together) to the link's farthest material (its stem tip - a
                # compact nub; a crossed plate measures its own half-extent instead and defers to the
                # separation via the near-parity gate below); the cheaper sign is the side to push back
                # through. Rigid translation feasibility is no
                # yardstick here: a bore-fit body cannot actually translate (any push re-presses the far
                # side of the bore), yet the wall crossing is still healed by exactly the protrusion.
                protrusions = []
                for i_side, side_clusters in enumerate(clusters_sides):
                    if not side_clusters:
                        continue
                    verts_side = (va, vb)[i_side]
                    heal_side = 0.0
                    for axis, verts_cluster in side_clusters:
                        proj_cluster = verts_cluster @ axis
                        proj_side = verts_side @ axis
                        retreat = min(proj_side.max() - proj_cluster.min(), proj_cluster.max() - proj_side.min())
                        heal_side = max(heal_side, retreat)
                    protrusions.append(heal_side)
                if not protrusions:
                    # No side has a coherent escape direction: crossings pressed from every side, a
                    # seat-family jam whose extraction resolves nothing a solver would report.
                    depth = min(overlap, separation)
                else:
                    heal = max(overlap, min(protrusions))
                    if heal < 0.8 * separation:
                        depth = heal
                    else:
                        # Near-parity means the crossing genuinely resolves by sliding out; the separation
                        # is the exact value while the marched heal carries the 1 mm grid quantization.
                        depth = separation
                if not np.isfinite(depth):
                    depth = overlap
            max_depth = max(max_depth, depth)
            if depth > cross_tol:
                crossings.append(Crossing(i_la, i_lb, depth))

    crossings.sort(key=lambda crossing: crossing.depth, reverse=True)
    return max_depth, crossings


def display_collision_pairs(pairs):
    """Open a self-contained interactive 3D viewer of colliding link pairs in the default browser.

    Each entry of `pairs` is `(geoms_a, geoms_b, label)`, where each `geoms` is a list of `(verts, faces)`
    collision meshes (torch or numpy); the geoms of a link are merged and the two links drawn blue and red,
    with a dropdown to switch between pairs. Each bead in the hover overlay marks a surface the cursor ray
    crosses (coloured by link, back walls flagged with a cross); a digit key (numpad or top row) picks the
    N-th surface directly, shift+scroll steps to enclosed surfaces, and clicking two beads reads their
    distance in millimetres.
    """
    data = []
    for geoms_a, geoms_b, label in pairs:
        entry = {"label": label}
        for key, geoms in (("a", geoms_a), ("b", geoms_b)):
            verts_all, faces_all, offset = [], [], 0
            for verts, faces in geoms:
                verts = tensor_to_array(verts, dtype=np.float64)
                faces = tensor_to_array(faces, dtype=np.int64)
                verts_all.append(verts)
                faces_all.append(faces + offset)
                offset += len(verts)
            entry[key] = {"v": np.concatenate(verts_all).tolist(), "f": np.concatenate(faces_all).tolist()}
        data.append(entry)
    template = (Path(__file__).parent / "mesh_pairs_viewer.html").read_text()
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as file:
        file.write(template.replace("__DATA__", json.dumps(data)))
    webbrowser.open(f"file://{file.name}")


def pprint_oneline(data, delimiter, digits=None):
    msg_items = []
    for key, value in data.items():
        if isinstance(value, Enum):
            value = value.name
        if digits is not None and isinstance(value, (numbers.Real, np.floating)):
            value = f"{value:.{digits}f}"
        msg_item = "=".join((key, str(value)))
        msg_items.append(msg_item)
    return delimiter.join(msg_items)
