import argparse
import os
import numpy as np
import time
from PIL import Image
import genesis as gs
from typing import List
import imageio
import cv2
import pickle
import torch
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


def save_2D_to_video(list_frames: List, filepath, fps: int = 60):

    # Convert frames to proper image format if needed (e.g., uint8)
    normalized_frames = [(1 - (frame - np.min(frame)) / (np.max(frame) - np.min(frame))) * 255 for frame in list_frames]
    converted_frames = [frame.astype(np.uint8) for frame in normalized_frames]

    # Create a writer object to write video files at 30 fps
    writer = imageio.get_writer(filepath, fps=fps)

    # Write frames to video
    for frame in converted_frames:
        writer.append_data(frame)

    # Close the writer to finish writing the video file
    writer.close()


def save_3D_to_video(list_frames: List, filepath, fps: int = 60):

    # Convert frames to proper image format if needed (e.g., uint8)
    converted_frames = [frame.astype(np.uint8) for frame in list_frames]

    # Create a writer object to write video files at 30 fps
    writer = imageio.get_writer(filepath, fps=fps)

    # Write frames to video
    for frame in converted_frames:
        # writer.append_data(frame)
        writer.append_data(frame[..., [2, 1, 0]])

    # Close the writer to finish writing the video file
    writer.close()


def save_tensor_to_image(frame: np.ndarray, filepath: str, normalize=False):
    if normalize:
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255

    frame = frame.astype(np.uint8)
    if len(frame.shape) == 3:
        cv2.imwrite(filepath, frame[..., [2, 1, 0]])
    else:
        cv2.imwrite(filepath, frame)
    return


def point_to_box_distance(point, box_center, box_size):
    """Calculate the minimum distance from a point to a box."""
    box_min = box_center - box_size / 2
    box_max = box_center + box_size / 2
    max_dist = np.maximum(0, np.maximum(box_min - point, point - box_max))
    return np.linalg.norm(max_dist)


def check_box_box_overlap(box1, box2):
    """Check overlap between two boxes."""
    pos1, size1, radi1, height1, shape1 = box1
    pos2, size2, radi2, height2, shape2 = box2
    half_size1 = size1 / 2
    half_size2 = size2 / 2
    for i in range(3):
        if abs(pos1[i] - pos2[i]) > (half_size1[i] + half_size2[i]):
            return False
    return True


def check_box_sphere_overlap(box1, sphere2):
    """Check if a box and a sphere overlap."""
    pos1, size1, radi1, height1, shape1 = box1
    pos2, size2, radi2, height2, shape2 = sphere2
    distance = point_to_box_distance(pos2, pos1, size1)
    return distance < radi2


def check_box_cylinder_overlap(box1, cylinder2):
    """Check if a box and a vertically aligned cylinder overlap."""
    pos1, size1, radi1, height1, shape1 = box1
    pos2, size2, radi2, height2, shape2 = cylinder2
    # Check xy-plane overlap first
    distance_xy = np.linalg.norm(pos1[:2] - pos2[:2])
    if distance_xy < (radi2 + max(size1[:2]) / 2):
        # Check z-axis overlap
        box_z_min = pos1[2] - size1[2] / 2
        box_z_max = pos1[2] + size1[2] / 2
        cylinder_z_min = pos2[2] - height2 / 2
        cylinder_z_max = pos2[2] + height2 / 2
        return max(box_z_min, cylinder_z_min) < min(box_z_max, cylinder_z_max)
    return False


def check_sphere_sphere_overlap(sphere1, sphere2):
    pos1, size1, radi1, height1, shape1 = sphere1
    pos2, size2, radi2, height2, shape2 = sphere2
    """ Check if two spheres overlap. """
    distance = np.linalg.norm(pos1 - pos2)
    return distance < (radi1 + radi2)


def check_sphere_cylinder_overlap(sphere1, cylinder2):
    """Check if a sphere and a vertically aligned cylinder overlap."""
    pos1, size1, radi1, height1, shape1 = sphere1
    pos2, size2, radi2, height2, shape2 = cylinder2
    # Check distance in the xy-plane
    distance_xy = np.linalg.norm(pos1[:2] - pos2[:2])
    if distance_xy < (radi1 + radi2):
        # Check z-axis overlap
        sphere_z_min = pos1[2] - radi1
        sphere_z_max = pos1[2] + radi1
        cylinder_z_min = pos2[2] - height2 / 2
        cylinder_z_max = pos2[2] + height2 / 2
        return max(sphere_z_min, cylinder_z_min) < min(sphere_z_max, cylinder_z_max)
    return False


def check_cylinder_cylinder_overlap(cylinder1, cylinder2):
    """Check if two cylinders overlap."""
    pos1, size1, radi1, height1, shape1 = cylinder1
    pos2, size2, radi2, height2, shape2 = cylinder2
    distance_xy = np.linalg.norm(pos1[:2] - pos2[:2])
    if distance_xy < (radi1 + radi2):
        # Check z-axis overlap
        z1_min = pos1[2] - height1 / 2
        z1_max = pos1[2] + height1 / 2
        z2_min = pos2[2] - height2 / 2
        z2_max = pos2[2] + height2 / 2
        return max(z1_min, z2_min) < min(z1_max, z2_max)
    return False


def check_overlap(new_meta, list_existing):
    """Check if the new box overlaps with any existing box."""
    overlap_functions = {
        ("Box", "Box"): lambda obj1, obj2: check_box_box_overlap(obj1, obj2),
        ("Box", "Sphere"): lambda obj1, obj2: check_box_sphere_overlap(obj1, obj2),
        ("Sphere", "Box"): lambda obj1, obj2: check_box_sphere_overlap(obj2, obj1),
        ("Box", "Cylinder"): lambda obj1, obj2: check_box_cylinder_overlap(obj1, obj2),
        ("Cylinder", "Box"): lambda obj1, obj2: check_box_cylinder_overlap(obj2, obj1),
        ("Sphere", "Sphere"): lambda obj1, obj2: check_sphere_sphere_overlap(obj1, obj2),
        ("Sphere", "Cylinder"): lambda obj1, obj2: check_sphere_cylinder_overlap(obj1, obj2),
        ("Cylinder", "Sphere"): lambda obj1, obj2: check_sphere_cylinder_overlap(obj2, obj1),
        ("Cylinder", "Cylinder"): lambda obj1, obj2: check_cylinder_cylinder_overlap(obj1, obj2),
    }
    overlap_all = False
    for exist_meta in list_existing:
        key = (exist_meta[-1], new_meta[-1])
        if key in overlap_functions:
            func = overlap_functions[key]
            if func(exist_meta, new_meta):
                overlap_all = True

    return overlap_all


def add_entity(scene, material, obj_shape: str, pos: np.ndarray, size: np.ndarray, radius: float, height: float):

    if obj_shape == "Box":
        obj = gs.morphs.Box(
            pos=pos,
            size=size,
        )
    elif obj_shape == "Sphere":
        obj = gs.morphs.Sphere(
            pos=pos,
            radius=radius,
        )
    elif obj_shape == "Cylinder":
        obj = gs.morphs.Cylinder(pos=pos, radius=radius, height=height)
    else:
        raise ValueError(f"Invalid shape: {obj_shape}")

    added_obj = scene.add_entity(material=material, morph=obj)
    return added_obj


def create_parameter(
    list_occupied: List,
    obj_shape: str,
):

    max_attempts = 100
    success = False
    new_size = None
    new_radius = None
    new_height = None
    for _ in range(max_attempts):
        new_pos = np.random.rand(3) * 10 + 2
        if obj_shape == "Box":
            new_size = np.random.rand(3) * 4
        else:
            new_radius = np.random.rand() * 2
            if obj_shape == "Cylinder":
                new_height = np.random.rand() * 3
        cur_meta = [new_pos, new_size, new_radius, new_height, obj_shape]
        if not check_overlap(cur_meta, list_occupied):
            success = True
            list_occupied.append(cur_meta)
            break
    return success, list_occupied, cur_meta


def add_force(rigid_solver, cur_idx, f_scale: int = 100, dim: int = 2):
    cur_pos = rigid_solver.get_links_pos(cur_idx)
    cur_pos[:, :, dim] -= 1
    force = f_scale * cur_pos
    rigid_solver.apply_links_external_force(
        force=force,
        links_idx=cur_idx,
    )
    return


def add_torque(rigid_solver, cur_idx, rot_direction, rot_scale: int = 5, dim: int = 2):
    torque = [0, 0, 0]
    torque[dim] = rot_direction * rot_scale
    torque = torch.tensor(torque).unsqueeze(0).unsqueeze(0)
    rigid_solver.apply_links_external_torque(
        torque=torque,
        links_idx=cur_idx,
    )
    return


def set_velocity(obj, v_array):
    v_init = gs.tensor(
        v_array,
    )
    obj.set_dofs_velocity(v_init)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--num_obj", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--output_folder", type=str, default="output_folder")
    args = parser.parse_args()

    num_obj = args.num_obj
    seed = args.seed
    root_folder = "."
    os.makedirs(root_folder, exist_ok=True)
    output_folder = args.output_folder
    output_folder = os.path.join(root_folder, output_folder)
    os.makedirs(output_folder, exist_ok=True)

    ########################## init ##########################
    gs.init(seed=seed, precision="32", logging_level="debug")

    ########################## create a scene ##########################
    # define scenario
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -10.0),
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=False,
            show_world_frame=False,
            show_link_frame=False,
            segmentation_level="entity",  # geom not supported yet
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_fov=40,
            max_FPS=60,
        ),
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),  # using rasterizer for camera rendering
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    B = 1

    list_shape = ["Box", "Sphere", "Cylinder"]
    dict_obj_idx = dict()
    list_occupied = []
    for obj_idx in range(num_obj):
        friction_ratio = np.random.rand()
        rigid_material = gs.materials.Rigid(needs_coup=True, coup_friction=friction_ratio)
        obj_shape = np.random.choice(list_shape)
        success, list_occupied, new_para = create_parameter(
            list_occupied=list_occupied,
            obj_shape=obj_shape,
        )
        if success:
            new_pos, new_size, new_radius, new_height, _ = new_para
            added_obj = add_entity(
                scene,
                material=rigid_material,
                obj_shape=obj_shape,
                pos=new_pos,
                size=new_size,
                radius=new_radius,
                height=new_height,
            )
            dict_obj_idx[obj_idx] = [added_obj, obj_shape]

    # set camera pos to the highest place
    max_z = 0
    max_x = 0
    max_y = 0
    for meta in list_occupied:
        new_pos, new_size, new_radius, new_height, new_shape = meta
        if new_shape == "Box":
            max_x = max(max_x, new_pos[0] + new_size[0] / 2)
            max_y = max(max_y, new_pos[1] + new_size[1] / 2)
            max_z = max(max_z, new_pos[2] + new_size[2] / 2)
        elif new_shape == "Sphere":
            max_x = max(max_x, new_pos[0] + new_radius)
            max_y = max(max_y, new_pos[1] + new_radius)
            max_z = max(max_z, new_pos[2] + new_radius)
        elif new_shape == "Cylinder":
            max_x = max(max_x, new_pos[0] + new_radius)
            max_y = max(max_y, new_pos[1] + new_radius)
            max_z = max(max_z, new_pos[2] + new_height / 2)

    max_x = np.ceil(max_x + 20)
    max_y = np.ceil(max_y + 20)
    max_z = np.ceil(max_z + 3)
    cam = scene.add_camera(
        res=(640, 480),
        pos=(max_x, max_y, max_z),
        lookat=(0.0, 0.0, 0.0),
        fov=35,
        GUI=False,
    )

    scene.build(n_envs=B, env_spacing=(1.0, 1.0))

    for obj_idx in range(num_obj):
        added_obj = dict_obj_idx[obj_idx][0]
        added_obj.set_friction_ratio(
            friction_ratio=0.5 + torch.rand(scene.n_envs, added_obj.n_links),
            ls_idx_local=np.arange(0, added_obj.n_links),
        )

    for solver in scene.sim.solvers:
        if not isinstance(solver, RigidSolver):
            continue
        rigid_solver = solver

    # cam.start_recording()
    horizon = 500
    recorded_depth = []
    recorded_seg = []
    recorded_seg_id = []

    # number of entity
    num_entities = len(scene.rigid_solver.entities)
    # create segments colors
    seg_idxc_to_color = np.random.randint(0, 255, (num_entities, 3), dtype=np.uint8)
    # set background colors
    seg_idxc_to_color[0, :] = 0
    link_idx = list(range(1, num_entities))
    list_changes = []
    for i in link_idx:
        obj_shape = dict_obj_idx[i - 1][1]
        if obj_shape not in ("Sphere", "Cylinder"):
            action = np.random.choice(["force", "torque"])
        else:
            action = "force"

        if action == "force":
            # f_scale = np.random.rand() * 50 - 25
            f_scale = np.random.rand() * 50 + 20
            list_changes.append(["force", i, f_scale])
        else:
            # rot_scale = np.random.rand() * 20 - 10
            rot_scale = np.random.rand() * 20 + 20
            list_changes.append(["torque", i, rot_scale])

    os.makedirs(output_folder, exist_ok=True)
    rotation_direction = 1

    for i in range(horizon):

        if i == 0:
            # for obj_id in dict_obj_idx.keys():
            #     if np.random.rand() < 0.5:
            #         new_v = np.random.rand(3)*2
            #         set_velocity(dict_obj_idx[obj_id][0], new_v)
            for init_changes in list_changes:
                action, idx, a_scale = init_changes
                if action == "force":
                    cur_idx = [idx]
                    dim = np.random.randint(3)
                    add_force(rigid_solver, cur_idx, f_scale=a_scale, dim=dim)
                else:
                    cur_idx = [idx]
                    add_torque(rigid_solver, cur_idx, rot_direction=rotation_direction, rot_scale=a_scale, dim=dim)

        scene.step()
        rgb, depth, segmentation, normal = cam.render(depth=True, segmentation=True, normal=False)
        # why there is -1
        filtered_segmentation = np.where(segmentation == -1, 0, segmentation)
        colored_segmentation = seg_idxc_to_color[filtered_segmentation]

        # recorded_depth.append(depth)
        recorded_seg.append(colored_segmentation)
        recorded_seg_id.append(segmentation)

        if (i + 50) % 100 == 0:
            rotation_direction *= -1

    # output normal rgb video
    # cam.stop_recording(save_to_filename=f'{output_folder}/video.mp4', fps=60)
    print(list_changes)

    # output depth & segmentation video
    # save_2D_to_video(recorded_depth, f'{output_folder}/depth.mp4', fps=60)
    save_3D_to_video(recorded_seg, f"{output_folder}/segments.mp4", fps=60)
    # with open(f'{output_folder}/segments_id.pkl', 'wb') as f:
    #     pickle.dump(recorded_seg_id, f)


if __name__ == "__main__":
    main()
