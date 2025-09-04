
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
import types
import genesis as gs
import numpy as np
from pid import PIDcontroller
from odom import Odom
from mavlink_sim import rc_command

from genesis.utils.geom import trans_quat_to_T, transform_quat_by_quat, transform_by_trans_quat

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class Genesis_env :
    def __init__(
            self, 
            env_config, 
            flight_config,
            num_envs=None,
        ):
        
        # configs
        self.env_config = env_config
        self.flight_config = flight_config

        # bool switches
        self.render_cam = self.env_config["render_cam"]
        self.use_rc = self.env_config["use_rc"]
        self.use_ros = self.env_config.get("use_ros", False)

        # flight
        self.controller = env_config["controller"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if num_envs is None:
            self.num_envs = self.env_config.get("num_envs", 1)
        else:
            self.num_envs = num_envs
        self.dt = self.env_config.get("dt", 0.01)           # default sim env update in 100hz
        self.cam_quat = torch.tensor(self.env_config.get("cam_quat", [0.5, 0.5, -0.5, -0.5]), device=self.device, dtype=gs.tc_float).expand(self.num_envs, -1)
        
        self.rendered_env_num = self.num_envs if self.render_cam else min(4, self.num_envs)
        
        # create scene
        self.scene = gs.Scene(
            sim_options = gs.options.SimOptions(dt = self.dt, substeps = 1),
            viewer_options = gs.options.ViewerOptions(
                max_FPS = self.env_config.get("max_vis_FPS", 15),
                camera_pos = (-3.0, 0.0, 3.0),
                camera_lookat = (0.0, 0.0, 1.0),
                camera_fov = 40,
            ),
            vis_options = gs.options.VisOptions(
                show_world_frame = False,
                rendered_envs_idx = list(range(self.rendered_env_num)),
                env_separate_rigid = True,
                shadow = False,
            ),
            rigid_options = gs.options.RigidOptions(
                dt = self.dt,
                constraint_solver = gs.constraint_solver.Newton,
                enable_collision = True,
                enable_joint_limit = True,
            ),
            show_viewer = self.env_config["show_viewer"],
        )

        # add plane (ground)
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # add drone
        drone = gs.morphs.Drone(
            file="examples/drone/controller/drone_urdf/drone.urdf", 
            pos=self.env_config["drone_init_pos"], 
            euler=(0, 0, 0),
            default_armature=self.flight_config.get("motor_inertia", 2.6e-07)
        )
        self.drone = self.scene.add_entity(drone)
        
        # set viewer
        if self.env_config["viewer_follow_drone"] is True:
            self.scene.viewer.follow_entity(self.drone)  # follow drone   
        
        # add odom for drone
        self.set_drone_imu()

        # add controller for drone
        self.set_drone_controller()

        # add drone camera
        self.set_drone_camera()

        # set target for vis
        self.set_target_phere_for_vis()

        # build world
        self.scene.build(n_envs = self.num_envs)

        # init
        self.drone_init_pos = self.drone.get_pos()
        self.drone_init_quat = self.drone.get_quat()
        self.drone.set_dofs_damping(torch.tensor([0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4]))  # Set damping to a small value to avoid numerical instability


    def step(self, action=None): 
        self.scene.step()
        self.drone.cam.set_FPV_cam_pos()
        if self.render_cam:
            self.drone.cam.depth = self.drone.cam.render(rgb=True, depth=True)[1]   # [1] is idx of depth img
        self.drone.controller.step(action)


    def set_drone_imu(self):
        odom = Odom(
            num_envs = self.num_envs,
            device = torch.device("cuda")
        )
        odom.set_drone(self.drone)
        setattr(self.drone, 'odom', odom) 

    def set_drone_camera(self):
        if (self.env_config.get("use_FPV_camera", False)):
            cam = self.scene.add_camera(
                res=self.env_config["cam_res"],
                pos=(-3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=58,
                GUI=True,
            )
        def set_FPV_cam_pos(self):
            self.cam.set_pose(
            transform = trans_quat_to_T(trans = self.get_pos(), 
                                        quat = transform_quat_by_quat(self.cam.cam_quat, self.odom.body_quat))[0].cpu().numpy()
        )
        setattr(cam, 'cam_quat', self.cam_quat)  
        setattr(cam, 'set_FPV_cam_pos', types.MethodType(set_FPV_cam_pos, self.drone))
        depth: np.ndarray = None
        setattr(cam ,'depth', depth)
        setattr(self.drone, 'cam', cam)

    def set_drone_controller(self):
        pid = PIDcontroller(
            num_envs = self.num_envs, 
            rc_command = rc_command,
            odom = self.drone.odom, 
            config = self.flight_config,
            device = torch.device("cuda"),
            use_rc = self.use_rc,
            controller = self.controller,
        )
        pid.set_drone(self.drone)
        setattr(self.drone, 'controller', pid)      

    def set_target_phere_for_vis(self):
        if self.env_config["vis_waypoints"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="genesis/assets/meshes/sphere.obj",
                    scale=0.02,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None