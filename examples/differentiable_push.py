import argparse

import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
            requires_grad=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 0.55),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -0.15, 2.42),
            camera_lookat=(0.5, 0.5, 0.1),
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    stick = scene.add_entity(
        material=gs.materials.Tool(friction=8.0),
        morph=gs.morphs.Mesh(
            file="meshes/stirrer.obj",
            scale=0.6,
            pos=(0.5, 0.5, 0.05),
            euler=(90.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 1.0, 1.0, 1.0),
        ),
    )
    obj1 = scene.add_entity(
        material=gs.materials.MPM.Elastic(rho=500),
        morph=gs.morphs.Box(
            lower=(0.2, 0.1, 0.05),
            upper=(0.4, 0.3, 0.15),
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.9, 0.9, 1.0),
        ),
        vis_mode="particle",
    )
    obj2 = scene.add_entity(
        material=gs.materials.MPM.Elastic(rho=500),
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            pos=(0.4, 0.55, 0.056),
            scale=0.07,
            euler=(90.0, 0.0, 90.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.8, 0.2, 1.0),
        ),
        vis_mode="particle",
    )

    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        pos=(1.5, 0.5, 2.42),
        lookat=(0.5, 0.5, 0.1),
        fov=30,
        GUI=True,
    )
    cam_1 = scene.add_camera(
        pos=(-3.0, 1.5, 2.0),
        lookat=(0.5, 0.5, 0.1),
        fov=30,
        GUI=True,
    )

    ########################## build ##########################
    scene.build()

    ########################## forward + backward twice ##########################
    for _ in range(2):
        scene.reset()
        horizon = 150
        init_pos = gs.tensor([0.3, 0.1, 0.28], requires_grad=True)

        # forward pass
        print("forward")
        timer = gs.tools.Timer()
        stick.set_position(init_pos)
        v_obj1_init = gs.tensor([0.0, -1.0, 0.0], requires_grad=True)
        obj1.set_velocity(v_obj1_init)
        pos_obj1_init = gs.tensor([0.3, 0.3, 0.1], requires_grad=True)
        obj1.set_position(pos_obj1_init)
        loss = 0
        v_list = []
        w_list = []
        for i in range(horizon):
            v_i = gs.tensor([0.0, 1.0, 0.0], requires_grad=True)
            # w_i = gs.tensor([2.0, 0.0, 0.0], requires_grad=True)
            # stick.set_velocity(vel=v_i, ang=w_i)
            stick.set_velocity(vel=v_i)
            v_list.append(v_i)
            # w_list.append(w_i)

            scene.step()
            # img0 = cam_0.render()
            # img1 = cam_1.render()

            # you can use a scene_state
            if i == 25:
                # compute loss
                goal = gs.tensor([0.5, 0.8, 0.05])
                mpm_particles = scene.get_state().solvers_state[3]
                loss += torch.pow(mpm_particles.pos[mpm_particles.active == 1] - goal, 2).sum()

            # you can also use an entity's state
            if i == horizon - 1:
                # compute loss
                goal = gs.tensor([0.5, 0.8, 0.05])
                state = obj1.get_state()
                loss += torch.pow(state.pos - goal, 2).sum()

        timer.stamp("forward took: ")
        # backward pass
        print("backward")
        loss.backward()  # this lets gradient flow all the way back to tensor input
        timer.stamp("backward took: ")
        for v_i in v_list:
            print(v_i.grad)
            v_i.zero_grad()
        # for w_i in w_list:
        #     print(w_i.grad)
        #     w_i.zero_grad()
        print(init_pos.grad)
        print(v_obj1_init.grad)
        print(pos_obj1_init.grad)
        init_pos.zero_grad()


if __name__ == "__main__":
    main()
