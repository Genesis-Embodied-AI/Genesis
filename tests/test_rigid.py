import subprocess
import unittest
import numpy as np
import genesis as gs


class TestRigid(unittest.TestCase):

    def test_simple(self):
        add = 1 + 1
        assert add == 2

    def test_forward_dynamics(self):
        gs.init(backend=gs.gpu, logging_level="warning", seed=0)
        scene = gs.Scene(
            show_viewer=False,
            rigid_options=gs.options.RigidOptions(
                dt=0.01,
                enable_collision=True,
                enable_joint_limit=True,
            ),
        )

        ########################## entities ##########################
        scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        )
        robot = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/anymal_c/urdf/anymal_c.urdf",
                pos=(0, 0, 1.8),
            ),
        )
        ########################## build ##########################
        n_envs = 0
        scene.build(n_envs=n_envs)
        scene.reset()

        qvel = robot.get_dofs_velocity()
        qvel = np.ones(len(qvel))
        robot.set_dofs_velocity(qvel)

        for i in range(10):
            scene.step()

        qpos = robot.get_qpos()
        qpos_ref = np.array(
            [
                0.0992,
                0.1016,
                1.8430,
                0.9957,
                0.0582,
                0.0538,
                0.0476,
                0.0795,
                0.0762,
                0.0843,
                0.0813,
                0.0730,
                0.0556,
                0.0907,
                0.0739,
                0.0695,
                0.0684,
                0.0621,
                0.0640,
            ]
        )
        diff = np.abs(qpos.cpu().numpy() - qpos_ref).max()
        print("diff: ", diff)
        gs.destroy()
        assert diff < 1e-4


if __name__ == "__main__":
    unittest.main()
