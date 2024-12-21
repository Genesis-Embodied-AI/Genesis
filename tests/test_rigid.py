import unittest
import numpy as np
import genesis as gs


class TestRigid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup the testing environment."""
        gs.init(backend=gs.gpu, logging_level="warning", seed=0)
        cls.scene = gs.Scene(
            show_viewer=False,
            rigid_options=gs.options.RigidOptions(
                dt=0.01,
                enable_collision=True,
                enable_joint_limit=True,
            ),
        )
        # Add entities
        cls.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        cls.robot = cls.scene.add_entity(
            gs.morphs.URDF(file="urdf/anymal_c/urdf/anymal_c.urdf", pos=(0, 0, 1.8))
        )
        # Build and reset the scene
        cls.scene.build(n_envs=0)
        cls.scene.reset()

    @classmethod
    def tearDownClass(cls):
        """Teardown the testing environment."""
        gs.destroy()

    def test_simple(self):
        """Test basic addition functionality."""
        self.assertEqual(1 + 1, 2)

    def test_forward_dynamics(self):
        """Test forward dynamics with the robot."""
        # Set robot velocity
        qvel = np.ones(len(self.robot.get_dofs_velocity()))
        self.robot.set_dofs_velocity(qvel)

        # Run simulation
        for _ in range(10):
            self.scene.step()

        # Get and compare the robot's position with the reference
        qpos = self.robot.get_qpos()
        qpos_ref = np.array(
            [
                0.0992, 0.1016, 1.8430, 0.9957, 0.0582, 0.0538, 0.0476, 0.0795,
                0.0762, 0.0843, 0.0813, 0.0730, 0.0556, 0.0907, 0.0739, 0.0695,
                0.0684, 0.0621, 0.0640,
            ]
        )
        diff = np.abs(qpos.cpu().numpy() - qpos_ref).max()

        # Print difference and assert if it's within acceptable range
        print(f"Max difference: {diff}")
        self.assertLess(diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
