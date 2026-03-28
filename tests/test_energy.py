"""Tests for RigidEntity energy calculation methods (get_kinetic_energy, get_potential_energy, get_total_energy)."""

import pytest
import torch

import genesis as gs


class TestEnergyFreeFall:
    """Test energy methods using free-falling rigid bodies (no collision)."""

    @pytest.mark.precision("64")
    def test_stationary_object_has_zero_kinetic_energy(self):
        """A stationary object should have zero kinetic energy."""
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)))
        sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0, 0, 2.0)))
        scene.build()

        ke = sphere.get_kinetic_energy()
        assert ke.item() == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.precision("64")
    def test_potential_energy_proportional_to_height(self):
        """PE should equal m * g * h for an object at height h with gravity along -z."""
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)))
        sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0, 0, 5.0)))
        scene.build()

        mass = sphere.get_mass()
        pe = sphere.get_potential_energy()
        expected_pe = mass * 9.81 * 5.0
        assert pe.item() == pytest.approx(expected_pe, rel=1e-6)

    @pytest.mark.precision("64")
    def test_total_energy_equals_sum(self):
        """Total energy should equal kinetic + potential energy."""
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)))
        sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0, 0, 3.0)))
        scene.build()

        # Step a few times so the object has both KE and PE
        for _ in range(10):
            scene.step()

        ke = sphere.get_kinetic_energy()
        pe = sphere.get_potential_energy()
        te = sphere.get_total_energy()
        assert te.item() == pytest.approx((ke + pe).item(), abs=1e-12)

    @pytest.mark.precision("64")
    def test_energy_conservation_free_fall(self):
        """Total energy should be approximately conserved during free fall (no collision)."""
        dt = 0.001
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, gravity=(0, 0, -9.81)))
        sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0, 0, 10.0)))
        scene.build()

        te_initial = sphere.get_total_energy().item()

        # Simulate free fall for 100 steps (0.1s of simulation)
        for _ in range(100):
            scene.step()

        te_final = sphere.get_total_energy().item()

        # Energy should be conserved to within a small tolerance
        # (numerical integration introduces a small drift)
        assert te_final == pytest.approx(te_initial, rel=1e-4)

    @pytest.mark.precision("64")
    def test_kinetic_energy_increases_during_free_fall(self):
        """Kinetic energy should increase as an object falls under gravity."""
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)))
        sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0, 0, 10.0)))
        scene.build()

        ke_before = sphere.get_kinetic_energy().item()
        assert ke_before == pytest.approx(0.0, abs=1e-12)

        for _ in range(10):
            scene.step()

        ke_after = sphere.get_kinetic_energy().item()
        assert ke_after > ke_before

    @pytest.mark.precision("64")
    def test_potential_energy_decreases_during_free_fall(self):
        """Potential energy should decrease as an object falls under gravity."""
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)))
        sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0, 0, 10.0)))
        scene.build()

        pe_before = sphere.get_potential_energy().item()

        for _ in range(10):
            scene.step()

        pe_after = sphere.get_potential_energy().item()
        assert pe_after < pe_before


class TestEnergyZeroGravity:
    """Test energy methods with zero gravity."""

    @pytest.mark.precision("64")
    def test_zero_gravity_zero_potential(self):
        """In zero gravity, potential energy should be zero."""
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, 0)))
        sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0, 0, 5.0)))
        scene.build()

        pe = sphere.get_potential_energy()
        assert pe.item() == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.precision("64")
    def test_zero_gravity_kinetic_energy_conservation(self):
        """In zero gravity with an initial velocity, kinetic energy should be conserved."""
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, 0)))
        sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0, 0, 0)))
        scene.build()

        # Give an initial velocity
        sphere.set_dofs_velocity(torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device))

        ke_initial = sphere.get_kinetic_energy().item()
        assert ke_initial > 0

        for _ in range(50):
            scene.step()

        ke_final = sphere.get_kinetic_energy().item()
        assert ke_final == pytest.approx(ke_initial, rel=1e-6)


class TestEnergyMultiLink:
    """Test energy methods with multi-link articulated bodies."""

    @pytest.mark.precision("64")
    def test_multi_link_energy_conservation(self):
        """Energy should be approximately conserved for a multi-link body in free fall."""
        dt = 0.001
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, gravity=(0, 0, -9.81)))
        # A box with a free joint acts as a simple multi-dof rigid body
        box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 5.0)))
        scene.build()

        te_initial = box.get_total_energy().item()

        for _ in range(50):
            scene.step()

        te_final = box.get_total_energy().item()
        assert te_final == pytest.approx(te_initial, rel=1e-4)


class TestEnergyReturnShape:
    """Test that energy methods return correct tensor shapes."""

    @pytest.mark.precision("64")
    def test_single_env_returns_scalar(self):
        """Without parallel envs, energy should be a scalar tensor."""
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)))
        sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0, 0, 2.0)))
        scene.build()

        ke = sphere.get_kinetic_energy()
        pe = sphere.get_potential_energy()
        te = sphere.get_total_energy()

        assert ke.dim() == 0
        assert pe.dim() == 0
        assert te.dim() == 0

    @pytest.mark.precision("64")
    def test_parallel_envs_returns_vector(self):
        """With parallel envs, energy should be a 1D tensor of shape (n_envs,)."""
        n_envs = 4
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
        )
        sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0, 0, 2.0)))
        scene.build(n_envs=n_envs)

        ke = sphere.get_kinetic_energy()
        pe = sphere.get_potential_energy()
        te = sphere.get_total_energy()

        assert ke.shape == (n_envs,)
        assert pe.shape == (n_envs,)
        assert te.shape == (n_envs,)

    @pytest.mark.precision("64")
    def test_parallel_envs_with_envs_idx(self):
        """Energy methods should respect envs_idx parameter."""
        n_envs = 4
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
        )
        sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0, 0, 2.0)))
        scene.build(n_envs=n_envs)

        envs_idx = [0, 2]
        ke = sphere.get_kinetic_energy(envs_idx=envs_idx)
        pe = sphere.get_potential_energy(envs_idx=envs_idx)
        te = sphere.get_total_energy(envs_idx=envs_idx)

        assert ke.shape == (2,)
        assert pe.shape == (2,)
        assert te.shape == (2,)


class TestEnergyNonStandardGravity:
    """Test energy methods with non-standard gravity directions."""

    @pytest.mark.precision("64")
    def test_gravity_along_x(self):
        """PE should work correctly when gravity is along the x-axis."""
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, gravity=(-9.81, 0, 0)))
        sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(3.0, 0, 0)))
        scene.build()

        mass = sphere.get_mass()
        pe = sphere.get_potential_energy()
        expected_pe = mass * 9.81 * 3.0
        assert pe.item() == pytest.approx(expected_pe, rel=1e-6)
