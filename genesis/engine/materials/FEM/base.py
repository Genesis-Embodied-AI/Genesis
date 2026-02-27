import quadrants as qd

import genesis as gs

from ..base import Material


@qd.data_oriented
class Base(Material):
    """
    The base class of MPM materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    E: float, optional
        Young's modulus, which controls stiffness. Default is 1e6.
    nu: float, optional
        Poisson ratio, describing the material's volume change under stress. Default is 0.2.
    rho: float, optional
        Material density (kg/m^3). Default is 1000.
    hydroelastic_modulus: float, optional
        Hydroelastic modulus for hydroelastic contact. Default is 1e7.
    friction_mu: float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance: float | None, optional
        IPC contact resistance/stiffness override for this material. ``None`` means
        use the IPC coupler global default (``IPCCouplerOptions.contact_resistance``).
    hessian_invariant: bool, optional
        If True, Hessian is computed only once. Default is False.
    """

    def __init__(
        self,
        E=1e6,
        nu=0.2,
        rho=1000.0,
        hydroelastic_modulus=1e7,
        friction_mu=0.1,
        contact_resistance=None,
        hessian_invariant=False,
    ):
        super().__init__()

        if friction_mu < 0:
            gs.raise_exception("`friction_mu` must be non-negative.")
        if contact_resistance is not None and contact_resistance < 0:
            gs.raise_exception("`contact_resistance` must be non-negative.")

        self._E = E
        self._nu = nu
        self._rho = rho
        self._hydroelastic_modulus = hydroelastic_modulus
        self._friction_mu = float(friction_mu)
        self._contact_resistance = float(contact_resistance) if contact_resistance is not None else None
        self.hessian_invariant = hessian_invariant
        self.hessian_ready = False

        # lame parameters: https://github.com/taichi-dev/taichi_elements/blob/d19678869a28b09a32ef415b162e35dc929b792d/engine/mpm_solver.py#L203
        self._mu = E / (2.0 * (1.0 + nu))
        self._lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # will be set when added to solver
        self._idx = None

    def build(self, fem_solver):
        pass

    @qd.func
    def pre_compute(self, J, F, i_e, i_b):
        pass

    @qd.func
    def update_stress(self, mu, lam, J, F, actu, m_dir):
        raise NotImplementedError

    @qd.func
    def compute_energy_gradient_hessian(self, mu, lam, J, F, actu, m_dir, i_e, i_b, hessian_field):
        raise NotImplementedError

    @qd.func
    def compute_energy(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        raise NotImplementedError

    @property
    def idx(self):
        return self._idx

    @property
    def E(self):
        """Young's modulus."""
        return self._E

    @property
    def nu(self):
        """Poisson ratio."""
        return self._nu

    @property
    def mu(self):
        """The first Lame parameters."""
        return self._mu

    @property
    def lam(self):
        """The second Lame parameters."""
        return self._lam

    @property
    def rho(self):
        """The rest density."""
        return self._rho

    @property
    def contact_stiffness(self):
        """The contact stiffness."""
        return self._hydroelastic_modulus

    @property
    def friction_mu(self):
        """The friction coefficient."""
        return self._friction_mu

    @property
    def contact_resistance(self):
        """IPC contact resistance/stiffness override, or None to use coupler default."""
        return self._contact_resistance
