import quadrants as qd

import genesis as gs

from .base import Base


@qd.func
def partialJpartialF(F):
    pJpF0 = F[:, 1].cross(F[:, 2])
    pJpF1 = F[:, 2].cross(F[:, 0])
    pJpF2 = F[:, 0].cross(F[:, 1])
    pJpF = qd.Matrix.cols([pJpF0, pJpF1, pJpF2])
    return pJpF


@qd.data_oriented
class Elastic(Base):
    """
    The elastic material class for FEM.

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
    model: str, optional
        Constitutive model to use for stress computation. Options are:
        - 'linear': Linear elasticity model
        - 'stable_neohookean': A numerically stable Neo-Hookean model
        - 'linear_corotated': Linear corotated elasticity model
        Default is 'linear'.
    contact_resistance: float | None, optional
        IPC contact resistance/stiffness override. ``None`` uses the coupler global
        default. Default is None.
    """

    def __init__(
        self,
        E=1e6,  # Young's modulus
        nu=0.2,  # Poisson's ratio
        rho=1000.0,  # density (kg/m^3)
        hydroelastic_modulus=1e7,  # hydroelastic_modulus for hydroelastic contact
        friction_mu=0.1,
        model="linear",
        contact_resistance=None,
    ):
        super().__init__(
            E=E,
            nu=nu,
            rho=rho,
            hydroelastic_modulus=hydroelastic_modulus,
            friction_mu=friction_mu,
            contact_resistance=contact_resistance,
        )

        if model == "linear":
            self.update_stress = self.update_stress_linear
            self.compute_energy_gradient_hessian = self.compute_energy_gradient_hessian_linear
            self.compute_energy_gradient = self.compute_energy_gradient_linear
            self.compute_energy = self.compute_energy_linear
            self.hessian_invariant = True
        elif model in ("stable_neohookean", "stable_neohooken"):
            self.update_stress = self.update_stress_stable_neohookean
            self.compute_energy_gradient_hessian = self.compute_energy_gradient_hessian_stable_neohookean
            self.compute_energy_gradient = self.compute_energy_gradient_stable_neohookean
            self.compute_energy = self.compute_energy_stable_neohookean
            self.hessian_invariant = False
            if model == "stable_neohooken":
                gs.logger.warning("The 'stable_neohooken' model is deprecated. Use 'stable_neohookean' instead.")
        elif model == "linear_corotated":
            self.build = self.build_linear_corotated
            self.pre_compute = self.pre_compute_linear_corotated
            self.update_stress = self.update_stress_linear_corotated
            self.compute_energy_gradient_hessian = self.compute_energy_gradient_hessian_linear_corotated
            self.compute_energy_gradient = self.compute_energy_gradient_linear_corotated
            self.compute_energy = self.compute_energy_linear_corotated
            self.hessian_static = False
        else:
            gs.raise_exception(f"Unrecognized constitutive model: {model}")

        self._model = model

    def build_linear_corotated(self, fem_solver):
        self.R = qd.field(dtype=gs.qd_mat3, shape=(fem_solver._B, fem_solver.n_elements))

    @qd.func
    def pre_compute_linear_corotated(self, J, F, i_e, i_b):
        # Computing Polar Decomposition instead of calling `R, P = qd.polar_decompose(F)` since `P` is not needed here
        U, S, V = qd.svd(F)
        R = U @ V.transpose()
        self.R[i_b, i_e] = R

    @qd.func
    def update_stress_linear(self, mu, lam, J, F, actu, m_dir):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        stress = mu * (F + F.transpose() - 2 * I) + lam * (F - I).trace() * I

        return stress

    @qd.func
    def update_stress_stable_neohookean(self, mu, lam, J, F, actu, m_dir):
        IC = F.norm_sqr()
        dJdF0 = F[:, 1].cross(F[:, 2])
        dJdF1 = F[:, 2].cross(F[:, 0])
        dJdF2 = F[:, 0].cross(F[:, 1])
        dJdF = qd.Matrix.cols([dJdF0, dJdF1, dJdF2])
        alpha = 1 + 0.75 * mu / lam
        stress = mu * (1 - 1 / (IC + 1)) * F + lam * (J - alpha) * dJdF

        return stress

    @qd.func
    def update_stress_linear_corotated(self, mu, lam, J, F, actu, m_dir):
        gs.raise_exception("Linear corotated stress update is not implemented yet.")

    @qd.func
    def compute_energy_gradient_hessian_linear(self, mu, lam, J, F, actu, m_dir, i_e, i_b, hessian_field):
        """
        Compute the energy, gradient, and Hessian for linear elasticity.

        Parameters
        ----------
        mu: float
            The first Lame parameter (shear modulus).
        lam: float
            The second Lame parameter (related to volume change).
        J: float
            The determinant of the deformation gradient F.
        F: qd.Matrix
            The deformation gradient matrix.
        actu: qd.Matrix
            The activation matrix (not used in linear elasticity).
        m_dir: qd.Matrix
            The material direction (not used in linear elasticity).
        hessian_field: qd.Matrix
            The Hessian of the energy with respect to the deformation gradient F.

        Returns
        -------
        energy: float
            The computed energy.
        gradient: qd.Matrix
            The gradient of the energy with respect to the deformation gradient F.

        Notes
        -------
        This implementation assumes small deformations and linear stress-strain relationship.
        It is adapted from the HOBAKv1 implementation for linear elasticity:
        https://github.com/theodorekim/HOBAKv1/blob/main/src/Hyperelastic/Volume/LINEAR.cpp

        """
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2

        gradient = 2.0 * mu * eps + lam * trEps * I

        # Zero out the matrix
        for i in qd.static(qd.grouped(qd.ndrange(3, 3))):
            hessian_field[i_b, i, i_e].fill(0.0)

        # Identity part
        for i, k in qd.static(qd.ndrange(3, 3)):
            hessian_field[i_b, i, i, i_e][k, k] = mu

        # Diagonal terms
        hessian_field[i_b, 0, 0, i_e][0, 0] += mu + lam
        hessian_field[i_b, 1, 1, i_e][1, 1] += mu + lam
        hessian_field[i_b, 2, 2, i_e][2, 2] += mu + lam

        # Off-diagonal terms
        hessian_field[i_b, 0, 1, i_e][1, 0] = hessian_field[i_b, 1, 0, i_e][0, 1] = mu
        hessian_field[i_b, 0, 2, i_e][2, 0] = hessian_field[i_b, 2, 0, i_e][0, 2] = mu
        hessian_field[i_b, 1, 2, i_e][2, 1] = hessian_field[i_b, 2, 1, i_e][1, 2] = mu

        # Pressure coupling terms
        hessian_field[i_b, 0, 1, i_e][0, 1] = hessian_field[i_b, 0, 2, i_e][0, 2] = lam
        hessian_field[i_b, 1, 0, i_e][1, 0] = hessian_field[i_b, 2, 0, i_e][2, 0] = lam
        hessian_field[i_b, 1, 2, i_e][1, 2] = hessian_field[i_b, 2, 1, i_e][2, 1] = lam
        return energy, gradient

    @qd.func
    def compute_energy_gradient_linear(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        """
        Compute the energy, gradient for linear elasticity.

        Parameters
        ----------
        mu: float
            The first Lame parameter (shear modulus).
        lam: float
            The second Lame parameter (related to volume change).
        J: float
            The determinant of the deformation gradient F.
        F: qd.Matrix
            The deformation gradient matrix.
        actu: qd.Matrix
            The activation matrix (not used in linear elasticity).
        m_dir: qd.Matrix
            The material direction (not used in linear elasticity).

        Returns
        -------
        energy: float
            The computed energy.
        gradient: qd.Matrix
            The gradient of the energy with respect to the deformation gradient F.

        Notes
        -------
        This implementation assumes small deformations and linear stress-strain relationship.
        It is adapted from the HOBAKv1 implementation for linear elasticity:
        https://github.com/theodorekim/HOBAKv1/blob/8420c51b795735d8fb912e0f8810f935d96fb636/src/Hyperelastic/Volume/LINEAR.cpp
        """
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2

        gradient = 2.0 * mu * eps + lam * trEps * I

        return energy, gradient

    @qd.func
    def compute_energy_linear(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        """
        Compute the energy for linear elasticity.

        Parameters
        ----------
        mu: float
            The first Lame parameter (shear modulus).
        lam: float
            The second Lame parameter (related to volume change).
        J: float
            The determinant of the deformation gradient F.
        F: qd.Matrix
            The deformation gradient matrix.
        actu: qd.Matrix
            The activation matrix (not used in linear elasticity).
        m_dir: qd.Matrix
            The material direction (not used in linear elasticity).

        Returns
        -------
        energy: float
            The computed energy.

        Notes
        -------
        This implementation assumes small deformations and linear stress-strain relationship.
        It is adapted from the HOBAKv1 implementation for linear elasticity:
        https://github.com/theodorekim/HOBAKv1/blob/8420c51b795735d8fb912e0f8810f935d96fb636/src/Hyperelastic/Volume/LINEAR.cpp
        """
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2

        return energy

    @qd.func
    def compute_energy_gradient_hessian_stable_neohookean(self, mu, lam, J, F, actu, m_dir, i_e, i_b, hessian_field):
        """
        Compute the energy, gradient, and Hessian for the stable Neo-Hookean model.

        Parameters
        ----------
        mu: float
            The first Lame parameter (shear modulus).
        lam: float
            The second Lame parameter (related to volume change).
        J: float
            The determinant of the deformation gradient F.
        F: qd.Matrix
            The deformation gradient matrix.
        actu: qd.Matrix
            The activation matrix (not used in stable Neo-Hookean).
        m_dir: qd.Matrix
            The material direction (not used in stable Neo-Hookean).
        hessian_field: qd.Matrix
            The Hessian of the energy with respect to the deformation gradient F.

        Returns
        -------
        energy: float
            The computed energy.
        gradient: qd.Matrix
            The gradient of the energy with respect to the deformation gradient F.

        Raises
        -------
        NotImplementedError
            This implementation does not compute the Hessian for the stable Neo-Hookean model.
            The Hessian needs SVD decomposition for accurate computation, which is not implemented here.

        Notes
        -------
        This implementation is adapted from the HOBAKv1 stable Neo-Hookean model:
        https://github.com/theodorekim/HOBAKv1/blob/main/src/Hyperelastic/Volume/SNH.cpp
        """
        raise NotImplementedError("Hessian computation is not implemented for stable_neohookean model.")

    @qd.func
    def compute_energy_gradient_stable_neohookean(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        """
        Compute the energy, gradient for the stable Neo-Hookean model.

        Parameters
        ----------
        mu: float
            The first Lame parameter (shear modulus).
        lam: float
            The second Lame parameter (related to volume change).
        J: float
            The determinant of the deformation gradient F.
        F: qd.Matrix
            The deformation gradient matrix.
        actu: qd.Matrix
            The activation matrix (not used in stable Neo-Hookean).
        m_dir: qd.Matrix
            The material direction (not used in stable Neo-Hookean).

        Returns
        -------
        energy: float
            The computed energy.
        gradient: qd.Matrix
            The gradient of the energy with respect to the deformation gradient F.

        Raises
        -------
        NotImplementedError
            This implementation does not compute the Gradient for the stable Neo-Hookean model.

        Notes
        -------
        This implementation is adapted from the HOBAKv1 stable Neo-Hookean model:
        https://github.com/theodorekim/HOBAKv1/blob/8420c51b795735d8fb912e0f8810f935d96fb636/src/Hyperelastic/Volume/SNH.cpp
        """
        gs.raise_exception("Gradient computation is not implemented for stable_neohookean model.")

    @qd.func
    def compute_energy_stable_neohookean(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        """
        Compute the energy for the stable Neo-Hookean model.

        Parameters
        ----------
        mu: float
            The first Lame parameter (shear modulus).
        lam: float
            The second Lame parameter (related to volume change).
        J: float
            The determinant of the deformation gradient F.
        F: qd.Matrix
            The deformation gradient matrix.
        actu: qd.Matrix
            The activation matrix (not used in stable Neo-Hookean).
        m_dir: qd.Matrix
            The material direction (not used in stable Neo-Hookean).

        Returns
        -------
        energy: float
            The computed energy.

        Notes
        -------
        This implementation is adapted from the HOBAKv1 stable Neo-Hookean model:
        https://github.com/theodorekim/HOBAKv1/blob/8420c51b795735d8fb912e0f8810f935d96fb636/src/Hyperelastic/Volume/SNH.cpp
        """
        _lambda = lam + mu
        _alpha = 1.0 + mu / _lambda

        Ic = F.norm_sqr()
        Jminus1 = J - _alpha
        energy = 0.5 * (mu * (Ic - 3.0) + _lambda * Jminus1**2)

        return energy

    @qd.func
    def compute_energy_gradient_hessian_linear_corotated(self, mu, lam, J, F, actu, m_dir, i_e, i_b, hessian_field):
        """
        Compute the energy, gradient, and Hessian for linear elasticity.

        Parameters
        ----------
        mu: float
            The first Lame parameter (shear modulus).
        lam: float
            The second Lame parameter (related to volume change).
        J: float
            The determinant of the deformation gradient F.
        F: qd.Matrix
            The deformation gradient matrix.
        actu: qd.Matrix
            The activation matrix (not used in linear elasticity).
        m_dir: qd.Matrix
            The material direction (not used in linear elasticity).
        hessian_field: qd.Matrix
            The Hessian of the energy with respect to the deformation gradient F.

        Returns
        -------
        energy: float
            The computed energy.
        gradient: qd.Matrix
            The gradient of the energy with respect to the deformation gradient F.

        Notes
        -------
        This implementation assumes small deformations and linear stress-strain relationship.
        It is adapted from the HOBAKv1 implementation for linear elasticity:
        https://github.com/theodorekim/HOBAKv1/blob/main/src/Hyperelastic/Volume/LINEAR.cpp

        """
        R = self.R[i_b, i_e]
        F_hat = R.transpose() @ F
        # E = 1/2(F_hat + F_hat.transpose()) - I
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2

        gradient = 2.0 * mu * R @ eps + lam * trEps * R

        # Zero out the matrix
        for i in qd.static(qd.grouped(qd.ndrange(3, 3))):
            hessian_field[i_b, i, i_e].fill(0.0)

        # Identity part
        for i, k in qd.static(qd.ndrange(3, 3)):
            hessian_field[i_b, i, i, i_e][k, k] = mu

        for i, j, alpha, beta in qd.ndrange(3, 3, 3, 3):
            hessian_field[i_b, j, beta, i_e][i, alpha] += mu * R[i, beta] * R[alpha, j] + lam * R[alpha, beta] * R[i, j]

        return energy, gradient

    @qd.func
    def compute_energy_gradient_linear_corotated(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        """
        Compute the energy, gradient for linear elasticity.

        Parameters
        ----------
        mu: float
            The first Lame parameter (shear modulus).
        lam: float
            The second Lame parameter (related to volume change).
        J: float
            The determinant of the deformation gradient F.
        F: qd.Matrix
            The deformation gradient matrix.
        actu: qd.Matrix
            The activation matrix (not used in linear elasticity).
        m_dir: qd.Matrix
            The material direction (not used in linear elasticity).

        Returns
        -------
        energy: float
            The computed energy.
        gradient: qd.Matrix
            The gradient of the energy with respect to the deformation gradient F.

        Notes
        -------
        This implementation assumes small deformations and linear stress-strain relationship.
        It is adapted from the HOBAKv1 implementation for linear elasticity:
        https://github.com/theodorekim/HOBAKv1/blob/main/src/Hyperelastic/Volume/LINEAR.cpp

        """
        F_hat = self.R[i_b, i_e].transpose() @ F
        # E = 1/2(F_hat + F_hat.transpose()) - I
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * self.R[i_b, i_e] @ eps + lam * trEps * self.R[i_b, i_e]

        return energy, gradient

    @qd.func
    def compute_energy_linear_corotated(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        """
        Compute the energy for linear elasticity.

        Parameters
        ----------
        mu: float
            The first Lame parameter (shear modulus).
        lam: float
            The second Lame parameter (related to volume change).
        J: float
            The determinant of the deformation gradient F.
        F: qd.Matrix
            The deformation gradient matrix.
        actu: qd.Matrix
            The activation matrix (not used in linear elasticity).
        m_dir: qd.Matrix
            The material direction (not used in linear elasticity).

        Returns
        -------
        energy: float
            The computed energy.

        Notes
        -------
        This implementation assumes small deformations and linear stress-strain relationship.
        It is adapted from the HOBAKv1 implementation for linear elasticity:
        https://github.com/theodorekim/HOBAKv1/blob/main/src/Hyperelastic/Volume/LINEAR.cpp

        """
        F_hat = self.R[i_b, i_e].transpose() @ F
        # E = 1/2(F_hat + F_hat.transpose()) - I
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2

        return energy

    @property
    def model(self):
        """The name of the constitutive model ('linear' or 'stable_neohookean')."""
        return self._model
