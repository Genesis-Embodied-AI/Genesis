import taichi as ti

import genesis as gs

from .base import Base


@ti.func
def partialJpartialF(F):
    pJpF0 = F[:, 1].cross(F[:, 2])
    pJpF1 = F[:, 2].cross(F[:, 0])
    pJpF2 = F[:, 0].cross(F[:, 1])
    pJpF = ti.Matrix.cols([pJpF0, pJpF1, pJpF2])
    return pJpF


@ti.data_oriented
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
        Default is 'linear'.
    """

    def __init__(
        self,
        E=1e6,  # Young's modulus
        nu=0.2,  # Poisson's ratio
        rho=1000.0,  # density (kg/m^3)
        hydroelastic_modulus=1e7,  # hydroelastic_modulus for hydroelastic contact
        friction_mu=0.1,
        model="linear",
    ):
        super().__init__(E, nu, rho, hydroelastic_modulus, friction_mu)

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
        else:
            gs.raise_exception(f"Unrecognized constitutive model: {model}")

        self._model = model

    @ti.func
    def update_stress_linear(self, mu, lam, J, F, actu, m_dir):
        I = ti.Matrix.identity(dt=gs.ti_float, n=3)
        stress = mu * (F + F.transpose() - 2 * I) + lam * (F - I).trace() * I

        return stress

    @ti.func
    def update_stress_stable_neohookean(self, mu, lam, J, F, actu, m_dir):
        IC = F.norm_sqr()
        dJdF0 = F[:, 1].cross(F[:, 2])
        dJdF1 = F[:, 2].cross(F[:, 0])
        dJdF2 = F[:, 0].cross(F[:, 1])
        dJdF = ti.Matrix.cols([dJdF0, dJdF1, dJdF2])
        alpha = 1 + 0.75 * mu / lam
        stress = mu * (1 - 1 / (IC + 1)) * F + lam * (J - alpha) * dJdF

        return stress

    @ti.func
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
        F: ti.Matrix
            The deformation gradient matrix.
        actu: ti.Matrix
            The activation matrix (not used in linear elasticity).
        m_dir: ti.Matrix
            The material direction (not used in linear elasticity).
        hessian_field: ti.Matrix
            The Hessian of the energy with respect to the deformation gradient F.

        Returns
        -------
        energy: float
            The computed energy.
        gradient: ti.Matrix
            The gradient of the energy with respect to the deformation gradient F.

        Notes
        -------
        This implementation assumes small deformations and linear stress-strain relationship.
        It is adapted from the HOBAKv1 implementation for linear elasticity:
        https://github.com/theodorekim/HOBAKv1/blob/main/src/Hyperelastic/Volume/LINEAR.cpp

        """
        I = ti.Matrix.identity(dt=gs.ti_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2

        gradient = 2.0 * mu * eps + lam * trEps * I

        # Zero out the matrix
        for i in ti.static(ti.grouped(ti.ndrange(3, 3))):
            hessian_field[i_b, i, i_e].fill(0.0)

        # Identity part
        for i, k in ti.static(ti.ndrange(3, 3)):
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

    @ti.func
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
        F: ti.Matrix
            The deformation gradient matrix.
        actu: ti.Matrix
            The activation matrix (not used in linear elasticity).
        m_dir: ti.Matrix
            The material direction (not used in linear elasticity).

        Returns
        -------
        energy: float
            The computed energy.
        gradient: ti.Matrix
            The gradient of the energy with respect to the deformation gradient F.

        Notes
        -------
        This implementation assumes small deformations and linear stress-strain relationship.
        It is adapted from the HOBAKv1 implementation for linear elasticity:
        https://github.com/theodorekim/HOBAKv1/blob/8420c51b795735d8fb912e0f8810f935d96fb636/src/Hyperelastic/Volume/LINEAR.cpp
        """
        I = ti.Matrix.identity(dt=gs.ti_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2

        gradient = 2.0 * mu * eps + lam * trEps * I

        return energy, gradient

    @ti.func
    def compute_energy_linear(self, mu, lam, J, F, actu, m_dir):
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
        F: ti.Matrix
            The deformation gradient matrix.
        actu: ti.Matrix
            The activation matrix (not used in linear elasticity).
        m_dir: ti.Matrix
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
        I = ti.Matrix.identity(dt=gs.ti_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2

        return energy

    @ti.func
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
        F: ti.Matrix
            The deformation gradient matrix.
        actu: ti.Matrix
            The activation matrix (not used in stable Neo-Hookean).
        m_dir: ti.Matrix
            The material direction (not used in stable Neo-Hookean).
        hessian_field: ti.Matrix
            The Hessian of the energy with respect to the deformation gradient F.

        Returns
        -------
        energy: float
            The computed energy.
        gradient: ti.Matrix
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

    @ti.func
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
        F: ti.Matrix
            The deformation gradient matrix.
        actu: ti.Matrix
            The activation matrix (not used in stable Neo-Hookean).
        m_dir: ti.Matrix
            The material direction (not used in stable Neo-Hookean).

        Returns
        -------
        energy: float
            The computed energy.
        gradient: ti.Matrix
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
        raise NotImplementedError("Gradient computation is not implemented for stable_neohookean model.")

    @ti.func
    def compute_energy_stable_neohookean(self, mu, lam, J, F, actu, m_dir):
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
        F: ti.Matrix
            The deformation gradient matrix.
        actu: ti.Matrix
            The activation matrix (not used in stable Neo-Hookean).
        m_dir: ti.Matrix
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

    @property
    def model(self):
        """The name of the constitutive model ('linear' or 'stable_neohookean')."""
        return self._model
