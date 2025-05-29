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
        model="linear",
    ):
        super().__init__(E, nu, rho)

        if model == "linear":
            self.update_stress = self.update_stress_linear
            self.compute_energy_gradient_hessian = self.compute_energy_gradient_hessian_linear
        elif model == "stable_neohookean":
            self.update_stress = self.update_stress_stable_neohookean
            self.compute_energy_gradient_hessian = self.compute_energy_gradient_hessian_stable_neohookean
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

    # https://github.com/theodorekim/HOBAKv1/blob/main/src/Hyperelastic/Volume/LINEAR.cpp
    @ti.func
    def compute_energy_gradient_hessian_linear(self, mu, lam, J, F, actu, m_dir):
        I = ti.Matrix.identity(dt=gs.ti_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps * trEps

        gradient = 2.0 * mu * eps + lam * trEps * I

        H = mu * ti.Matrix.identity(dt=gs.ti_float, n=9)
        H[0, 0] += mu + lam
        H[4, 4] += mu + lam
        H[8, 8] += mu + lam

        H[1, 3] = H[3, 1] = mu
        H[2, 6] = H[6, 2] = mu
        H[5, 7] = H[7, 5] = mu

        H[0, 4] = H[0, 8] = H[4, 8] = lam
        H[4, 0] = H[8, 0] = H[8, 4] = lam
        return energy, gradient, H

    # https://github.com/theodorekim/HOBAKv1/blob/main/src/Hyperelastic/Volume/SNH.cpp
    @ti.func
    def compute_energy_gradient_hessian_stable_neohookean(self, mu, lam, J, F, actu, m_dir):
        _mu = mu
        _lambda = lam + mu
        _alpha = 1.0 + _mu / _lambda

        Ic = (F.transpose() @ F).trace()
        Jminus1 = J - _alpha
        energy = 0.5 * (_mu * (Ic - 3.0) + _lambda * Jminus1 * Jminus1)

        pJpF = partialJpartialF(F)
        gradient = _mu * F + _lambda * Jminus1 * pJpF

        raise NotImplementedError("Hessian computation is not implemented for stable_neohookean model.")
        return energy, gradient

    @property
    def model(self):
        """The name of the constitutive model ('linear' or 'stable_neohookean')."""
        return self._model
