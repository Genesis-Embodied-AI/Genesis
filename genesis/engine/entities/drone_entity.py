import os
import xml.etree.ElementTree as etxml

import torch
import taichi as ti

import genesis as gs
import genesis.utils.misc as mu

from .rigid_entity import RigidEntity


@ti.data_oriented
class DroneEntity(RigidEntity):
    def _load_URDF(self, morph, surface):
        super()._load_URDF(morph, surface)

        # additional drone specific attributes
        properties = etxml.parse(os.path.join(mu.get_assets_dir(), morph.file)).getroot()[0].attrib
        self._KF = float(properties["kf"])
        self._KM = float(properties["km"])

        self._n_propellers = len(morph.propellers_link_names)
        self._COM_link_idx = self.get_link(morph.COM_link_name).idx

        propellers_links = gs.List([self.get_link(name) for name in morph.propellers_link_names])
        self._propellers_link_idxs = torch.tensor(
            [link.idx for link in propellers_links], dtype=gs.tc_int, device=gs.device
        )
        try:
            self._propellers_vgeom_idxs = torch.tensor(
                [link.vgeoms[0].idx for link in propellers_links], dtype=gs.tc_int, device=gs.device
            )
            self._animate_propellers = True
        except Exception:
            gs.logger.warning("No visual geometry found for propellers. Skipping propeller animation.")
            self._animate_propellers = False

        self._propellers_spin = torch.tensor(morph.propellers_spin, dtype=gs.tc_float, device=gs.device)
        self._model = morph.model

    def _build(self):
        super()._build()

        self._propellers_revs = torch.zeros(
            self._solver._batch_shape(self._n_propellers), dtype=gs.tc_float, device=gs.device
        )
        self._prev_prop_t = None

    def set_propellels_rpm(self, propellels_rpm):
        if self._prev_prop_t == self.sim.cur_step_global:
            gs.raise_exception("`set_propellels_rpm` can only be called once per step.")
        self._prev_prop_t = self.sim.cur_step_global

        propellels_rpm = self.solver._process_dim(
            torch.as_tensor(propellels_rpm, dtype=gs.tc_float, device=gs.device)
        ).T.contiguous()
        if len(propellels_rpm) != len(self._propellers_link_idxs):
            gs.raise_exception("Last dimension of `propellels_rpm` does not match `entity.n_propellers`.")
        if torch.any(propellels_rpm < 0):
            gs.raise_exception("`propellels_rpm` cannot be negative.")
        self._propellers_revs = (self._propellers_revs + propellels_rpm) % (60 / self.solver.dt)

        self.solver._kernel_set_drone_rpm(
            self._n_propellers,
            self._COM_link_idx,
            self._propellers_link_idxs,
            propellels_rpm,
            self._propellers_spin,
            self.KF,
            self.KM,
            self._model == "RACE",
        )

    def update_propeller_vgeoms(self):
        if self._animate_propellers:
            self.solver._update_drone_propeller_vgeoms(
                self._n_propellers, self._propellers_vgeom_idxs, self._propellers_revs, self._propellers_spin
            )

    @property
    def model(self):
        return self._model

    @property
    def KF(self):
        return self._KF

    @property
    def KM(self):
        return self._KM

    @property
    def n_propellers(self):
        return self._n_propellers

    @property
    def COM_link_idx(self):
        return self._COM_link_idx

    @property
    def propellers_idx(self):
        return self._propellers_link_idxs

    @property
    def propellers_spin(self):
        return self._propellers_spin
