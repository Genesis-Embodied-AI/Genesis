import os
import xml.etree.ElementTree as etxml

import numpy as np
import taichi as ti
import torch

import genesis as gs
import genesis.utils.misc as mu
from genesis.utils.geom import quat_to_xyz
from genesis.utils.pid import PIDController

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
        self._propellers_link_idxs = np.array([link.idx for link in propellers_links], dtype=gs.np_int)
        try:
            self._propellers_vgeom_idxs = np.array([link.vgeoms[0].idx for link in propellers_links], dtype=gs.np_int)
            self._animate_propellers = True
        except Exception:
            gs.logger.warning("No visual geometry found for propellers. Skipping propeller animation.")
            self._animate_propellers = False

        self._propellers_spin = np.array(morph.propellers_spin, dtype=gs.np_float)
        self._model = morph.model

    def _build(self):
        super()._build()

        self._propellers_revs = np.zeros(self._solver._batch_shape(self._n_propellers), dtype=gs.np_float)
        self._prev_prop_t = None

    def set_propellels_rpm(self, propellels_rpm):
        if self._prev_prop_t == self.sim.cur_step_global:
            gs.raise_exception("`set_propellels_rpm` can only be called once per step.")
        self._prev_prop_t = self.sim.cur_step_global

        propellels_rpm = self.solver._process_dim(np.array(propellels_rpm, dtype=gs.np_float)).T
        if len(propellels_rpm) != len(self._propellers_link_idxs):
            gs.raise_exception("Last dimension of `propellels_rpm` does not match `entity.n_propellers`.")
        if np.any(propellels_rpm < 0):
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
    
class DronePIDController():
    def __init__(self, drone: DroneEntity, dt, base_rpm, pid_params):
        self.__pid_pos_x = PIDController(
            kp=pid_params[0][0], 
            ki=pid_params[0][1], 
            kd=pid_params[0][2])
        self.__pid_pos_y = PIDController(kp=pid_params[1][0],
             ki=pid_params[1][1], 
             kd=pid_params[1][2])
        self.__pid_pos_z = PIDController(kp=pid_params[2][0],
             ki=pid_params[2][1], 
             kd=pid_params[2][2])

        self.__pid_vel_x = PIDController(kp=pid_params[3][0],
             ki=pid_params[3][1], 
             kd=pid_params[3][2])
        self.__pid_vel_y = PIDController(kp=pid_params[4][0],
             ki=pid_params[4][1], 
             kd=pid_params[4][2])
        self.__pid_vel_z = PIDController(kp=pid_params[5][0],
             ki=pid_params[5][1], 
             kd=pid_params[5][2])

        self.__pid_att_roll  = PIDController(kp=pid_params[6][0],
             ki=pid_params[6][1], 
             kd=pid_params[6][2])
        self.__pid_att_pitch = PIDController(kp=pid_params[7][0],
             ki=pid_params[7][1], 
             kd=pid_params[7][2])
        self.__pid_att_yaw   = PIDController(kp=pid_params[8][0],
             ki=pid_params[8][1], 
             kd=pid_params[8][2])

        self.drone = drone
        self.__dt = dt
        self.__base_rpm = base_rpm

    def __get_drone_pos(self) -> torch.Tensor:
        return self.drone.get_pos()

    def __get_drone_vel(self) -> torch.Tensor:
        return self.drone.get_vel()
    
    def __get_drone_att(self) -> torch.Tensor:
        quat = self.drone.get_quat()
        # print(quat_to_xyz(quat))
        return quat_to_xyz(quat)
    
    def __mixer(self, thrust, roll, pitch, yaw, x_vel, y_vel) -> torch.Tensor:
        M1 = self.__base_rpm + (thrust - roll - pitch - yaw - x_vel + y_vel)
        M2 = self.__base_rpm + (thrust - roll + pitch + yaw + x_vel + y_vel)
        M3 = self.__base_rpm + (thrust + roll + pitch - yaw + x_vel - y_vel)
        M4 = self.__base_rpm + (thrust + roll - pitch + yaw - x_vel - y_vel)
        # print("pitch =", pitch)
        # print("roll =", roll)

        return torch.Tensor([M1, M2, M3, M4])

    def update(self, target) -> np.ndarray:
        curr_pos = self.__get_drone_pos()
        curr_vel = self.__get_drone_vel()
        curr_att = self.__get_drone_att()

        err_pos_x = target[0] - curr_pos[0]
        err_pos_y = target[1] - curr_pos[1]
        err_pos_z = target[2] - curr_pos[2]

        vel_des_x = self.__pid_pos_x.update(err_pos_x, self.__dt)
        vel_des_y = self.__pid_pos_y.update(err_pos_y, self.__dt)
        vel_des_z = self.__pid_pos_z.update(err_pos_z, self.__dt)

        error_vel_x = vel_des_x - curr_vel[0]
        error_vel_y = vel_des_y - curr_vel[1]
        error_vel_z = vel_des_z - curr_vel[2]

        x_vel_del   = self.__pid_vel_x.update(error_vel_x, self.__dt)
        y_vel_del  = self.__pid_vel_y.update(error_vel_y, self.__dt)
        thrust_des = self.__pid_vel_z.update(error_vel_z, self.__dt)

        err_roll  = 0. - curr_att[0]
        err_pitch = 0. - curr_att[1]
        err_yaw   = 0. - curr_att[2]

        roll_del  = self.__pid_att_roll.update(err_roll, self.__dt)
        pitch_del = self.__pid_att_pitch.update(err_pitch, self.__dt)
        yaw_del   = self.__pid_att_yaw.update(err_yaw, self.__dt)

        prop_rpms = self.__mixer(thrust_des, roll_del, pitch_del, yaw_del, x_vel_del, y_vel_del)
        prop_rpms = prop_rpms.cpu()
        prop_rpms - prop_rpms.numpy()

        return prop_rpms