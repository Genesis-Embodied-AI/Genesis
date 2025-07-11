import taichi as ti
import genesis as gs
import torch
from .base_sensor import Sensor
from genesis.engine.entities import RigidEntity

@ti.data_oriented
class ContactSensor(Sensor):
    _all_contact_links_idx = None
    _last_updated_step = -1

    @staticmethod
    def get_valid_entity_types():
        return (RigidEntity)

    def __init__(self, entity, link_idx=None):
        super().__init__(entity)
        if link_idx:
            self.links_idx = torch.atleast_1d(torch.as_tensor(link_idx, dtype=torch.int32, device=gs.device))
        else:
            self.links_idx = torch.arange(
                entity._link_start, entity._link_start+entity.n_links, dtype=torch.int32, device=gs.device
            )

    @classmethod
    def _update_shared_buffer(cls, sim):
        if sim.n_envs == 0:
            contacts_info = sim.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=True)
            contact_links = torch.cat([contacts_info['link_a'], contacts_info['link_b']], dim=0)
            cls._all_contact_links_idx = contact_links.unique()
        else:
            contacts_info = sim.rigid_solver.collider.get_contacts(as_tensor=True, to_torch=True)
            contact_links = torch.cat([contacts_info['link_a'], contacts_info['link_b']], dim=1)
            cls._all_contact_links_idx = contact_links

    def read(self, envs_idx=None):
        cls = type(self)
        if cls._last_updated_step != self._sim.cur_step_global:
            cls._update_shared_buffer(self._sim)
            cls._last_updated_step = self._sim.cur_step_global

        if self.n_envs == 0:
            is_contact = torch.isin(self.links_idx, self._all_contact_links_idx).any().item()
        else:
            is_contact = torch.isin(
                self.links_idx.unsqueeze(0).expand(self.n_envs, -1),
                self._all_contact_links_idx
            ).any(dim=1)
            
        return is_contact[envs_idx] if envs_idx is not None else is_contact
    
    @property
    def _all_contact_links_idx(self):
        return self.__class__._all_contact_links_idx
