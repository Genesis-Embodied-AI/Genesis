from .base_sensor import Sensor
from genesis.engine.entities import RigidEntity

class ContactSensor(Sensor):
    _compatible_entity_types = (RigidEntity)

    def __init__(self, entity, link_idx=None):
        super().__init__(entity)
        self.link_idx = link_idx

    def read(self):
        if self.link_idx is None:
            return self.entity.get_contacts()['geom_a'].size(0) > 0
        else:
            contacts = self.entity.get_contacts()
            return self.link_idx in contacts['link_a'] or self.link_idx in contacts['link_b']
