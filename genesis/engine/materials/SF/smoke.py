from .base import Base


class Smoke(Base):
    @property
    def sampler(self):
        return "regular"
