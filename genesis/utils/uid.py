import uuid

from genesis.repr_base import RBC


class UID(RBC):
    def __init__(self) -> None:
        self.uid = uuid.uuid4().hex

    def _repr_brief(self):
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self._repr_type()}('{self.full()}')"

    def __format__(self, format_spec) -> str:
        return f"<{self.short()}>"

    def __str__(self) -> str:
        return self.uid

    def full(self) -> str:
        return f"{self.uid[:7]}-{self.uid[7:]}"

    def short(self) -> str:
        return self.uid[:7]
