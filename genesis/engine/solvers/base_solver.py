from typing import TYPE_CHECKING
import numpy as np
import taichi as ti
import torch
from genesis.utils.misc import ti_field_to_torch

import genesis as gs
from genesis.engine.entities.base_entity import Entity
from genesis.repr_base import RBC


if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class Solver(RBC):
    def __init__(self, scene: "Scene", sim: "Simulator", options):
        self._uid = gs.UID()
        self._sim = sim
        self._scene = scene
        self._dt: float = options.dt
        self._substep_dt: float = options.dt / sim.substeps

        if hasattr(options, "gravity"):
            self._gravity = ti.field(dtype=gs.ti_vec3, shape=())
            self._gravity.from_numpy(np.array(options.gravity, dtype=gs.np_float))
        else:
            self._gravity = None

        self._entities: list[Entity] = gs.List()

        # force fields
        self._ffs = list()

    def _add_force_field(self, force_field):
        self._ffs.append(force_field)

    def dump_ckpt_to_numpy(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}

        for attr_name, field in self.__dict__.items():
            if not isinstance(field, ti.Field):
                continue

            key_base = ".".join((self.__class__.__name__, attr_name))
            data = field.to_numpy()

            # StructField → data is a dict: flatten each member
            if isinstance(data, dict):
                for sub_name, sub_arr in data.items():
                    arrays[f"{key_base}.{sub_name}"] = (
                        sub_arr if isinstance(sub_arr, np.ndarray) else np.asarray(sub_arr)
                    )
            else:
                arrays[key_base] = data if isinstance(data, np.ndarray) else np.asarray(data)

        return arrays

    def load_ckpt_from_numpy(self, arr_dict: dict[str, np.ndarray]) -> None:
        for attr_name, field in self.__dict__.items():
            if not isinstance(field, ti.Field):
                continue

            key_base = ".".join((self.__class__.__name__, attr_name))
            member_prefix = key_base + "."

            # ---- StructField: gather its members -----------------------------
            member_items = {}
            for saved_key, saved_arr in arr_dict.items():
                if saved_key.startswith(member_prefix):
                    sub_name = saved_key[len(member_prefix) :]
                    member_items[sub_name] = saved_arr

            if member_items:  # we found at least one sub-member
                field.from_numpy(member_items)
                continue

            # ---- Ordinary field ---------------------------------------------
            if key_base not in arr_dict:
                continue  # nothing saved for this attribute

            arr = arr_dict[key_base]
            field.from_numpy(arr)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        return self._uid

    @property
    def scene(self):
        return self._scene

    @property
    def sim(self):
        return self._sim

    @property
    def dt(self):
        return self._dt

    @property
    def is_built(self):
        return self._scene._is_built

    @property
    def substep_dt(self):
        return self._substep_dt

    @property
    def gravity(self):
        return self._gravity.to_numpy() if self._gravity is not None else None

    @property
    def entities(self):
        return self._entities

    @property
    def n_entities(self):
        return len(self._entities)

    def _repr_brief(self):
        repr_str = f"{self._repr_type()}: {self._uid}, n_entities: {self.n_entities}"
        return repr_str
