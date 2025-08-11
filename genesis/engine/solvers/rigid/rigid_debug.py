import taichi as ti


class Debug:

    @classmethod
    def enable(cls, enable: bool = True):
        cls.validate = enable
        cls.assertf = cls._assert_impl if enable else cls._null_func

    @classmethod
    @ti.func
    def _assert_impl(cls, id: ti.u32, condition: bool):#, message):
        if not condition:
            print(f"Assertion ({id:#x}) failed")

    @classmethod
    @ti.func
    def _null_func(cls, id: ti.u32, condition: bool):
        pass

    # static construction
    assertf = _null_func
    validate = False
