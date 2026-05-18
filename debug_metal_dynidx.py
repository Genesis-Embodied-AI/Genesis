import quadrants as qd
import numpy as np

qd.init(arch=qd.metal)

result = qd.field(dtype=qd.f32, shape=(8,))
result2 = qd.field(dtype=qd.f32, shape=(8,))


@qd.kernel
def test_dynamic_index():
    for i in range(8):
        result[i] = qd.f32(i * 10.0)


@qd.kernel
def test_dynamic_index_local_counter():
    count = 0
    for i in range(8):
        result2[count] = qd.f32(i * 10.0)
        count += 1


@qd.kernel
def test_dynamic_index_conditional():
    count = 0
    for i in range(8):
        if i % 2 == 0:
            result[count] = qd.f32(i * 100.0)
            count += 1


test_dynamic_index()
print("Direct index:", [float(result[i]) for i in range(8)])

test_dynamic_index_local_counter()
print("Local counter:", [float(result2[i]) for i in range(8)])

for i in range(8):
    result[i] = 0.0

test_dynamic_index_conditional()
print("Conditional:", [float(result[i]) for i in range(8)])
