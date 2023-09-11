import numpy as np
import sys
import pytest
import os
import platform
if platform.system() == 'Windows':
    os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from num_boost import *


def assert_operation(a, b, array):
    assert np.allclose(a.data, b, equal_nan=True)
    a_ = Tensor(array)
    predict_type = result_type(Add, a.dtype,
                               a.data.itemsize, a_.dtype, a_.data.itemsize)
    o = a.astype(predict_type) + a_.astype(predict_type)
    o1 = b.astype(nb_type_2_np[predict_type]) + array.astype(nb_type_2_np[predict_type])
    indice = np.where(o.data != o1.astype(o.data.dtype))
    assert np.allclose(o.data, o1.astype(o.data.dtype), equal_nan=True), (f"predicted type: {predict_type},"
                                                                          f"a type: {a.data.dtype},"
                                                                          f" {a_.data.dtype}\n"
                                                                          f" {nb_type_2_np[predict_type]}\n"
                                                                          f"tensor diff: {o[indice]}\n\n"
                                                                          f"array diff: {o1[indice]}")
    predict_type = result_type(Sub, a.dtype,
                               a.data.itemsize, a_.dtype, a_.data.itemsize)
    o = a.astype(predict_type) - a_.astype(predict_type)
    o1 = b.astype(nb_type_2_np[predict_type]) - array.astype(nb_type_2_np[predict_type])
    indice = np.where(o.data != o1.astype(o.data.dtype))
    assert np.allclose(o.data, o1.astype(o.data.dtype), equal_nan=True), (f"predicted type: {predict_type},"
                                                                          f"a type: {a.data.dtype},"
                                                                          f" {a_.data.dtype}\n"
                                                                          f" {nb_type_2_np[predict_type]}\n"
                                                                          f"tensor diff: {o[indice]}\n\n"
                                                                          f"array diff: {o1[indice]}")
    predict_type = result_type(Mul, a.dtype,
                               a.data.itemsize, a_.dtype, a_.data.itemsize)
    o = a.astype(predict_type) * a_.astype(predict_type)
    o1 = b.astype(nb_type_2_np[predict_type]) * array.astype(nb_type_2_np[predict_type])
    indice = np.where(o.data != o1.astype(o.data.dtype))
    assert np.allclose(o.data, o1.astype(o.data.dtype), equal_nan=True), (f"predicted type: {predict_type},"
                                                                          f"a type: {a.data.dtype},"
                                                                          f" {a_.data.dtype}\n"
                                                                          f" {nb_type_2_np[predict_type]}\n"
                                                                          f"tensor diff: {o[indice]}\n\n"
                                                                          f"array diff: {o1[indice]}")
    predict_type = result_type(Div, a.dtype, a.data.itemsize,
                               a_.dtype, a_.data.itemsize)
    o = a.astype(predict_type) / a_.astype(predict_type)
    o1 = b.astype(nb_type_2_np[predict_type]) / array.astype(nb_type_2_np[predict_type])
    indice = np.where(o.data != o1.astype(o.data.dtype))
    assert np.allclose(o.data, o1.astype(o.data.dtype), equal_nan=True), (f"predicted type: {predict_type},"
                                                                          f"a type: {a.data.dtype},"
                                                                          f" {a_.data.dtype}\n"
                                                                          f" {nb_type_2_np[predict_type]}\n"
                                                                          f"tensor diff: {o[indice]}\n\n"
                                                                          f"array diff: {o1[indice]}")


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float16)])
def test_C_Tensor_float16_2_float32(array):
    a = Tensor(array).astype(float32)
    b = np.array(array).astype(np.float32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float32)])
def test_C_Tensor_float32_2_float32(array):
    a = Tensor(array).astype(float32)
    b = np.array(array).astype(np.float32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float64)])
def test_C_Tensor_float64_2_float32(array):
    a = Tensor(array).astype(float32)
    b = np.array(array).astype(np.float32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


# @pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.longdouble)])
# def test_C_Tensor_float128_2_float32(array):
#     a = Tensor(array).astype(float32)
#     b = np.array(array).astype(np.float32)
#     assert_operation(a, b, array)
#     assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.int8)])
def test_C_Tensor_int8_2_float32(array):
    a = Tensor(array).astype(float32)
    b = np.array(array).astype(np.float32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.int16)])
def test_C_Tensor_int16_2_float32(array):
    a = Tensor(array).astype(float32)
    b = np.array(array).astype(np.float32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 50, 3).astype(np.int32)])
def test_C_Tensor_int32_2_float32(array):
    a = Tensor(array).astype(float32)
    b = np.array(array).astype(np.float32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 50, 3).astype(np.int64)])
def test_C_Tensor_int64_2_float32(array):
    a = Tensor(array).astype(float32)
    b = np.array(array).astype(np.float32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 50, 3)).astype(np.uint8)])
def test_C_Tensor_uint8_2_float32(array):
    a = Tensor(array).astype(float32)
    b = np.array(array).astype(np.float32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 50, 3)).astype(np.uint16)])
def test_C_Tensor_uint16_2_float32(array):
    a = Tensor(array).astype(float32)
    b = np.array(array).astype(np.float32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 50, 3)).astype(np.uint32)])
def test_C_Tensor_uint32_2_float32(array):
    a = Tensor(array).astype(float32)
    b = np.array(array).astype(np.float32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 50, 3)).astype(np.uint64)])
def test_C_Tensor_uint64_2_float32(array):
    a = Tensor(array).astype(float32)
    b = np.array(array).astype(np.float32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float32)])
def test_C_Tensor_float32_2_float64(array):
    a = Tensor(array).astype(float64)
    b = np.array(array).astype(np.float64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, rtol=0.01, atol=0.01, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float16)])
def test_C_Tensor_float16_2_float64(array):
    a = Tensor(array).astype(float64)
    b = np.array(array).astype(np.float64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float64)])
def test_C_Tensor_float64_2_float64(array):
    a = Tensor(array).astype(float64)
    b = np.array(array).astype(np.float64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, rtol=0.01, atol=0.01, equal_nan=True)


# @pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.longdouble)])
# def test_C_Tensor_float128_2_float64(array):
#     a = Tensor(array).astype(float64)
#     b = np.array(array).astype(np.float64)
#     assert np.allclose(a.data, b, rtol=0.01, atol=0.01, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.int8)])
def test_C_Tensor_int8_2_float64(array):
    a = Tensor(array).astype(float64)
    b = np.array(array).astype(np.float64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.int16)])
def test_C_Tensor_int16_2_float64(array):
    a = Tensor(array).astype(float64)
    b = np.array(array).astype(np.float64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.int32)])
def test_C_Tensor_int32_2_float64(array):
    a = Tensor(array).astype(float64)
    b = np.array(array).astype(np.float64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.int64)])
def test_C_Tensor_int64_2_float64(array):
    a = Tensor(array).astype(float64)
    b = np.array(array).astype(np.float64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint8)])
def test_C_Tensor_uint8_2_float64(array):
    a = Tensor(array).astype(float64)
    b = np.array(array).astype(np.float64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint16)])
def test_C_Tensor_uint16_2_float64(array):
    a = Tensor(array).astype(float64)
    b = np.array(array).astype(np.float64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint32)])
def test_C_Tensor_uint32_2_float64(array):
    a = Tensor(array).astype(float64)
    b = np.array(array).astype(np.float64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint64)])
def test_C_Tensor_uint64_2_float64(array):
    a = Tensor(array).astype(float64)
    b = np.array(array).astype(np.float64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float32)])
def test_C_Tensor_float32_2_float16(array):
    a = Tensor(array).astype(float16)
    b = np.array(array).astype(np.float16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float16)])
def test_C_Tensor_float16_2_float16(array):
    a = Tensor(array).astype(float16)
    b = np.array(array).astype(np.float16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float64)])
def test_C_Tensor_float64_2_float16(array):
    a = Tensor(array).astype(float16)
    b = np.array(array).astype(np.float16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


# @pytest.mark.parametrize("array", [np.random.randn(200, 500).astype(np.float64)])
# def test_C_Tensor_float128_2_float16(array):
#     a = Tensor(array).astype(float16)
#     b = np.array(array).astype(np.float16)
#     assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.int8)])
def test_C_Tensor_int8_2_float16(array):
    a = Tensor(array).astype(float16)
    b = np.array(array).astype(np.float16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.int16)])
def test_C_Tensor_int16_2_float16(array):
    a = Tensor(array).astype(float16)
    b = np.array(array).astype(np.float16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.int32)])
def test_C_Tensor_int32_2_float16(array):
    a = Tensor(array).astype(float16)
    b = np.array(array).astype(np.float16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.arange(20000).astype(np.int64)])
def test_C_Tensor_int64_2_float16(array):
    a = Tensor(array).astype(float16)
    b = np.array(array).astype(np.float16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint8)])
def test_C_Tensor_uint8_2_float16(array):
    a = Tensor(array).astype(float16)
    b = np.array(array).astype(np.float16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint16)])
def test_C_Tensor_uint16_2_float16(array):
    a = Tensor(array).astype(float16)
    b = np.array(array).astype(np.float16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint32)])
def test_C_Tensor_uint32_2_float16(array):
    a = Tensor(array).astype(float16)
    b = np.array(array).astype(np.float16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint64)])
def test_C_Tensor_uint64_2_float16(array):
    a = Tensor(array).astype(float16)
    b = np.array(array).astype(np.float16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.int8)])
def test_C_Tensor_int8_2_int16(array):
    a = Tensor(array).astype(int16)
    b = np.array(array).astype(np.int16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.int16)])
def test_C_Tensor_int16_2_int16(array):
    a = Tensor(array).astype(int16)
    b = np.array(array).astype(np.int16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.int32)])
def test_C_Tensor_int32_2_int16(array):
    a = Tensor(array).astype(int16)
    b = np.array(array).astype(np.int16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.int64)])
def test_C_Tensor_int64_2_int16(array):
    a = Tensor(array).astype(int16)
    b = np.array(array).astype(np.int16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint8)])
def test_C_Tensor_uint8_2_int16(array):
    a = Tensor(array).astype(int16)
    b = np.array(array).astype(np.int16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint16)])
def test_C_Tensor_uint16_2_int16(array):
    a = Tensor(array).astype(int16)
    b = np.array(array).astype(np.int16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint32)])
def test_C_Tensor_uint32_2_int16(array):
    a = Tensor(array).astype(int16)
    b = np.array(array).astype(np.int16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint64)])
def test_C_Tensor_uint64_2_int16(array):
    a = Tensor(array).astype(int16)
    b = np.array(array).astype(np.int16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float16)])
def test_C_Tensor_float16_2_int16(array):
    a = Tensor(array).astype(int16)
    b = np.array(array).astype(np.int16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float32)])
def test_C_Tensor_float32_2_int16(array):
    a = Tensor(array).astype(int16)
    b = np.array(array).astype(np.int16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float64)])
def test_C_Tensor_float64_2_int16(array):
    a = Tensor(array).astype(int16)
    b = np.array(array).astype(np.int16)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.int8)])
def test_C_Tensor_int8_2_int32(array):
    a = Tensor(array).astype(int32)
    b = np.array(array).astype(np.int32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.int16)])
def test_C_Tensor_int16_2_int32(array):
    a = Tensor(array).astype(int32)
    b = np.array(array).astype(np.int32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.int32)])
def test_C_Tensor_int32_2_int32(array):
    a = Tensor(array).astype(int32)
    b = np.array(array).astype(np.int32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.int64)])
def test_C_Tensor_int64_2_int32(array):
    a = Tensor(array).astype(int32)
    b = np.array(array).astype(np.int32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint8)])
def test_C_Tensor_uint8_2_int32(array):
    a = Tensor(array).astype(int32)
    b = np.array(array).astype(np.int32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint16)])
def test_C_Tensor_uint16_2_int32(array):
    a = Tensor(array).astype(int32)
    b = np.array(array).astype(np.int32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint32)])
def test_C_Tensor_uint32_2_int32(array):
    a = Tensor(array).astype(int32)
    b = np.array(array).astype(np.int32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint64)])
def test_C_Tensor_uint64_2_int32(array):
    a = Tensor(array).astype(int32)
    b = np.array(array).astype(np.int32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float16)])
def test_C_Tensor_float16_2_int32(array):
    a = Tensor(array).astype(int32)
    b = np.array(array).astype(np.int32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float32)])
def test_C_Tensor_float32_2_int32(array):
    a = Tensor(array).astype(int32)
    b = np.array(array).astype(np.int32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float64)])
def test_C_Tensor_float64_2_int32(array):
    a = Tensor(array).astype(int32)
    b = np.array(array).astype(np.int32)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.int8)])
def test_C_Tensor_int8_2_int64(array):
    a = Tensor(array).astype(int64)
    b = np.array(array).astype(np.int64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.int16)])
def test_C_Tensor_int16_2_int64(array):
    a = Tensor(array).astype(int64)
    b = np.array(array).astype(np.int64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.int32)])
def test_C_Tensor_int32_2_int64(array):
    a = Tensor(array).astype(int64)
    b = np.array(array).astype(np.int64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.int64)])
def test_C_Tensor_int64_2_int64(array):
    a = Tensor(array).astype(int64)
    b = np.array(array).astype(np.int64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint8)])
def test_C_Tensor_uint8_2_int64(array):
    a = Tensor(array).astype(int64)
    b = np.array(array).astype(np.int64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint16)])
def test_C_Tensor_uint16_2_int64(array):
    a = Tensor(array).astype(int64)
    b = np.array(array).astype(np.int64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint32)])
def test_C_Tensor_uint32_2_int64(array):
    a = Tensor(array).astype(int64)
    b = np.array(array).astype(np.int64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(0, 100, size=(200, 500, 3)).astype(np.uint64)])
def test_C_Tensor_uint64_2_int64(array):
    a = Tensor(array).astype(int64)
    b = np.array(array).astype(np.int64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float16)])
def test_C_Tensor_float16_2_int64(array):
    a = Tensor(array).astype(int64)
    b = np.array(array).astype(np.int64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float32)])
def test_C_Tensor_float32_2_int64(array):
    a = Tensor(array).astype(int64)
    b = np.array(array).astype(np.int64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randn(200, 500, 3).astype(np.float64)])
def test_C_Tensor_float64_2_int64(array):
    a = Tensor(array).astype(int64)
    b = np.array(array).astype(np.int64)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int8)])
def test_C_Tensor_int8_2_uint(array):
    a = Tensor(array).astype(uint)
    b = np.array(array).astype(np.uint)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int16)])
def test_C_Tensor_int16_2_uint(array):
    a = Tensor(array).astype(uint)
    b = np.array(array).astype(np.uint)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int32)])
def test_C_Tensor_int32_2_uint(array):
    a = Tensor(array).astype(uint)
    b = np.array(array).astype(np.uint)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int64)])
def test_C_Tensor_int64_2_uint(array):
    a = Tensor(array).astype(uint)
    b = np.array(array).astype(np.uint)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int_)])
def test_C_Tensor_int_2_uint(array):
    a = Tensor(array).astype(uint)
    b = np.array(array).astype(np.uint)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float16)])
def test_C_Tensor_float16_2_uint(array):
    a = Tensor(array).astype(uint)
    b = np.array(array).astype(np.uint)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float32)])
def test_C_Tensor_float32_2_uint(array):
    a = Tensor(array).astype(uint)
    b = np.array(array).astype(np.uint)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float64)])
def test_C_Tensor_float64_2_uint(array):
    a = Tensor(array).astype(uint)
    b = np.array(array).astype(np.uint)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int8)])
def test_C_Tensor_int8_2_short(array):
    a = Tensor(array).astype(short)
    b = np.array(array).astype(np.short)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int16)])
def test_C_Tensor_int16_2_short(array):
    a = Tensor(array).astype(short)
    b = np.array(array).astype(np.short)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int32)])
def test_C_Tensor_int32_2_short(array):
    a = Tensor(array).astype(short)
    b = np.array(array).astype(np.short)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int64)])
def test_C_Tensor_int64_2_short(array):
    a = Tensor(array).astype(short)
    b = np.array(array).astype(np.short)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int_)])
def test_C_Tensor_int_2_short(array):
    a = Tensor(array).astype(short)
    b = np.array(array).astype(np.short)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float16)])
def test_C_Tensor_float16_2_short(array):
    a = Tensor(array).astype(short)
    b = np.array(array).astype(np.short)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float32)])
def test_C_Tensor_float32_2_short(array):
    a = Tensor(array).astype(short)
    b = np.array(array).astype(np.short)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float64)])
def test_C_Tensor_float64_2_short(array):
    a = Tensor(array).astype(short)
    b = np.array(array).astype(np.short)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int8)])
def test_C_Tensor_int8_2_int(array):
    a = Tensor(array).astype(int_)
    b = np.array(array).astype(np.int_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int16)])
def test_C_Tensor_int16_2_int(array):
    a = Tensor(array).astype(int_)
    b = np.array(array).astype(np.int_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int32)])
def test_C_Tensor_int32_2_int(array):
    a = Tensor(array).astype(int_)
    b = np.array(array).astype(np.int_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int64)])
def test_C_Tensor_int64_2_int(array):
    a = Tensor(array).astype(int_)
    b = np.array(array).astype(np.int_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int_)])
def test_C_Tensor_int_2_int(array):
    a = Tensor(array).astype(int_)
    b = np.array(array).astype(np.int_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float16)])
def test_C_Tensor_float16_2_int(array):
    a = Tensor(array).astype(int_)
    b = np.array(array).astype(np.int_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float32)])
def test_C_Tensor_float32_2_int(array):
    a = Tensor(array).astype(int_)
    b = np.array(array).astype(np.int_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float64)])
def test_C_Tensor_float64_2_int(array):
    a = Tensor(array).astype(int_)
    b = np.array(array).astype(np.int_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int8)])
def test_C_Tensor_int8_2_ushort(array):
    a = Tensor(array).astype(ushort)
    b = np.array(array).astype(np.ushort)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int16)])
def test_C_Tensor_int16_2_ushort(array):
    a = Tensor(array).astype(ushort)
    b = np.array(array).astype(np.ushort)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int32)])
def test_C_Tensor_int32_2_ushort(array):
    a = Tensor(array).astype(ushort)
    b = np.array(array).astype(np.ushort)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int64)])
def test_C_Tensor_int64_2_ushort(array):
    a = Tensor(array).astype(ushort)
    b = np.array(array).astype(np.ushort)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int_)])
def test_C_Tensor_int_2_ushort(array):
    a = Tensor(array).astype(ushort)
    b = np.array(array).astype(np.ushort)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float16)])
def test_C_Tensor_float16_2_ushort(array):
    a = Tensor(array).astype(ushort)
    b = np.array(array).astype(np.ushort)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float32)])
def test_C_Tensor_float32_2_ushort(array):
    a = Tensor(array).astype(ushort)
    b = np.array(array).astype(np.ushort)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float64)])
def test_C_Tensor_float64_2_ushort(array):
    a = Tensor(array).astype(ushort)
    b = np.array(array).astype(np.ushort)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int8)])
def test_C_Tensor_int8_2_bool_(array):
    a = Tensor(array).astype(bool_)
    b = np.array(array).astype(np.bool_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int16)])
def test_C_Tensor_int16_2_bool_(array):
    a = Tensor(array).astype(bool_)
    b = np.array(array).astype(np.bool_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int32)])
def test_C_Tensor_int32_2_bool_(array):
    a = Tensor(array).astype(bool_)
    b = np.array(array).astype(np.bool_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int64)])
def test_C_Tensor_int64_2_bool_(array):
    a = Tensor(array).astype(bool_)
    b = np.array(array).astype(np.bool_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.int_)])
def test_C_Tensor_int_2_bool_(array):
    a = Tensor(array).astype(bool_)
    b = np.array(array).astype(np.bool_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float16)])
def test_C_Tensor_float16_2_bool_(array):
    a = Tensor(array).astype(bool_)
    b = np.array(array).astype(np.bool_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float32)])
def test_C_Tensor_float32_2_bool_(array):
    a = Tensor(array).astype(bool_)
    b = np.array(array).astype(np.bool_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)


@pytest.mark.parametrize("array", [np.random.randint(-100, 100, size=(200, 500, 3)).astype(np.float64)])
def test_C_Tensor_float64_2_bool_(array):
    a = Tensor(array).astype(bool_)
    b = np.array(array).astype(np.bool_)
    assert_operation(a, b, array)
    assert np.allclose(a.data, b, equal_nan=True)