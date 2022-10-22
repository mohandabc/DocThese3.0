from d_utils import remove_hair, getLargestCC, expand_img
import numpy as np
import pytest

from d_utils.utils import expand_img

def test_remove_hair_dtype():
    x = np.zeros((3, 3))
    x_f32 = x.astype(np.float32)
    x_u8 = x.astype(np.uint8)
    x_b = x.astype(bool)

    assert remove_hair(x).dtype == np.float64
    assert remove_hair(x_f32).dtype == np.float64
    assert remove_hair(x_u8).dtype == np.float64
    with pytest.raises(Exception) as e_info:
        remove_hair(x_b)

    for i in range(1,4):
        x = np.zeros((3, 3, i))
        x_f32 = x.astype(np.float32)
        x_u8 = x.astype(np.uint8)
        x_b = x.astype(bool)

        assert remove_hair(x).dtype == np.float64
        assert remove_hair(x_f32).dtype == np.float64
        assert remove_hair(x_u8).dtype == np.float64
        with pytest.raises(Exception) as e_info:
            remove_hair(x_b)

def test_largestCC():
    for i in range(1,4):
        x = np.zeros((3, 3, i))
        x_f32 = x.astype(np.float32)
        x_u8 = x.astype(np.uint8)
        x_b = x.astype(bool)

        assert getLargestCC(x).dtype == np.uint8
        assert getLargestCC(x_f32).dtype == np.uint8
        assert getLargestCC(x_u8).dtype == np.uint8
        assert getLargestCC(x_b).dtype == np.uint8

def test_expand_img():
    x = np.zeros((3, 3))
    assert expand_img(x).shape == (9, 9)

    x = np.zeros((3, 3, 1))
    assert expand_img(x).shape == (9, 9, 1)

    x = np.zeros((3, 3, 3))
    assert expand_img(x).shape == (9, 9, 3)