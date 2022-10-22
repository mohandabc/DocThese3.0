import numpy as np
from fixtures import IMGS_TO_TEST, segmentation_output

from d_segmentation import image_reader

def test_segmentation_shape(segmentation_output):
    assert segmentation_output.shape == (1,1,1)

def test_segmentation_dtype(segmentation_output):
    assert segmentation_output.dtype == (1,1,1)

def test_image_reader():
    res = next(image_reader(IMGS_TO_TEST))
    assert res.dtype == np.float64

