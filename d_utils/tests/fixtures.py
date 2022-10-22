import pytest 
from d_utils import remove_hair
from d_segmentation import Segmentation, image_reader
from skimage import io
from skimage.transform import rescale
from pathlib import Path

IMGS_TO_TEST = Path('imgs_to_test') 
MODEL = Path('models_to_test') / 'model.h5'

@pytest.fixture
def input_img():
    img = io.imread(IMGS_TO_TEST / '2.PNG')
    rescaled = rescale(img, 0.12, channel_axis=2, anti_aliasing=True)
    return rescaled

@pytest.fixture
def output_img(input_img):
    res = remove_hair(input_img)
    return res

@pytest.fixture
def segmentation_output():
    seg = Segmentation(IMGS_TO_TEST, MODEL)
    segmentation, _, _, _ = next(seg.f_segmentation())
    return segmentation

