from d_utils import superpixelate

from skimage import io

img = io.imread('1.png')

res = superpixelate(img, 'slic')

io.imsave('res.png', res)
