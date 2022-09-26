from skimage.transform import rescale, resize
from time import time
from d_utils import superpixelate
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2gray
from skimage import io
import numpy as np

img = io.imread('2.JPG')
img = rescale(img, 0.2, channel_axis=2, anti_aliasing=True)
gray_img = rgb2gray(img)

config = {'n_segments' : 200, 'compactness' : 5, 'sigma': 20, 'start_label': 1}
config_watershed = {'markers':100, 'compactness':0.0002}

# t = time()
# res1 = superpixelate(gray_img, 'slic')
# print(f"slick : {time() - t}")

# t = time()
# res2 = superpixelate(gray_img, 'felzenszwalb')
# print(f"felzenszwalb : {time() - t}")

# t = time()
# res3 = superpixelate(img, 'quickshift')
# print(f"quickshift : {time() - t}")

t = time()
res4 = superpixelate(img, 'watershed', config_watershed)
print(f"watershed : {time() - t}")


sp = np.unique(res4)
sp_to_process = []
for sup in sp:
    superpixel = res4 == sup
    hist, edge = np.histogram(gray_img[superpixel], bins=10, range=(0,1))
    # if(abs(hist[0] - hist[1]) < (hist[0] + hist[1])*0.1 ):
        # sp_to_process.append(sup)
    print(hist)

print(sp_to_process)
# io.imsave('slic.png', mark_boundaries(gray_img, res1))
# io.imsave('felzenszwalb.png', mark_boundaries(gray_img, res2))
# io.imsave('quickshift.png', mark_boundaries(gray_img, res3))
io.imsave('watershed.png', mark_boundaries(gray_img, res4))
