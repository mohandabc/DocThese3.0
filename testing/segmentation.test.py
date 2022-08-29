# In this file we will use the trained model to segment, then perform a diagnosis
# %%

from skimage import io
from d_segmentation import segmentation
# %%
img = io.imread("1.png")
segment_result = segmentation.segment(img = img, 
                                        model = "model\\model_1.h5", 
                                        size=0.2,
                                        superpixelate_method='slic')
io.imsave('res.png', segment_result)

# %%
gt = io.imread("gt.png")
ret = segmentation.compute_accuracy(segmentation_result=segment_result, ground_truth=gt)
print(ret)
