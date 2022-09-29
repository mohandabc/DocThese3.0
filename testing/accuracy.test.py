from d_segmentation import compute_accuracy
from skimage import io

img = io.imread(f"")
gt = io.imread(f"")

ret = compute_accuracy(segmentation_result=img, ground_truth=gt)
print(f'------->Result\n{ret}\n\n')
