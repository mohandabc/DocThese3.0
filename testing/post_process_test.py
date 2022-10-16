from skimage.measure import label
from skimage import io, img_as_ubyte
import numpy as np
from pathlib import Path
import os 

def getLargestCC(segmentation):
    labels, _ = label(segmentation, background=0, return_num=True, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC


base_path = Path('res')

res_folders = [dir for dir in os.listdir(base_path) if (base_path/dir).is_dir()]

for model_results in res_folders:
    results = [img for img in os.listdir(base_path/model_results) if not (base_path/model_results/img).is_dir()]
    for res in results:
        res_path = base_path / model_results / res
        res_img = io.imread(res_path)
        output = getLargestCC(res_img)
        io.imsave(base_path / model_results / f'processed_{res}', img_as_ubyte(output))

