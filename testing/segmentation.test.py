# In this file we will use the trained model to segment, then perform a diagnosis


from statistics import mode
from tkinter import image_names
from skimage import io, img_as_ubyte
from d_segmentation import Segmentation
import os
from pathlib import Path

models_path = Path("models_to_test")
models_to_test = os.listdir(models_path)
imgs_to_test = Path("img_to_test")

results_path = Path("res")
if not results_path.exists():
    os.mkdir(Path('res'))

for model in models_to_test:
    model_name = model.split('.')[0]
    data_type = model_name.split('_')[0]
    model_path = models_path / model

    print(f"---------------- USING MODEL {model}-------------------\n")
    model_result_path = results_path / model_name
    if not model_result_path.exists():
        os.mkdir(model_result_path)
    if not (model_result_path / 'EXTRA').exists():
        os.mkdir(model_result_path / 'EXTRA')

    segmentation = Segmentation(imgs_to_test, model_path, data_type)
    seg_results = segmentation.f_segmentation()

    print('Start Segmentation...........\n')
    for seg, name, sp1, sp2 in seg_results:
        print('Segmentation over ...........\n')
        
        res_path = model_result_path / name
        io.imsave(res_path, img_as_ubyte(seg))
        io.imsave(model_result_path / 'EXTRA' / f'slic_{name}', img_as_ubyte(sp1))
        io.imsave(model_result_path / 'EXTRA' / f'wat_{name}', img_as_ubyte(sp2))
        print('Start Segmentation...........\n')
