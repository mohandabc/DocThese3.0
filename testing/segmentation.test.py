# In this file we will use the trained model to segment, then perform a diagnosis


from statistics import mode
from tkinter import image_names
from skimage import io, img_as_ubyte
from d_segmentation import segmentation
import os


models_to_test = os.listdir("models_to_test\\")
img_to_test = os.listdir("img_to_test\\")
for model in models_to_test:
    model_name = model.split('.')[0]
    data_type = model_name.split('_')[0]
    model_path = os.path.join("models_to_test\\", model)
    if data_type != 'RGB':
        continue

    print(f"---------------- USING MODEL {model}-------------------\n")
    for image in img_to_test:
        print(f"====> {image}\n")
        img_name = image.split('.')[0]
        
        img_path = os.path.join("img_to_test\\", image)

        img = io.imread(img_path)
        resize_factor = round((800 / img.shape[0]), 2)
        segment_result = segmentation.segment(img = img, 
                                        model = model_path, 
                                        size=resize_factor,
                                        superpixelate_method='watershed')
                                        
        res_name = f'{model_name}_{img_name}.png'
        try:
            os.mkdir(f"res\\{model_name}\\")
        except:
            pass
        res_path = os.path.join(f"res\\{model_name}\\", res_name)
        io.imsave(res_path, img_as_ubyte(segment_result))

        # gt = io.imread(f"gts\\gt_{img_name}.JPG")

        # ret = segmentation.compute_accuracy(segmentation_result=segment_result, ground_truth=gt)
        # print(f'------->Result\n{ret}\n\n')
