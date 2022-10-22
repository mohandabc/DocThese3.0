from skimage.transform import rescale, resize
from keras.models import load_model
from skimage import io, color

import numpy as np
from d_utils import utils, timer, getLargestCC
import os 
from pathlib import Path
from ._generators import image_reader, generate_windows, generate_windows_superpixels
from._utils import restore_cut_parts

class Segmentation:
    def __init__(self, imgs_path : Path = None, model_path : Path = None, data_type :str = 'RGB'):
        self.imgs_path = imgs_path
        self.data_type = data_type
        if imgs_path != None:
            self.img_reader = image_reader(self.imgs_path)
        self.model_path = model_path
        if model_path != None:
            self.model = load_model(model_path)
        # self.superpixel_method

    def f_segmentation(self):
        """F_segmentation impliments the complete methode and it works on a folder of images not one image
        It segments image using two quick superpixel methods, then computes difference and intersection
        The difference pixels are classifed again pixel by pixel. this result replaces the frst difference and
        added to the intersection to create final segmentation
        """
        
        config_watershed = {'markers':50, 'compactness':0.0002}
        config_slic ={'n_segments' : 50, 'compactness' : 1, 'sigma': 50, 'start_label': 1}

        resize_factor = None
        for image, img_name in self.img_reader:
            print(f'==+>{img_name}\n')
            i_cut, j_cut = utils.remove_black_corners(image)
            image = image[i_cut:j_cut, i_cut:j_cut]

            og_image_shape = image.shape
            # if image height is higher than 400px reduce size to 400px, keep ratio 
            if image.shape[0]>300:
                resize_factor = round((300 / image.shape[0]), 2)
                image = rescale(image, resize_factor, channel_axis=2, anti_aliasing=True)
                print('resized to : ', image.shape)
            
            image = utils.remove_hair(image)
            segment_result_slic, sp_map1 = self.segment_sp(img = image, 
                                            model = self.model, 
                                            superpixelate_method='slic',
                                            config=config_slic)
            segment_result_wat, sp_map2 = self.segment_sp(img = image, 
                                            model = self.model, 
                                            superpixelate_method='watershed',
                                            config=config_watershed)

            difference = segment_result_slic != segment_result_wat
            
            intersection = (segment_result_slic==1) & (segment_result_wat ==1)
            # io.imsave(Path('res') / 'EXTRA' / f'diff_{img_name}', img_as_ubyte(difference))
            # io.imsave(Path('res') / 'EXTRA' / f'inter_{img_name}', img_as_ubyte(intersection))
            
            # rgb_image = image.copy()
            if self.data_type != 'RGB':
                # image = color.convert_colorspace(image, 'RGB', self.data_type)
                image = utils.convert_data(image, self.data_type)
            expanded_img = utils.expand_img(image)
            difference_gen = generate_windows(expanded_img, image.shape[0], image.shape[1], difference)
            # if len(difference_gen)<=0:
            try:
                predictions = self.model.predict(difference_gen)
            except:
                yield intersection, img_name, segment_result_slic, segment_result_wat
                continue

            rounded = np.argmax(predictions, axis=-1)
            print(f"{len(rounded)} Pixels reclassified")

            # merge the intersection result with classification result of the difference

            i=0
            for line in range(intersection.shape[0]):
                for col in range(intersection.shape[1]):
                    if difference[line, col] == True:
                        intersection[line, col] = rounded[i]
                        i+=1
                        if(i>=len(rounded)):
                            break
                if(i>=len(rounded)):
                    break

            res = getLargestCC(intersection)
            # if size != None:
            res = resize(res, og_image_shape, order=0)

            segmentation = restore_cut_parts(res, i_cut, j_cut)
            yield segmentation, img_name, segment_result_slic, segment_result_wat
    

    def segment_sp(self, img, model, superpixelate_method = 'watershed', config = None):
        """Segments input image into 2 classes using trained model
        Classify superpixels generated with superpixelate_method
        
        Inputs:
        - img : (any, any, any) image to segment
        - model : CNN model trained for the task of classification pixels
        - superpixelate_method : method to be used to create superpixel map, slic by default

        Output : segmented image
        """
        if len(img.shape)>3:
            img = img[:, :, :3] #in case image has alpha channel

        superpixel_map = utils.superpixelate(img, superpixelate_method, config)
        if self.data_type != 'RGB':
            # img = color.convert_colorspace(img, 'RGB', self.data_type)
            img = utils.convert_data(img, self.data_type)
        expanded_img = utils.expand_img(img)
        sp_img_gen = generate_windows_superpixels(expanded_img, superpixel_map)

        prediction = model.predict(sp_img_gen)

        rounded = np.argmax(prediction, axis=-1)     

        res = np.zeros(superpixel_map.shape, dtype=float)
        for i, v in enumerate(rounded):
            res[superpixel_map == i+1] = v       

        return [res, superpixel_map]

    @timer
    def segment(self, img, model : str, size : float= None):
        """Segments input image into 2 classes using trained model
        Classify each pixel of the image
        
        Inputs:
        - img : (any, any, any) image to segment
        - model : CNN model trained for the task of classification pixels
        - size : a factor to rescale img before segmentation, result is scaled back to original

        Output : segmented image
        """



        width = og_width = img.shape[0]
        length = og_length = img.shape[1]
        img = img[:, :, :3] #in case image has alpha channel

        if size != None:
            img = rescale(img, size, channel_axis=2, anti_aliasing=True)
            width = img.shape[0]
            length  = img.shape[1]
            print('resized to : ', img.shape)

        expanded_img = utils.expand_img(img)
        img_gen = generate_windows(expanded_img, width, length)


        cnn_model = load_model(model)
        prediction = cnn_model.predict(img_gen)

        rounded = np.argmax(prediction, axis=-1)

        i=0
        for line in range(width):
            for col in range(length):
                img[line, col] = rounded[i]
                i+=1

        if size != None:
            img = resize(img, (og_width, og_length), order=0)

        return img
