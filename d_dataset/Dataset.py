from random import random
import os
from skimage import io, img_as_ubyte
import numpy as np
from skimage.segmentation import mark_boundaries

from d_utils import remove_black_corners, superpixelate, find_borders
CONFIG = {

    "nb_processed_imgs" : 200,
    "n_levels" : 2, 
    "levels":[
             (31, 31), 
             (63, 63), 
             (127, 127), 
             (255, 255), 
            ],
    "train_split": 0.7,  #70% off all images will be in train folder, 30% in test folder
    "batch_size" : 64,
        }

class Dataset():

    """This class prprocesses images to create datasets to feed CNN"""

    metadata = {}
    current_image_name = 0
    current_image = []
    current_lesion = []
    output_path = ''
    
    def __init__(self, images_folder = "", output_folder="..\\Data\\Dataset\\"):
        self.images_path = images_folder
        self.output_folder = output_folder
        self.config = CONFIG

    def set_config(self, key, value):
        self.config[key] = value
 
    def process(self):
        # get images to process (limit : nb_processed_imgs)
        images = (next(os.walk(self.images_path))[1])[0:self.config['nb_processed_imgs']]
        for image in images:
            print('\nprocessing ', image)
            image_path = self.images_path + image + '\\' + image+'_Dermoscopic_Image\\' +image+'.bmp' 
            mask_path = self.images_path + image + '\\' + image+'_lesion\\' +image+"_lesion.bmp" 

            # Read the image and the correspondant mask
            img = io.imread(image_path,plugin='matplotlib')
            lesion_mask = io.imread(mask_path, plugin='matplotlib')

            # Remove the black corners
            # Resize the image and the mask to ignore the black corners
            i,j, k, l = remove_black_corners(img)
            self.current_image = img[i:j, k:l, :]
            self.current_lesion = lesion_mask[i:j, k:l]

            segments = superpixelate(self.current_image, 'slic')

            # Create the output directory and the files names
            image_name = image_path.split('\\')[-1].split(".")[0]
            self.output_path = self.output_folder + image_name +'\\' 
            try:
                os.makedirs(self.output_path)
            except:
                pass
            output_image_name = image_name+'.png'
            output_mask_name = image_name+'_lesion.png'

            # Save the new resized image, mask and a visual representation of the superpixels
            io.imsave(self.output_path + output_image_name, img_as_ubyte(img))
            io.imsave(self.output_path + output_mask_name, img_as_ubyte(self.current_lesion))
            io.imsave(self.output_path + 'spx_' + output_image_name, mark_boundaries(self.current_image, segments))
            
            self.save_windows(segments)
            print("number of windows : " , len(np.unique(segments)))
    
    def save_windows(self, segments):

        n_levels = self.config['n_levels']+1
        for level in range(1, n_levels):
            try:
                base_path = f"{self.output_folder}Dataset\\data{str(level)}"
                os.makedirs(f'{base_path}\\train\\0')
                os.makedirs(f'{base_path}\\train\\1')
                os.makedirs(f'{base_path}\\test\\0')
                os.makedirs(f'{base_path}\\test\\1')
            except:
                pass
        
        unique_segments = np.unique(segments)
        for c in unique_segments:
            rand = random()
            folder = 'test'
            if rand < self.config['train_split']:
                folder = 'train'
            sp_class = 0
            self.current_image_name += 1
            
            # these variables are used to save the start x, y and end x,y 
            # of the first window (ref window)
            base_first_x = 0
            base_last_x = 0
            base_first_y = 0
            base_last_y = 0
            for level in range(1, n_levels):
                if level == 1:
                    #LEVEL 1
                    # mask = segments==c
                    first_x, last_x, first_y, last_y = find_borders(segments, c)
                    # The length (last_x - first_x) should be odd in order to have a center
                    # Same for (last_y - first_y)
                    if((last_x - first_x +1)%2) == 0:
                        last_x -= 1
                    if((last_y - first_y +1)%2) == 0:
                        last_y -= 1
                    base_first_x, base_last_x, base_first_y, base_last_y = first_x, last_x, first_y, last_y

                    # Find the classification of the center
                    sp_class = self.current_lesion[first_x + (last_x-first_x)//2, first_y + (last_y-first_y)//2, 0]//255
                else:
                    # FIXME: Manage the border oveflow
                    # FIXME: Make sur legths are odd
                    first_x = int(base_first_x - ((base_last_x-base_first_x)/2*(level-1)))
                    if first_x < 0 : first_x = 0

                    last_x = base_last_x + ((base_last_x-base_first_x)/2*(level-1))
                    if last_x > self.current_image.shape[0] : last_x = self.current_image.shape[0]-1

                    first_y = int(base_first_y - ((base_last_y-base_first_y)/2*(level-1)))
                    if first_y < 0 : first_y = 0
                    last_y = base_last_y + ((base_last_y-base_first_y)/2*(level-1))
                    if last_y > self.current_image.shape[1] : last_y = self.current_image.shape[1]-1
                    
                    # first_x = first_x.item()
                    last_x = int(last_x)
                    # first_y = first_y.item()
                    last_y = int(last_y)

                res = self.current_image[first_x:last_x+1, first_y:last_y+1, :]
                io.imsave(f'{self.output_folder}\\Dataset\\data{level}\\{folder}\\{round(sp_class)}\\{self.current_image_name}.png', img_as_ubyte(res))

            
    