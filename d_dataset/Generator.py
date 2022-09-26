import enum
from random import random
import tensorflow as tf
from keras.utils.data_utils import Sequence
from keras.utils import load_img
import numpy as np
import os
from skimage import io, transform
from skimage.color import rgb2gray
import random

def sRGB_to_linear(img, normalize = False):
    if normalize == True:
        return(img/255)**2.2
    return (img)**2.2
def sRGB_to_XYZ(img):
    res = img.copy()
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.072175 ],
        [0.0193339, 0.119192 , 0.9503041]
        ])
    for i in range(img.shape[0]):
        for j in range (img.shape[1]):
            res[i,j] = np.dot(M, (img[i,j])**2.2)*100
    return res



class DataGenerator(Sequence):
    def __init__(self, data1_path, data2_path, batchSize=64, shuffle = True, seed=None, validation_split = None, subset = None) -> None:
        self.data1_path = data1_path
        self.data2_path = data2_path
        self.batchSize = batchSize
        self.subset = subset
        self.data_type = 'RGB'
        
        self.images_list1 = os.listdir(data1_path+'\\0')
        N0 = len(self.images_list1)

        self.images_list1.extend(os.listdir(data1_path+'\\1'))
        N1 = len(self.images_list1) - N0

        self.images_list2 = os.listdir(data2_path+'\\0')
        self.images_list2.extend(os.listdir(data2_path+'\\1'))

        self.labels = np.hstack((np.zeros(N0, dtype=int), np.ones(N1, dtype=int)))

        # Shuffle everything the same way
        if shuffle:
            tmp = list(zip(self.images_list1, self.images_list2, self.labels))
            if seed is not None:
                random.Random(seed).shuffle(tmp)
            else:
                random.shuffle(tmp)
            self.images_list1, self.images_list2, self.labels = zip(*tmp)

        # Create Validation dataset
        if validation_split is not None:
            n = int(len(self.images_list1) * validation_split)
            print(f"validation split {n}")
            print(n)
            if subset == 'train':
                self.images_list1 = self.images_list1[n:]
                self.images_list2 = self.images_list2[n:]
                self.labels = self.labels[n:]
            if subset == 'validation':
                self.images_list1 = self.images_list1[:n]
                self.images_list2 = self.images_list2[:n]
                self.labels = self.labels[:n]

            

        self.on_epoch_end()
    
    def set_data_type(self, data_type):
        self.data_type = data_type
    
    def process(self, img):
        """
        input shape (x, y, 3)
        output shape (x, y, 1)
        """
        def rgb2r(img):
            return img[:,:,0]
        def rgb2g(img):
            return img[:,:,1]
        def rgb2b(img):
            return img[:,:,2]
        def rgb2rg(img):
            return img[:,:,0:2]
        def rgb2gb(img):
            return img[:,:,1:3]
        def rgb2rb(img):
            return img[:,:,0:3:2]

        operations = {
            'gray' : rgb2gray,
            'R' : rgb2r,
            'G' : rgb2g,
            'B' : rgb2b,
            'RG' : rgb2rg,
            'GB' : rgb2gb,
            'RB' : rgb2rb,
            'HSV' : tf.image.rgb_to_hsv,
            'XYZ' : sRGB_to_XYZ,
        }
        return operations[self.data_type](img)

    def __len__(self):
        return int(np.ceil(len(self.images_list1) / self.batchSize))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images_list1))

    def __getitem__(self, index):
        
        indexes = self.indexes[index * self.batchSize : (index+1) * self.batchSize]

        tmp_list1 = [transform.resize(io.imread(os.path.join(self.data1_path, 
                                                                str(self.labels[k]), 
                                                                self.images_list1[k])), (31,31)) for k in indexes]

        tmp_list2 = [transform.resize(io.imread(os.path.join(self.data2_path, 
                                                                str(self.labels[k]), 
                                                                self.images_list2[k])), (63,63)) for k in indexes]
        if self.data_type != 'RGB':
            tmp_list1 = [self.process(img) for img in tmp_list1]
            tmp_list2 = [self.process(img) for img in tmp_list2]

        images1 = tf.convert_to_tensor(tmp_list1)
        images2 = tf.convert_to_tensor(tmp_list2)

        labels = tf.convert_to_tensor(self.labels[index * self.batchSize : (index+1) * self.batchSize], dtype=tf.int8)

        return [images1, images2], labels