import numpy as np
from skimage.segmentation import felzenszwalb,  slic, quickshift, watershed
from time import time

def expand_img(img):
    h_flip = img.copy()[:,::-1]
    line = np.concatenate((h_flip ,img, h_flip), axis=1)

    v_flip = line.copy()[::-1, :]
    final = np.concatenate((v_flip, line, v_flip), axis=0)
    return final

# remove black edges
def remove_black_corners(img):
    """Remove the black corners of dermoscopic images.
    
    This function converts the RGB image into a grayscale image then 
    finds indecies where the image should be cropped.
    It starts from the top left corners and scans pixels diagonally 
    and deletes according lines and columns until one of two conditions 
    is met ; either the pixel value is above a predifined threshold, or a 
    predefined number of lines and columns have already been deleted.
    This last condition ensures not cropping the whole image in case of 
    large lesion that touches the borders.

    INPUTS:
    - img : and RGB image.

    OUTPUTS:
    - i, j, k, l : Indecies; crop image from column i to k and line j to l.
    """
    
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    #convert to grayscal image
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    threshold = 128
    limit = 50
    stop = False
    i = 1
    j = 1
    while not stop:
        if gray[i, j]>threshold or i == limit:
            stop = True
        else:
            i+=1
            j+=1
    stop = False
    k = -2
    l = -2
    while not stop:
        if gray[k, l]>threshold or k == -limit:
            stop = True
        else:
            k-=1
            l-=1
    
    return i, k, j, l


def superpixelate(img, method):
    """Takes an image and generates a map of superpixels using the methode 
    specified in argument

    Args:
        img (numpy array): numpy array representing the input image
        method : method to use to superpixelate image

    Returns:
        numpy array: image devided into superpixels
    """

    config = {
    "slic": {'n_segments' : 200, 'compactness' : 5, 'sigma': 50, 'start_label': 1},
    "quickshift": {'kernel_size' : 3, 'max_dist' : 6, 'ratio' : 0.5},
    "felzenszwalb": {'scale' : 50, 'sigma': 0.5, 'min_size': 50}
        }
    cfg = config[method]

    try:
        if method == 'slic':
            segments = slic(img, n_segments=cfg['n_segments'], compactness=cfg['compactness'], sigma=cfg['sigma'], start_label=cfg['start_label'])
        elif method == 'felzenszwalb':
            segments = felzenszwalb(img, scale=cfg['scale'], sigma=cfg['sigma'], min_size=cfg['min_size'])
        elif method =='quickshift':
            segments = quickshift(img, kernel_size=cfg['kernel_size'], max_dist=cfg['max_dist'], ratio=cfg['ratio'])
    except:
        raise Exception("Super pixel method failed")
    return segments

def find_borders(mask):
    """"Finds borders of the window that contains a superpixel
    
    INPUTS: 
    - mask : a mask representing a single superpixel

    OUTPUTS:
    - first_x, last_x, first_y, last_y : Coordinates of the window that contains a superpixel
    """
    
    first_x = -1
    last_x = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]: 
                last_x = i
                if first_x == -1:
                    first_x = i
                    
    first_y = -1
    last_y = 0
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if mask[j, i]: 
                last_y = i
                if first_y == -1:
                    first_y = i
    return first_x, last_x, first_y, last_y


def timer(function):
    def wrapper(*args, **kwargs):
        start = time()
        res = function(*args, **kwargs)
        end = time()
        print(f'{function.__name__}: {end - start} s')
        return res
    return wrapper