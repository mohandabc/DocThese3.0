import numpy as np
from skimage.segmentation import felzenszwalb,  slic, quickshift, watershed
from skimage import restoration,morphology, color
from skimage.color import rgb2gray
from skimage.filters import sobel
from time import time
from tensorflow import image

def remove_hair(image):
    if len(image.shape) < 3:
        print('nothing to do')
        return image
    gray = color.rgb2gray(image)
    black_hat = morphology.black_tophat(gray)
    threshold = black_hat > 0.04
    image_result = restoration.inpaint.inpaint_biharmonic(image, threshold, channel_axis=-1)
    print('hair removed')
    return image_result

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
    - i, j : Indecies; crop image from column i to j and line i to j.
    """
    
    #convert to grayscal image
    gray = img.copy()
    if len(img.shape) == 3:
        gray = rgb2gray(img)
    threshold = 0.3
    i_limit = img.shape[0]*0.6
    j_limit = img.shape[1]*0.6 
    stop = False
    i = 1
    while not stop:
        if (gray[i, i]>threshold).all() or i >= i_limit:
            stop = True
        else:
            i+=2
    stop = False
    j = -2
    while not stop:
        if (gray[j, j]>threshold).all() or j <= -j_limit:
            stop = True
        else:
            j-=2
    
    return i, j


def superpixelate(img, method, config=None):
    """Takes an image and generates a map of superpixels using the methode 
    specified in argument

    Args:
        img (numpy array): numpy array representing the input image
        method : method to use to superpixelate image

    Returns:
        numpy array: image devided into superpixels
    """

    configurations = {
    "slic": {'n_segments' : 200, 'compactness' : 5, 'sigma': 50, 'start_label': 1},
    "quickshift": {'kernel_size' : 3, 'max_dist' : 6, 'ratio' : 0.5},
    "felzenszwalb": {'scale' : 50, 'sigma': 0.5, 'min_size': 50},
    "watershed" : {'markers':250, 'compactness':0.001}
        }
    
    cfg = configurations[method]
    if config != None:
        cfg = config


    try:
        if method == 'slic':
            segments = slic(img, n_segments=cfg['n_segments'], compactness=cfg['compactness'], sigma=cfg['sigma'], start_label=cfg['start_label'])
        elif method == 'felzenszwalb':
            segments = felzenszwalb(img, scale=cfg['scale'], sigma=cfg['sigma'], min_size=cfg['min_size'])
        elif method =='quickshift':
            segments = quickshift(img, kernel_size=cfg['kernel_size'], max_dist=cfg['max_dist'], ratio=cfg['ratio'])
        elif method == 'watershed':
            segments = watershed(sobel(rgb2gray(img)), markers = cfg['markers'], compactness=cfg['compactness'])
    except:
        raise Exception("Super pixel method failed")
    return segments

def find_borders(map, superpixel):
    """"Finds borders of the window that contains a superpixel
    
    INPUTS: 
    - map : the map of superpixels generated
    - superpixel : the superpixel considered

    OUTPUTS:
    - first_x, last_x, first_y, last_y : Coordinates of the window that contains a superpixel
    """
    W = map.shape[0]
    L = map.shape[1]
    first_y = L
    last_y = 0

    first_x = W
    last_x = 0
    
    found_line = False
    for i in range(0, W, 3):
        where = np.where(map[i] == superpixel)
        len_where = len(where[0])
        
        if(len_where>0):
            found_line= True
            min_y = where[0][0] #np.min(where)
            max_y = where[0][-1] #np.max(where)
            if min_y < first_y : first_y = min_y
            if max_y > last_y : last_y = max_y

            if first_x == W : first_x = i
            if i > last_x : last_x = i
        else:
            if found_line == True:
                break
    return first_x, last_x, first_y, last_y

def find_borders_depricated(mask):
    """"Finds borders of the window that contains a superpixel
    
    INPUTS: 
    - mask : the mask of superpixels generated

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

def convert_data(img, data_type):
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
        'HSV' : image.rgb_to_hsv,
        'XYZ' : sRGB_to_XYZ,
    }
    return operations[data_type](img)