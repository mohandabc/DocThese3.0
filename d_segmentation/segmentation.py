from skimage.transform import rescale, resize
from keras.models import load_model
from skimage import io
from skimage.segmentation import mark_boundaries
from tensorflow import convert_to_tensor
import numpy as np
from d_utils import utils, timer

def get_window(img, centerX, centerY):
    minX = centerX-15
    maxX = centerX + 16
    
    minX2 = centerX-31
    maxX2 = centerX + 32

    minY = centerY - 15
    maxY = centerY + 16
    
    minY2 = centerY - 31
    maxY2 = centerY + 32

    win1 = img[minX:maxX, minY:maxY]
    win2 = img[minX2:maxX2, minY2:maxY2]
    return win1,win2

def generate_windows_superpixels(img, sp_map):
    superpixels = np.sort(np.unique(sp_map))
    batch1 = []
    batch2 = []
    for c in superpixels:
        mask = sp_map==c
        x1, x2, y1, y2 = utils.find_borders(mask)
        center_x = (x1 + x2)//2 + sp_map.shape[0]
        center_y = (y1 + y2)//2 + sp_map.shape[1]
        win1,win2 = get_window(img, center_x, center_y)
        batch1.append(win1)
        batch2.append(win2)
        if len(batch1) == 64:
                batch1_t = convert_to_tensor(batch1)
                batch2_t = convert_to_tensor(batch2)
                batch1 = []
                batch2 = []
                yield [batch1_t, batch2_t], []
    if len(batch1)>0:
        batch1_t = convert_to_tensor(batch1)
        batch2_t = convert_to_tensor(batch2)
        yield [batch1_t, batch2_t], [] #Last batch may be smaller than 64


def genrate_windows(img, w, l):
    """Generate 31x31 and 63x63 windows for each pixel in img"""
    batch1 = []
    batch2 = []
    for line in range(w, 2*w):
        for col in range(l, 2*l):

            win1, win2 = get_window(img, line, col)

            batch1.append(win1)
            batch2.append(win2)
            
            if len(batch1) == 64:
                batch1_t = convert_to_tensor(batch1)
                batch2_t = convert_to_tensor(batch2)
                batch1 = []
                batch2 = []
                yield [batch1_t, batch2_t], []

    if len(batch1)>0:
        batch1_t = convert_to_tensor(batch1)
        batch2_t = convert_to_tensor(batch2)
        yield [batch1_t, batch2_t], [] #Last batch may be smaller than 64

def segment_sp(img, model : str, size : float= None, superpixelate_method = 'watershed', config = None):
    """Segments input image into 2 classes using trained model
    Classify superpixels generated with superpixelate_method
    
    Inputs:
    - img : (any, any, any) image to segment
    - model : CNN model trained for the task of classification pixels
    - size : a factor to rescale img before segmentation, result is scaled back to original
    - superpixelate_method : method to be used to create superpixel map, slic by default

    Output : segmented image
    """

    og_width = img.shape[0]
    og_length = img.shape[1]
    img = img[:, :, :3] #in case image has alpha channel

    if size != None:
        img = rescale(img, size, channel_axis=2, anti_aliasing=True)
        print('resized to : ', img.shape)

    superpixel_map = utils.superpixelate(img, superpixelate_method, config)
    expanded_img = utils.expand_img(img)
    sp_img_gen = generate_windows_superpixels(expanded_img, superpixel_map)


    cnn_model = load_model(model)
    print('Start Segmentation...........\n')
    prediction = cnn_model.predict(sp_img_gen)
    print('Segmentation over ...........\n')

    rounded = np.argmax(prediction, axis=-1)
    # rounded = []
    # for p in prediction:
    #     if 0.2 < p[0] < 0.8:
    #         rounded.append(0.5)
    #     else:
    #         rounded.append(np.argmax(p))
    

    res = np.zeros(superpixel_map.shape, dtype=float)
    for i, v in enumerate(rounded):
        res[superpixel_map == i+1] = v

    # unique, counts = np.unique(res, return_counts=True)
    # print(f"{unique}")

    if size != None:
        res = resize(res, (og_width, og_length), order=0)

    return [res, superpixel_map]

@timer
def segment(img, model : str, size : float= None):
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
    img_gen = genrate_windows(expanded_img, width, length)


    cnn_model = load_model(model)
    print('Start Segmentation...........\n')
    prediction = cnn_model.predict(img_gen)
    print('Segmentation over ...........\n')

    rounded = np.argmax(prediction, axis=-1)

    i=0
    for line in range(width):
        for col in range(length):
            img[line, col] = rounded[i]
            i+=1

    if size != None:
        img = resize(img, (og_width, og_length), order=0)

    return img

def compute_accuracy(segmentation_result, ground_truth):
    seg_res = segmentation_result[:,:,0]
    g_truth = ground_truth[:,:,0]
    w = seg_res.shape[0]
    l = seg_res.shape[1]
    N=w*l
    TP = TN = FP = FN = 0

    TP = np.count_nonzero(seg_res & g_truth)
    FP = np.count_nonzero(g_truth[seg_res >= 125]<125)
    FN = np.count_nonzero(g_truth[seg_res < 125]>=125)
    TN = N - TP - FP - FN
    return {'sensitivity' : TP/(TP+FN),
            'Specificity' : TN/(TN+FP),
            'Accuracy':(TP+TN)/N}