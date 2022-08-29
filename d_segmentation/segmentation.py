from skimage.transform import rescale, resize
from keras.models import load_model
from skimage import io
from tensorflow import convert_to_tensor
import numpy as np
from d_utils import utils, timer

def genrate_windows(img, w, l):
    """Generate 31x31 and 63x63 windows for each pixel in img"""
    batch1 = []
    batch2 = []
    for line in range(w, 2*w):
        for col in range(l, 2*l):

            minX = line-15
            maxX = line + 16
            
            minX2 = line-31
            maxX2 = line + 32

            minY = col - 15
            maxY = col + 16
            
            minY2 = col - 31
            maxY2 = col + 32

            win1 = img[minX:maxX, minY:maxY]
            win2 = img[minX2:maxX2, minY2:maxY2]

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
def generate_windows_for_superpxels(img, w, l):
    pass

@timer
def segment(img, model : str, size : float= None, superpixelate_method = 'slic'):
    """Segments input image into 2 classes using trained model
    
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

    superpixels = utils.superpixelate(img, superpixelate_method)    
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

def compute_accuracy(segmentation_result, ground_truth ) :
    seg_res = segmentation_result[:,:,:3]
    g_truth = ground_truth[:,:,:3]
    w = seg_res.shape[0]
    l = seg_res.shape[1]
    TP = TN = FP = FN = 0

    for i in range(w):
        for j in range(l):
            if (seg_res[i,j] == g_truth[i,j]).all() and (g_truth[i,j]>=125).all(): TP+=1
            if (seg_res[i,j] == g_truth[i,j]).all() and (g_truth[i,j]<125).all(): TN+=1
            if (seg_res[i,j] != g_truth[i,j]).all() and (g_truth[i,j]>=125).all(): FN+=1
            if (seg_res[i,j] != g_truth[i,j]).all() and (g_truth[i,j]<125).all(): FP+=1

    return {'sensitivity' : TP/(TP+FN),
            'Specificity' : TN/(TN+FP),
            'Accuracy':(TP+TN)/(TP+TN+FP+FN)}

def compute_accuracy_old(segmentation_result : str, ground_truth : str) -> float:
    seg_res = io.imread(segmentation_result)[:,:,:3]
    g_truth = io.imread(ground_truth)[:,:,:3]

    TP_TN = np.count_nonzero([seg_res == g_truth])
    FP_FN = np.count_nonzero([seg_res != g_truth])

    accuracy = TP_TN / (TP_TN + FP_FN)
    return accuracy