from tensorflow import convert_to_tensor
import numpy as np
from d_utils import utils
from pathlib import Path
from skimage import io
import os

def image_reader(path :Path):      
    img_list = os.listdir(path)
    for img_name in img_list:
        img = io.imread(path / img_name)
        
        yield img, img_name

def _get_window(img, centerX, centerY):
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
        x1, x2, y1, y2 = utils.find_borders(sp_map, c)
        center_x = (x1 + x2)//2 + sp_map.shape[0]
        center_y = (y1 + y2)//2 + sp_map.shape[1]
        win1,win2 = _get_window(img, center_x, center_y)
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


def generate_windows(img, w, l, diff_mask = None):
    """Generate 31x31 and 63x63 windows for each pixel in img"""
    batch1 = []
    batch2 = []
    for line in range(w, 2*w):
        for col in range(l, 2*l):
            if not diff_mask is None and diff_mask[line-w, col-l] == False:
                continue
            win1, win2 = _get_window(img, line, col)

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