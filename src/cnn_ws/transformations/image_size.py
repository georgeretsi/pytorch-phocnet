'''
Created on Sep 3, 2017

@author: ssudholt
'''
import numpy as np
from skimage.transform import resize

def check_size(img, min_image_width_height, fixed_image_size=None):
    '''
    checks if the image accords to the minimum and maximum size requirements
    or fixed image size and resizes if not
    
    :param img: the image to be checked
    :param min_image_width_height: the minimum image size
    :param fixed_image_size:
    '''
    if fixed_image_size is not None:
        if len(fixed_image_size) != 2:
            raise ValueError('The requested fixed image size is invalid!')
        new_img = resize(image=img, output_shape=fixed_image_size[::-1])
        new_img = new_img.astype(np.float32)
        return new_img
    elif np.amin(img.shape[:2]) < min_image_width_height:
        if np.amin(img.shape[:2]) == 0:
            return None
        scale = float(min_image_width_height + 1) / float(np.amin(img.shape[:2]))
        new_shape = (int(scale * img.shape[0]), int(scale * img.shape[1]))
        new_img = resize(image=img, output_shape=new_shape)
        new_img = new_img.astype(np.float32)
        return new_img
    else:
        return img