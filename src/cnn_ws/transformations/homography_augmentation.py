'''
Created on Sep 3, 2017

@author: ssudholt
'''

import numpy as np
import cv2

class HomographyAugmentation(object):
    '''
    Class for creating homography augmentation transformations in the pytorch
    framework.
    '''

    def __init__(self, random_limits=(0.9, 1.1)):
        '''
        Constructor
        '''
        self.random_limits = random_limits

    def __call__(self, img):
        '''
        Creates an augmentation by computing a homography from three
        points in the image to three randomly generated points
        '''
        y, x = img.shape[:2]
        fx = float(x)
        fy = float(y)
        src_point = np.float32([[fx/2, fy/3,],
                                [2*fx/3, 2*fy/3],
                                [fx/3, 2*fy/3]])
        random_shift = (np.random.rand(3,2) - 0.5) * 2 * (self.random_limits[1]-self.random_limits[0])/2 + np.mean(self.random_limits)
        dst_point = src_point * random_shift.astype(np.float32)
        transform = cv2.getAffineTransform(src_point, dst_point)
        #border_value = 0
        if img.ndim == 3:
            border_value = np.median(np.reshape(img, (img.shape[0]*img.shape[1], -1)), axis=0)
        else:
            border_value = np.median(img)
        warped_img = cv2.warpAffine(img, transform, dsize=(x,y), borderValue=float(border_value))
        return warped_img
