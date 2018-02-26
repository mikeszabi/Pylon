# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:51:06 2018

@author: SzMike
"""

import numpy as np
import time
import cv2
import os
print(cv2.__version__)
from src_helper.file_helper import imagelist_in_depth
from matplotlib import pyplot as plt

import object_roi_detector as oroid

base_dir=r'E:\\'
#base_dir=r'C:\\Users\\fodrasz\\'
base_data_path=os.path.join(base_dir,'OneDrive','Annotation','IDB_Pylon','pylon1152_output')
PYLON_images_path           = os.path.join(base_data_path,'JPEGImages')


#img_channels = 3 # Number of color channels of the input images

model_file=r'./models/ssd7_pylon.h5'


roid = oroid.ssd_detection(model_file=model_file)
imp=oroid.image_prepare(new_height = 272, dx_roi_pct=25, crop_mode='middle')
    


"""
SET PARAMETERS
"""

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'concretepylon', 'metalpylon', 'woodpylon']

merge_dict = {'concretepylon':'pylon','metalpylon':'pylon','woodpylon':'pylon'}
merged_classes=['background','pylon']
#
## fix size: 576x576
#img_height = 272 # Height of the input images
#img_width = 272 # Width of the input images

# SET THE PROPER PATH of images to be processed

"""
LOAD IMAGES
"""


image_list=imagelist_in_depth(PYLON_images_path,level=1)

"""
EVALUATE
"""

i=5
image_file=image_list[i]

im=imp.load_image(image_file)
if im is not None:
    im_square=imp.resize_square_image()