# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import time
import cv2
import os
print(cv2.__version__)
from src_helper.file_helper import imagelist_in_depth
from matplotlib import pyplot as plt

import object_roi_detector as oroid

#base_dir=r'E:\\'
base_dir=r'C:\\Users\\fodrasz\\'
base_data_path=os.path.join(base_dir,'OneDrive','Annotation','IDB_Pylon','pylon1152_output')
PYLON_images_path           = os.path.join(base_data_path,'JPEGImages')


#img_channels = 3 # Number of color channels of the input images

model_file=r'./models/ssd300_pylon.h5'


roid = oroid.ssd_detection(model_file=model_file)
imp=oroid.image_prepare(new_height = 272, dx_roi_pct=25, crop_mode='middle')
    
   

"""
# LOADING models with custom objects: AnchorBoxes and ssd_loss
"""
#ssd_loss = SSDLoss(neg_pos_ratio=1, n_neg_min=0, alpha=1.0)
#model=load_model(r'./models/ssd7_pylon.h5',custom_objects={'AnchorBoxes':AnchorBoxes,'compute_loss': ssd_loss.compute_loss})

# If model is saved in json format
#model = model_from_json(loaded_model_json)
#model.load_weights(r'./models/ssd7_pylon_weights_benchmark.h5')


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

roi_box=roid.detect_roi(im_square)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print(roi_box[0])

fig, ax = plt.subplots(1,2)
#plt.figure(figsize=(20,12))
ax[0].imshow(im_square)

for box in roi_box[0]:
    label = '{}: {:.2f}'.format(merged_classes[int(box[0])], box[1])
    ax[0].add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))  
    ax[0].text(box[2], box[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

if len(roi_box[0])==1:
    
    x_roi_start_square=roi_box[0][0][2]
    
    x_roi_start=imp.calc_roi_stripe(x_roi_start_square)
    
    im_roi=imp.crop_roi_stripe(x_roi_start)
    
    ax[1].imshow(im_roi)
