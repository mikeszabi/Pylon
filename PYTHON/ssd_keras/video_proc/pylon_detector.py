# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import time
import cv2
import os
print(cv2.__version__)
from src_ssd.keras_ssd7 import build_model
from keras import backend as K
from keras.models import load_model
from src_ssd.keras_ssd_loss import SSDLoss
from src_ssd.keras_layer_AnchorBoxes import AnchorBoxes
from src_ssd.keras_layer_L2Normalization import L2Normalization
from src_ssd.ssd_box_encode_decode_utils import decode_y2
from PIL import Image
from src_helper.file_helper import imagelist_in_depth
from matplotlib import pyplot as plt

#Doesn't work...
#model=load_model(r'./models/ssd7_pylon.h5')

"""
SET PARAMETERS
"""

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'concretepylon', 'metalpylon', 'woodpylon']

merge_dict = {'concretepylon':'pylon','metalpylon':'pylon','woodpylon':'pylon'}
merged_classes=['background','pylon']

"""
"""

### To be optimized
img_height = 576 # Height of the input images
img_width = 576 # Width of the input images
img_channels = 3 # Number of color channels of the input images
n_classes = len(merged_classes) # Number of classes including the background class
#min_scale = 0.32 # The scaling factor for the smallest anchor boxes
#max_scale = 0.96 # The scaling factor for the largest anchor boxes

### To be optimized
scales = [0.3, 0.5, 0.7, 0.8, 0.9] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.

### To be optimized
aspect_ratios = [0.15, 0.25, 0.35, 0.45]
two_boxes_for_ar1 = False
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries

### ???
variances = [0.5, 0.5, 0.5, 0.5] # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids' # Whether the box coordinates to be used as targets for the model should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = False
# These are the spatial dimensions (height, width) of the predictor layers. The `SSDBoxEncoder` constructor needs this information.
#predictor_sizes = [[58, 35], 
#                   [28, 16],
#                   [13,  7],
#                   [ 5,  2]]

# 4: Set the batch size.

"""
CREATE MODEL
"""


K.clear_session() # Clear previous models from memory.
# The output `predictor_sizes` is needed below to set up `SSDBoxEncoder`
model, predictor_sizes = build_model(image_size=(img_height, img_width, img_channels),
                                 n_classes=n_classes,
                                 min_scale=None, # You could pass a min scale and max scale instead of the `scales` list, but we're not doing that here
                                 max_scale=None,
                                 scales=scales,
                                 aspect_ratios_global=aspect_ratios,
                                 aspect_ratios_per_layer=None,
                                 two_boxes_for_ar1=two_boxes_for_ar1,
                                 limit_boxes=limit_boxes,
                                 variances=variances,
                                 coords=coords,
                                 normalize_coords=normalize_coords)


model.load_weights(r'./models/ssd7_pylon_weights.h5')

"""
LOAD AND PROCESS IMAGES
"""
# fix size: 576x576
new_height=img_height

# SET THE PROPER PATH
base_data_path=r'c:\Users\fodrasz\OneDrive\Annotation\IDB_Pylon\pylon1152_output'
PYLON_images_path           = os.path.join(base_data_path,'JPEGImages')

image_list=imagelist_in_depth(PYLON_images_path,level=1)

"""
EVALUATE
"""

i=1
image_file=image_list[i]

image = cv2.imread(image_file)
#    cv2.imshow("original", image)
#    cv2.waitKey(0)
scale = new_height / image.shape[0]
dim = (int(image.shape[1] * scale),new_height)
 
# perform the actual resizing of the image and show it
im_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

im_square=np.zeros((1,img_height,img_width,3),'uint8')

image_aspect_ratio = dim[0] / dim[1]


if image_aspect_ratio < 1:
    im_square[0,:,int(img_width/2-dim[0]/2):int(img_width/2+dim[0]/2),:]=im_resized
else:
    im_square[0,:,:,:]=im_resized[:,int(dim[0]/2-img_width/2):int(dim[0]/2+img_width/2),:]
   

#plt.imshow(im_square[0,:,:,:])
"""
PREDICT
"""

t=time.time()
y_pred = model.predict(im_square)
# 4: Decode the raw prediction `y_pred`
y_pred_decoded = decode_y2(y_pred,
                           confidence_thresh=0.5,
                           iou_threshold=0.01,
                           top_k=1,
                           input_coords='centroids',
                           normalize_coords=False,
                           img_height=1152,
                           img_width=1152)

print(time.time()-t)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print(y_pred_decoded[0])
plt.figure(figsize=(20,12))
plt.imshow(im_square[0,:,:,:])

current_axis = plt.gca()

for box in y_pred_decoded[0]:
    label = '{}: {:.2f}'.format(merged_classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))  
    current_axis.text(box[2], box[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

