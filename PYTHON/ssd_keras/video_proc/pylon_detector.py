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
from keras.models import load_model, model_from_json
from src_ssd.keras_ssd_loss import SSDLoss
from src_ssd.keras_layer_AnchorBoxes import AnchorBoxes
from src_ssd.keras_layer_L2Normalization import L2Normalization
from src_ssd.ssd_box_encode_decode_utils import decode_y2
from PIL import Image
from src_helper.file_helper import imagelist_in_depth
from matplotlib import pyplot as plt


img_channels = 3 # Number of color channels of the input images


"""
# LOADING models with custom objects: AnchorBoxes and ssd_loss
"""
ssd_loss = SSDLoss(neg_pos_ratio=1, n_neg_min=0, alpha=1.0)
model=load_model(r'./models/ssd7_pylon.h5',custom_objects={'AnchorBoxes':AnchorBoxes,'compute_loss': ssd_loss.compute_loss})

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

# fix size: 576x576
img_height = 272 # Height of the input images
img_width = 272 # Width of the input images

# SET THE PROPER PATH of images to be processed
base_data_path=r'c:\Users\fodrasz\OneDrive\Annotation\IDB_Pylon\pylon1152_output'
PYLON_images_path           = os.path.join(base_data_path,'JPEGImages')

"""
LOAD IMAGES
"""


image_list=imagelist_in_depth(PYLON_images_path,level=1)

"""
EVALUATE
"""

i=30
image_file=image_list[i]

image = cv2.imread(image_file)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#    cv2.imshow("original", image)
#    cv2.waitKey(0)
scale = img_height / image.shape[0]
dim = (int(image.shape[1] * scale),img_height)
 
# perform the actual resizing of the image and show it
im_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

if img_channels==1:
    im_resized=np.expand_dims(im_resized[:,:,2], axis=2)
    
im_square=np.zeros((1,img_height,img_width,img_channels),'uint8')

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
                           confidence_thresh=0.25,
                           iou_threshold=0.25,
                           top_k='all',
                           input_coords='centroids',
                           normalize_coords=False,
                           img_height=img_height,
                           img_width=img_width)

print(time.time()-t)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print(y_pred_decoded[0])
plt.figure(figsize=(20,12))
plt.imshow(np.squeeze(im_square[0,:,:,:]))

current_axis = plt.gca()

for box in y_pred_decoded[0]:
    label = '{}: {:.2f}'.format(merged_classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))  
    current_axis.text(box[2], box[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

