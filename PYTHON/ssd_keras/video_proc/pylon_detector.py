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

base_dir=r'E:\\'
#base_dir=r'C:\\Users\\fodrasz\\'
#base_data_path=os.path.join(base_dir,'OneDrive','Annotation','IDB_Pylon','pylon1152_output')
base_data_path=os.path.join(base_dir,'OneDrive','Annotation','Videos','Frames')

#PYLON_images_path           = os.path.join(base_data_path,'JPEGImages')
PYLON_images_path           = os.path.join(base_data_path,'Pylon_4')


normalize_coords=True
crop_mode='left' # left, right, middle


"""
init
"""
model_file=r'./models/ssd7_pylon.h5'
roid = oroid.ssd_detection(model_file=model_file,normalize_coords=normalize_coords)

imp=oroid.image_prepare(new_height = roid.im_height, new_width = roid.im_width, crop_mode=crop_mode,preserved_bytes=12)
    
classes = ['background',
           'concretepylon', 'metalpylon', 'woodpylon']

merge_dict = {'concretepylon':'pylon','metalpylon':'pylon','woodpylon':'pylon'}
merged_classes=['background','pylon']    
   

"""
LOAD IMAGES
"""


image_list=imagelist_in_depth(PYLON_images_path,level=1)

"""
EVALUATE
"""

i=3
image_file=image_list[i]

im=imp.load_image_from_file(image_file)
if im is not None:
    im_crop=imp.resize_crop_image()

buffer_file=os.path.splitext(os.path.basename(image_file))[0]+'.bin'

oroid.write_to_buffer(buffer_file,im_crop)

bbb=oroid.buffer_from_file(buffer_file)

imp.convert_image_from_buffer(bbb)
imp.resize_crop_image()


if  roid.im_channels==1:
    im_in=cv2.cvtColor(imp.im_crop,cv2.COLOR_RGB2GRAY)
else:
    im_in=imp.im_crop

roi_box=roid.detect_roi(imp.im_crop,confidence_thresh=0.01, iou_threshold=0.45)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print(roi_box[0])

if len(roi_box[0]):
    box_coord=imp.calc_roi_coords(roi_box[0][0][2:6])
    box_coord=[max(coord,0) for coord in box_coord]
    
    # here is the output of the evaluation
    output=imp.get_output_buffer(box_coord)

fig, ax = plt.subplots(1,2)
#plt.figure(figsize=(20,12))
ax[0].imshow(imp.im_crop)

for box in roi_box[0]:
    label = '{}: {:.2f}'.format(merged_classes[int(box[0])], box[1])
    ax[0].add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))  
    ax[0].text(box[2], box[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

if len(roi_box[0])==1:
    
    x_roi_start_crop=roi_box[0][0][2]
    
    x_roi_start=imp.calc_roi_stripe(x_roi_start_crop)
    
    im_roi=imp.crop_roi_stripe(x_roi_start)
    
    ax[1].imshow(im_roi)
