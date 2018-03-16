# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 19:15:03 2018

@author: fodrasz
"""

import numpy as np
import time
import cv2
import os
from PIL import Image

print(cv2.__version__)
from src_helper.file_helper import imagelist_in_depth, check_folder
from matplotlib import pyplot as plt

import object_roi_detector as oroid

"""
SET PARAMETERS
"""
crop_lower_pct=0
crop_mode='middle'
model_file=r'./models/ssd7_pylon.h5'



#base_dir=r'E:\\'
base_dir=r'C:\\Users\\fodrasz\\'

date='20180313'
run_id='695096513'

test_dir=os.path.join(base_dir,'OneDrive\Annotation\eon_test')
bin_dir=os.path.join(base_dir,'OneDrive\Annotation\eon_bin')
detect_dir=os.path.join(base_dir,'OneDrive\Annotation\eon_detect')

run_images_path           = os.path.join(test_dir,date,run_id)

save_bin_image_path = os.path.join(bin_dir,date,run_id)
save_detect_image_path = os.path.join(detect_dir,date,run_id)

check_folder(save_bin_image_path,create=True)
check_folder(save_detect_image_path,create=True)

#img_channels = 3 # Number of color channels of the input images



roid = oroid.ssd_detection(model_file=model_file,normalize_coords=True)
imp=oroid.image_prepare(new_height = 256, new_width=136,dx_roi_pct=25, crop_mode=crop_mode,crop_lower_pct=crop_lower_pct)
    


"""
LOAD IMAGES
"""


image_list=imagelist_in_depth(run_images_path,level=1)

"""
EVALUATE
"""
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 10, ( roid.im_height,roid.im_width))


for i, image_file in enumerate(image_list):

    #i=10
    print(i)
    image_file=image_list[i]
    
    im=imp.load_image_from_file(image_file)
    if im is not None:
        im_crop=imp.resize_crop_image()
    
        buffer_file=os.path.join(save_bin_image_path,os.path.splitext(os.path.basename(image_file))[0]+'.bin')
        detect_file=os.path.join(save_detect_image_path,os.path.splitext(os.path.basename(image_file))[0]+'.jpg')
        
        oroid.write_to_buffer(buffer_file,im_crop)
        
        bbb=oroid.buffer_from_file(buffer_file)
        
        imp.convert_image_from_buffer(bbb)
        imp.resize_crop_image()
        
        
        roi_box=roid.detect_roi(imp.im_crop,confidence_thresh=0.01, iou_threshold=0.45)
        
        frame_out=imp.im_crop.copy()
        
        for box in roi_box[0]:
            print('yes')
        #    label = '{}: {:.2f}'.format(merged_classes[int(box[0])], box[1])
            startX=int(box[2])
            startY=int(box[4])
            endX=int(box[3])
            endY=int(box[5])
                
            frame_out=cv2.rectangle(frame_out, (startX, startY), (endX, endY),(0,255,0), 3)
#        if len(roi_box[0])>0:
#            break
        
        #            cv2.imshow('frame',frame_out)
        #            cv2.waitKey(0)
            # write the flipped frame
        
        im_out = Image.fromarray(frame_out)
        im_out.save(detect_file)
            
    #  
    #cv2.imshow('frame',frame_out)
    #cv2.waitKey(1)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break