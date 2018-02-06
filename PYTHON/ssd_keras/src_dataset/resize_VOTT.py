# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
#import json
from src_helper import file_helper
import cv2


new_height=1152

ann_name='E1'
new_ann_name='pylon1152'

base_dir=r'E:'
base_dir=r'C:\Users\fodrasz'

ann_image_path=os.path.join(base_dir,'OneDrive','Annotation','IDB_Pylon','pylon_orig',ann_name)

new_ann_image_path=os.path.join(base_dir,'OneDrive','Annotation','IDB_Pylon',new_ann_name)

#ann= json.load(open(ann_json))
#
#new_ann=ann.copy()

image_list_indir=file_helper.imagelist_in_depth(ann_image_path,level=1)

for i,image_file in enumerate(image_list_indir):
    
    print(i)

    image_file=image_list_indir[i] 
#    exif_data = file_helper.get_exif_data(image_file)
#    
#    if exif_data:
#        exif_data['Image Orientation']
        
    image = cv2.imread(image_file)
#    cv2.imshow("original", image)
#    cv2.waitKey(0)
    scale = new_height / image.shape[0]
    dim = (int(image.shape[1] * scale),new_height)
 
    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    resized_image_file=os.path.join(new_ann_image_path,ann_name+'_'+os.path.basename(image_file))
    cv2.imwrite(resized_image_file,resized)
#    cv2.imshow("resized", resized)
#    cv2.waitKey(0)
    
#    for j in range(len(ann['frames'][str(i)])):
#        new_ann['frames'][str(i)][j]['x1']=int(new_ann['frames'][str(i)][j]['x1']*scale)
    
     


    
