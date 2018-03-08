# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:51:20 2017

@author: fodrasz
"""

import numpy as np
import cv2
import os
print(cv2.__version__)
import object_roi_detector as oroid


import time
normalize_coords=True
crop_mode='left' # left, right, middle

"""
init
"""
model_file=r'./models/ssd7_pylon.h5'
roid = oroid.ssd_detection(model_file=model_file,normalize_coords=normalize_coords)

imp=oroid.image_prepare(new_height = roid.im_height, new_width = roid.im_width, dx_roi_pct=25, crop_mode=crop_mode)
    
classes = ['background',
           'concretepylon', 'metalpylon', 'woodpylon']

merge_dict = {'concretepylon':'pylon','metalpylon':'pylon','woodpylon':'pylon'}
merged_classes=['background','pylon']

"""
read
"""

#base_dir=r'E:\\'
base_dir=r'C:\\Users\\fodrasz\\'

video_stream=os.path.join(base_dir,'OneDrive\Annotation\Videos\VB_short.mp4')

#cap = cv2.VideoCapture(video_stream)
#cap.get(cv2.CAP_PROP_FRAME_COUNT)
#fps=cap.get(cv2.CAP_PROP_FPS)
#
##while(cap.isOpened()):
##    ret, frame = cap.read()
##
##    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##
##cv2.imshow('frame',gray)
##if cv2.waitKey(1) & 0xFF == ord('q'):
##    break
#
#cap.release()
#cv2.destroyAllWindows()

"""
write
"""
cap = cv2.VideoCapture(video_stream)
fps=cap.get(cv2.CAP_PROP_FPS)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, fps, ( roid.im_height,roid.im_width))



while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        imp.im=frame.copy()
        im_crop=imp.resize_crop_image()

        roi_box=roid.detect_roi(im_crop)
        
      
#        cv2.imshow('frame',im_square[0,:,:,:])
#        cv2.waitKey(0)
        """
        PREDICT
        """
        
        t=time.time()
        roi_box=roid.detect_roi(im_crop,confidence_thresh=0.5, iou_threshold=0)

        
        print(time.time()-t)
        
        frame_out=im_crop.copy()
        if  roid.im_channels==1:
            frame_out=cv2.cvtColor(frame_out,cv2.COLOR_GRAY2RGB)
            
        for box in roi_box[0]:
            print('yes')
            label = '{}: {:.2f}'.format(merged_classes[int(box[0])], box[1])
            startX=int(box[2])
            startY=int(box[4])
            endX=int(box[3])
            endY=int(box[5])
            
            frame_out=cv2.rectangle(frame_out, (startX, startY), (endX, endY),(0,0,255), 3)

#            cv2.imshow('frame',frame_out)
#            cv2.waitKey(0)
        # write the flipped frame
        frame_out=cv2.cvtColor(frame_out,cv2.COLOR_RGB2BGR)
        out.write(frame_out)

        cv2.imshow('frame',frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()