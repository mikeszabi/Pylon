# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:51:20 2017

@author: fodrasz
"""

import numpy as np
import cv2
import os
print(cv2.__version__)
import time

"""
import sys
import imp
imp.reload(sys.modules['object_roi_detector'])
"""

import object_roi_detector as oroid


#base_dir=r'E:\\'
base_dir=r'C:\\Users\\fodrasz\\'
normalize_coords=True
crop_mode='left' # left, right, middle
crop_lower_pct=0.5

# ToDo: check dir
videos_available=['VB_short.mp4','VB_long.m4v','2018_0311_142010_004.MOV']
vid_sel=0
"""
init
"""
model_file=r'./models/ssd7_pylon.h5'
roid = oroid.ssd_detection(model_file=model_file,normalize_coords=normalize_coords)

imp=oroid.image_prepare(new_height = int(roid.im_height/(1-crop_lower_pct)), new_width = roid.im_width, dx_roi_pct=25, crop_mode=crop_mode,crop_lower_pct=crop_lower_pct)


"""
read
"""

video_dir=os.path.join(base_dir,'OneDrive','Annotation','Videos')
video_stream=os.path.join(video_dir,videos_available[vid_sel])

"""
write
"""
cap = cv2.VideoCapture(video_stream)
fps=cap.get(cv2.CAP_PROP_FPS)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
#fps=max(fps,20)
out = cv2.VideoWriter('output.avi',fourcc, fps, ( roid.im_width, int(roid.im_height/(1-crop_lower_pct))),True)



while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        imp.im=frame.copy()
        imp.resize_crop_image()
        
        """
        PREDICT
        """

        t=time.time()

        roi_box=roid.detect_roi(imp.im_crop)

        
        print(time.time()-t)
        
        frame_out=imp.im_pad.copy()
#        if  roid.im_channels==1:
#            frame_out=cv2.cvtColor(frame_out,cv2.COLOR_GRAY2RGB)
#            
        for box in roi_box[0]:
            roi_coords=imp.calc_roi_coords(box[2:])
            print('yes')
            label = '{}: {:.2f}'.format('Detection', box[1])
            startX=int(roi_coords[0])
            startY=int(roi_coords[2])
            endX=int(roi_coords[1])
            endY=int(roi_coords[3])
            
            frame_out=cv2.rectangle(frame_out, (startX, startY), (endX, endY),(0,0,255), 3)

#            cv2.imshow('frame',frame_out)
#            cv2.waitKey(0)
        # write the flipped frame
        frame_out=cv2.cvtColor(frame_out,cv2.COLOR_RGB2BGR)

        out.write(frame_out)

        cv2.imshow('frame',frame_out)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()