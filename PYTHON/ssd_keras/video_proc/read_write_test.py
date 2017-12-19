# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:51:20 2017

@author: fodrasz
"""

import numpy as np
import cv2
print(cv2.__version__)

import time

img_height = 300 # Height of the input images
img_width = 300 # Width of the input images
new_height=img_height

"""
read
"""

video_stream=r'C:\Users\fodrasz\OneDrive\Annotation\Videos\VB_short.mp4'

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
out = cv2.VideoWriter('output.avi',fourcc, fps, (img_height,img_width))



while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        scale = new_height / frame.shape[0]
        dim = (int(frame.shape[1] * scale),new_height)
         
        # perform the actual resizing of the image and show it
        im_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        im_square=np.zeros((1,img_height,img_width,3),'uint8')
        
        image_aspect_ratio = dim[0] / dim[1]
        
        
        if image_aspect_ratio < 1:
            im_square[0,:,int(img_width/2-dim[0]/2):int(img_width/2+dim[0]/2),:]=im_resized
        else:
            im_square[0,:,:,:]=im_resized[:,int(dim[0]/2-img_width/2):int(dim[0]/2+img_width/2),:]
           
        
#        cv2.imshow('frame',im_square[0,:,:,:])
#        cv2.waitKey(0)
        """
        PREDICT
        """
        
        t=time.time()
        y_pred = model.predict(im_square)
        # 4: Decode the raw prediction `y_pred`
        y_pred_decoded = decode_y2(y_pred,
                                   confidence_thresh=0.50,
                                   iou_threshold=0.01,
                                   top_k='all',
                                   input_coords='centroids',
                                   normalize_coords=False,
                                   img_height=img_height,
                                   img_width=img_width)
        
        print(time.time()-t)
        
        frame_out=im_square[0,:,:,:].copy()
        
        for box in y_pred_decoded[0]:
            label = '{}: {:.2f}'.format(merged_classes[int(box[0])], box[1])
            startX=int(box[2])
            startY=int(box[4])
            endX=int(box[3])
            endY=int(box[5])
            
            frame_out=cv2.rectangle(frame_out, (startX, startY), (endX, endY),(0,0,255), 3)

#            cv2.imshow('frame',frame_out)
#            cv2.waitKey(0)
        # write the flipped frame
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