# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:19:00 2017

@author: Szabolcs
"""

import warnings

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



def create_image(image_file,new_height = 272, crop_mode='middle'):
    
    
    # Parameters
    # image_file - PATH of the input image
    # new_height - the desired output image hight
    # crop_mode (left,right, middle) - define how to crop landscape images
    
    # Output
    # square image with the desired height in uint8 format
    
    # loads image from file
    image = cv2.imread(image_file)
    if image is None:
        print('Image file could not be read')
        return None
    if image.ndim==1:
        image=np.expand_dims(image, axis=2)
    elif not image.ndim==3:
        print('Wrong image format')
        return None

    # converting to RGB
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    # resize to new_height x new_height
    scale = new_height / image.shape[0]
    dim = (int(image.shape[1] * scale),new_height)
         
    im_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        
    
    im_square=np.zeros((new_height,new_height,image.ndim),'uint8')
        
    image_aspect_ratio = dim[0] / dim[1]
        
    
    # ToDo: save crop dimensions
        
    if image_aspect_ratio < 1:
        im_square[:,int(new_height/2-dim[0]/2):int(new_height/2+dim[0]/2),:]=im_resized
    else:
        if crop_mode=='middle':
            im_square=im_resized[:,int(dim[0]/2-new_height/2):int(dim[0]/2+new_height/2),:]
        elif crop_mode=='left':
            im_square=im_resized[:,0:new_height,:]
        elif crop_mode=='right':
            im_square=im_resized[:,dim[0]-new_height:dim[0],:]
            
    return im_square

#def crop_roi(im,new_height = 272, crop_mode='middle'):
#def roi_on_original
    
class ssd_detection:
    def __init__(self,model_file=None):

        # load detection model
        print('...loading detection model')

        ssd_loss = SSDLoss(neg_pos_ratio=1, n_neg_min=0, alpha=1.0)
        self.model=load_model(model_file,custom_objects={'AnchorBoxes':AnchorBoxes,'compute_loss': ssd_loss.compute_loss})

        # model specific parameters
        self.im_height=self.model.input_shape[1]
        self.im_width=self.model.input_shape[2] 
        self.im_channels=self.model.input_shape[3]         
 
    
    def detect_roi(self, im, confidence_thresh=0.01, iou_threshold=0.25):

        # detecting pylon ROI on images
        # input: numpy image
        # output: ROI coordinates
        
        # check if im numpy
        # check size: 
        # check dimension
        
        assert type(im)==np.ndarray, "Image is Not numpy array"
        assert im.shape[0]==self.im_height, "Image has wrong height"
        assert im.shape[1]==self.im_width, "Image has wrong width"
        assert im.ndim==3, "Image has wrong number of channels"

        im=np.expand_dims(im, axis=0) # first singleton dimension is needed as model input        
            
        y_pred = self.model.predict(im)
        # Decode the raw prediction `y_pred`
        y_pred_decoded = decode_y2(y_pred,
                                   confidence_thresh=confidence_thresh,
                                   iou_threshold=iou_threshold,
                                   top_k=1, # only one detection per image
                                   input_coords='centroids',
                                   normalize_coords=False,
                                   img_height=self.im_height,
                                   img_width=self.im_width)

#Arguments:
#        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
#            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
#            boxes predicted by the model per image and the last axis contains
#            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
#        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
#            positive class in order to be considered for the non-maximum suppression stage for the respective class.
#            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
#            stage, while a larger value will result in a larger part of the selection process happening in the confidence
#            thresholding stage. Defaults to 0.01, following the paper.
#        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
#            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
#            to the box score. Defaults to 0.45 following the paper.
#        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
#            non-maximum suppression stage. Defaults to 200, following the paper.
#        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
#            for the format `(cx, cy, w, h)` (box center coordinates, width, and height) or 'minmax'
#            for the format `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.
#        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
#            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
#            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
#            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
#            coordinates. Requires `img_height` and `img_width` if set to `True`. Defaults to `False`.
#        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
#        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.

        
        return y_pred_decoded[0][0][2]


if __name__ == "__main__":
    
    
    model_file=r'./models/ssd7_pylon.h5'
    image_file=r'e:\OneDrive\Annotation\Picturio\TEST\metal_1.jpg'

#   creating gui
    roid = ssd_detection(model_file=model_file)

    im_square=create_image(image_file,new_height = 272, crop_mode='middle')
    
    roi=roid.detect_roi(im_square)