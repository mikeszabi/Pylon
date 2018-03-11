# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:19:00 2017

@author: Szabolcs
"""


import numpy as np
import cv2
from keras.models import load_model
from src_ssd.keras_ssd_loss import SSDLoss
from src_ssd.keras_layer_AnchorBoxes import AnchorBoxes
from src_ssd.keras_layer_L2Normalization import L2Normalization
from src_ssd.ssd_box_encode_decode_utils import decode_y2

#from matplotlib import pyplot as plt
#
#import scipy.misc

def buffer_from_file(buffer_file):
    buffer=None
    with open(buffer_file, "rb") as binary_file:
        # Read the whole file at once
        buffer = binary_file.read()
    return buffer
    
def write_to_buffer(buffer_file,im):
    im.astype('uint8').tofile(buffer_file)

class image_prepare:
    def __init__(self,new_height = 512, new_width=272, dx_roi_pct=20, crop_mode='middle', preserved_bytes=0):
        self.im=None # original image
        self.scale=None # resize scale
        self.im_crop=None # small square image - input of detection model
        self.x0_crop=None # x coordinate of the resized image relative to the cropped image coordinate
        self.im_roi=None # roi cropped from original image
        
        self.new_height=new_height
        self.new_width=new_width
        self.crop_mode=crop_mode
        self.dx_roi_crop=int(dx_roi_pct*self.new_width/100) # fix width of the roi as the pct of the width
        
        # This class stores the original and resized/cropped images and the roi coordinates 
        
        # new_height - the desired output image hight
        # crop_mode (left,right, middle) - define how to crop landscape images
        
        self.preserved_bytes=preserved_bytes
        self.im_info= bytes([0x00]*preserved_bytes)

    def load_image_from_file(self,image_file):
        
        # Parameters
        # image_file - PATH of the input image
        
        
        # Output
        # numpy array
        # square image with the desired height in uint8 format
        
        # loads image from file
        im_0 = cv2.imread(image_file)
        if im_0 is None:
            print('Image file could not be read')
            return None
        if im_0.ndim==2:
            im_0=np.expand_dims(self.im, axis=2)
        elif not im_0.ndim==3:
            print('Wrong image format')
            return None
        else:
            # converting to RGB from BGR
            self.im=cv2.cvtColor(im_0,cv2.COLOR_BGR2RGB)
            
        return self.im
    
    def convert_image_from_buffer(self,buffer):
        self.im=np.frombuffer(buffer, dtype=np.uint8, count=-1, offset=0)
        self.im=self.im.reshape((self.new_height,self.new_width,3))
        self.im_info=buffer[0:self.preserved_bytes]
    
    def resize_crop_image(self):
    
        # resize to new_height x new_width - new_width is fixed
        self.scale = self.new_height / self.im.shape[0]
        dim = (int(self.im.shape[1] * self.scale),self.new_height)
        image_aspect_ratio = dim[0] / dim[1] # width/height
            
        im_resized = cv2.resize(self.im, dim, interpolation = cv2.INTER_AREA)
        
    
        self.im_cropped=np.zeros((self.new_height,self.new_width,self.im.ndim),'uint8')                 
        
        # ToDo: save crop dimensions
            
        if image_aspect_ratio < self.new_width/self.new_height:
            # portrait type
            self.x0_crop=-int(self.new_width/2-dim[0]/2)
            self.im_crop[:,int(self.new_width/2-dim[0]/2):int(self.new_width/2+dim[1]/2),:]=im_resized
        else:
            if self.crop_mode=='middle':
                self.x0_crop=int(dim[0]/2-self.new_width/2)
            elif self.crop_mode=='left':
                self.x0_crop=0
            elif self.crop_mode=='right':
                self.x0_crop=dim[0]-self.new_width
            
            self.im_crop=im_resized[:,self.x0_crop:self.new_width+self.x0_crop,:]

        return self.im_crop

    def calc_roi_coords(self,roi_coords):
        
        # transform ROI coords to the original image 
        # input: ROI coords on the cropped image  (x,y,dx,dy) x=0,y=0 upper left corber of the image
        
        x,y,dx,dy=roi_coords
        
        # transform back to the original image
        
        x_orig=int((x+self.x0_crop)/self.scale)
        y_orig=int(y/self.scale)
        dx_orig=int(dx/self.scale)
        dy_orig=int(dy/self.scale)
        
        return [x_orig,y_orig,dx_orig,dy_orig]
    
    def calc_roi_stripe(self,x_roi_start):
        
        # input: x_roi_start - the start x coordinate of the ROI strip on the cropped image
        
        assert x_roi_start < self.im_crop.shape[1], "Inproper ROI"       
        
        # calculate the roi stripe on the original image

        x_roi_end=min(x_roi_start+self.dx_roi_crop,self.im_crop.shape[1])
        
        # transform back to the original image
        
        x_roi_end=int((x_roi_end+self.x0_crop)/self.scale)
        x_roi_start=x_roi_end-int(self.dx_roi_crop/self.scale)
        
        return x_roi_start
    
    def crop_roi_stripe(self,x_roi_start):
        
        assert x_roi_start < self.im.shape[1], "Inproper ROI"       
        
        # create the roi stripe on the original image

        self.im_roi=self.im[:,x_roi_start:x_roi_start+int(self.dx_roi_crop/self.scale),:]
        
        return self.im_roi
    
    def get_output_buffer(self,roi_coord_orig):
        coord_info=bytes(np.asarray(roi_coord_orig,'uint16'))
        self.im_info

        # concatenate bytes
        con=self.im_info+coord_info
        
        return con

    
class ssd_detection:
    def __init__(self,model_file=None, normalize_coords = False):

        # load detection model
        print('...loading detection model')

        ssd_loss = SSDLoss(neg_pos_ratio=1, n_neg_min=0, alpha=1.0)
        self.model=load_model(model_file,custom_objects={'AnchorBoxes':AnchorBoxes,'compute_loss': ssd_loss.compute_loss,'L2Normalization':L2Normalization})

        # model specific parameters
        self.im_height=self.model.input_shape[1]
        self.im_width=self.model.input_shape[2] 
        self.im_channels=self.model.input_shape[3]   
        
        self.normalize_coords=normalize_coords
 
    
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

        # if gray model
        if  self.im_channels==1:
            im=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            im=np.expand_dims(im, axis=2)
        
        im=np.expand_dims(im, axis=0) # first singleton dimension is needed as model input        
            
        y_pred = self.model.predict(im)
        # Decode the raw prediction `y_pred`
        y_pred_decoded = decode_y2(y_pred,
                                   confidence_thresh=confidence_thresh,
                                   iou_threshold=iou_threshold,
                                   top_k=1, # only one detection per image
                                   input_coords='centroids',
                                   normalize_coords=self.normalize_coords,
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
        
        return y_pred_decoded


if __name__ == "__main__":
    
    
    model_file=r'./models/ssd7_pylon.h5'
    roid = ssd_detection(model_file=model_file)


    image_file=r'e:\OneDrive\Annotation\Picturio\TEST\concrete_1.jpg'

    imp=image_prepare(new_height = 272, dx_roi_pct=25, crop_mode='middle')
    
 
    im=imp.load_image(image_file)
    if im is not None:
        im_square=imp.resize_square_image()
    
#        name, ext=image_file.split('.')
#        scipy.misc.imsave(name+'_small.'+ext, im_square)

    roi_box=roid.detect_roi(im_square)
    
#    np.set_printoptions(precision=2, suppress=True, linewidth=90)
#    print("Predicted boxes:\n")
#    print(roi_box[0])
#    
#    fig, ax = plt.subplots(1,2)
#    #plt.figure(figsize=(20,12))
#    ax[0].imshow(im_square)
#    
#    for box in roi_box[0]:
#        ax[0].add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))  
#    
    if len(roi_box[0])==1:
        
        x_roi_start_square=roi_box[0][0][2]
        
        x_roi_start=imp.calc_roi_stripe(x_roi_start_square)
        
#        im_roi=imp.crop_roi_stripe(x_roi_start)
        
#        ax[1].imshow(im_roi)
