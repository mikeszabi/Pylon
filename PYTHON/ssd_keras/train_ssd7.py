# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:39:41 2017

@author: fodrasz
"""

#from keras.models import load_model
from math import ceil
#import numpy as np
from matplotlib import pyplot as plt

from src_ssd.keras_ssd_loss import SSDLoss
#from src_ssd.keras_layer_AnchorBoxes import AnchorBoxes
from src_ssd.ssd_box_encode_decode_utils import SSDBoxEncoder
from src_ssd.keras_ssd7 import build_model

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau #TensorBoard
#from keras.models import Model
#from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation
#from keras.regularizers import l2
 


# CALL prepare_data

"""
Parameters
"""
model_name = r'./models/ssd7_pylon'


### To be optimized
img_height = 256 # Height of the input images
img_width = 136 # Width of the input images
size = min(img_width, img_height)

img_channels = 3 # Number of color channels of the input images
n_classes = len(merged_classes) # Number of classes including the background class
gray=False
if img_channels==1:
    gray=True
#min_scale = 0.32 # The scaling factor for the smallest anchor boxes
#max_scale = 0.96 # The scaling factor for the largest anchor boxes

### To be optimized
scales = [0.6, 0.7, 0.8, 0.9, 1] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.

### To be optimized
aspect_ratios = [0.25, 0.3, 0.35, 0.4]
#w= scale*size*sqrt(aspect_ratio) - size==smaller size
#max(img_width/(np.sqrt(aspect_ratios)*size))
#h=scale*size/sqrt(aspect_ratio)
#max(img_height*np.sqrt(aspect_ratios)/size)

two_boxes_for_ar1 = False
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
variances = [1, 1, 1, 1] # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids' # Whether the box coordinates to be used as targets for the model should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = True

batch_size = 16 # Change the batch size if you like, or if you run into memory issues with your GPU.
epochs = 5


"""
CREATE MODEL
"""


K.clear_session() # Clear previous models from memory.
# The output `predictor_sizes` is needed below to set up `SSDBoxEncoder`
model, predictor_sizes = build_model(image_size=(img_height, img_width, img_channels),
                                 n_classes=n_classes,
                                 mode='training',
                                 l2_regularization=0.0005,
                                 min_scale=None, # You could pass a min scale and max scale instead of the `scales` list, but we're not doing that here
                                 max_scale=None,
                                 scales=scales,
                                 aspect_ratios_global=aspect_ratios,
                                 aspect_ratios_per_layer=None,
                                 two_boxes_for_ar1=two_boxes_for_ar1,
                                 steps=steps,
                                 offsets=offsets,
                                 limit_boxes=limit_boxes,
                                 variances=variances,
                                 coords=coords,
                                 normalize_coords=normalize_coords,
                                 return_predictor_sizes=True)

#model.load_weights('./ssd7_weights.h5')

"""
Compile model
"""

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = SSDLoss(neg_pos_ratio=1, n_neg_min=0, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)



"""
Create BATCH 
"""

ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=None,
                                max_scale=None,
                                scales=scales,
                                aspect_ratios_global=aspect_ratios,
                                aspect_ratios_per_layer=None,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.75, ### To be optimized
                                neg_iou_threshold=0.25, ### To be optimized
                                coords=coords,
                                normalize_coords=normalize_coords)

# Diagnostics:
ssd_box_encoder.wh_list_diag


# 5: Set the image processing / data augmentation options and create generator handles.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         train=True,
                                         shuffle=False,
                                         ssd_box_encoder=ssd_box_encoder,
                                         equalize=False,
                                         brightness=(0.75, 1.25, 0.25),
                                         flip=0.5,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=(img_height, img_width, 1, 3), # This one is important because the Pascal VOC images vary in size
                                         random_pad_and_resize=False, #(img_height, img_width, 1, 3,0.5), # This one is important because the Pascal VOC images vary in size
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=gray,
                                         limit_boxes=True, # While the anchor boxes are not being clipped, the ground truth boxes should be
                                         include_thresh=0.75)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     train=True,
                                     shuffle=True,
                                     ssd_box_encoder=ssd_box_encoder,
                                     equalize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     max_crop_and_resize=(img_height, img_width, 1, 3), # This one is important because the Pascal VOC images vary in size
                                     random_pad_and_resize=False, #(img_height, img_width, 1, 3,1), # This one is important because the Pascal VOC images vary in size
                                     random_crop=False,
                                     crop=False,
                                     resize=False,
                                     gray=gray,
                                     limit_boxes=True,
                                     include_thresh=0.75)



# Get the number of samples in the training and validations datasets to compute the epoch lengths below.
n_train_samples = train_dataset.get_n_samples()
n_val_samples   = val_dataset.get_n_samples()

# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch <= 15: return 0.001
    else: return 0.0001
    
  

history = model.fit_generator(generator = train_generator,
                              steps_per_epoch = ceil(n_train_samples/batch_size),
                              epochs = epochs,
                              callbacks = [ModelCheckpoint(r'./checkpoints/ssd7_weights_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True,
                                           mode='auto',
                                           period=1),
                                           LearningRateScheduler(lr_schedule),
                                           EarlyStopping(monitor='val_loss',
                                                         min_delta=0.001,
                                                         patience=7),
                                           ReduceLROnPlateau(monitor='val_loss',
                                                             factor=0.5,
                                                             patience=5,
                                                             epsilon=0.01,
                                                             cooldown=0)],
                              validation_data = val_generator,
                              validation_steps = ceil(n_val_samples/batch_size))

# TODO: Set the filename (without the .h5 file extension!) under which to save the model and weights.
#       Do the same in the `ModelCheckpoint` callback above.
model.save('{}.h5'.format(model_name))
model.save_weights('{}_weights.h5'.format(model_name))

#model_json=model.to_json()
#with open('{}.json'.format(model_name), "w") as json_file:
#    json_file.write(model_json)
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#from keras.models import model_from_json
#
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")

print()
print("Model saved under {}.h5".format(model_name))
print("Weights also saved separately under {}_weights.h5".format(model_name))
print()


plt.figure(figsize=(20,12))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='upper right', prop={'size': 24});