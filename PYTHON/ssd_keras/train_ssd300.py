# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:39:41 2017

@author: fodrasz
"""

from math import ceil
import numpy as np

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, Reshape, Concatenate

from src_ssd.keras_layer_AnchorBoxes import AnchorBoxes
from src_ssd.keras_layer_L2Normalization import L2Normalization
from src_ssd.keras_ssd_loss import SSDLoss
from src_ssd.ssd_box_encode_decode_utils import SSDBoxEncoder


"""
Parameters
"""

model_name = r'./models/ssd300_pylon'


img_height = 272 # Height of the input images
img_width = 272 # Width of the input images
img_channels = 3 # Number of color channels of the input images
gray=False
if img_channels==1:
    gray=True
n_classes = len(merged_classes) # Number of classes including the background class
min_scale = 0.32 # The scaling factor for the smallest anchor boxes
max_scale = 0.96 # The scaling factor for the largest anchor boxes
scales = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [[0.2, 0.3, 0.4],
                 [0.2, 0.25, 0.3, 0.35,0.4],
                 [0.2, 0.25, 0.3, 0.35,0.4],
                 [0.2, 0.25, 0.3, 0.35,0.4],
                 [0.2, 0.3, 0.4],
                 [0.2, 0.3, 0.4]]# The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = False
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.5, 0.5, 0.5, 0.5] # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids' # Whether the box coordinates to be used as targets for the model should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = True
# These are the spatial dimensions (height, width) of the predictor layers. The `SSDBoxEncoder` constructor needs this information.


# 1: Set the batch size.

batch_size = 8 # Change the batch size if you like, or if you run into memory issues with your GPU.
epochs = 50

def ssd_300(image_size,
            n_classes,
            min_scale=0.1,
            max_scale=0.9,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[0.5, 1.0, 2.0],
                                     [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                                     [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                                     [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                                     [0.5, 1.0, 2.0],
                                     [0.5, 1.0, 2.0]],
            two_boxes_for_ar1=True,
            limit_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=False):
    '''
    Build a Keras model with SSD_300 architecture, see references.

    The base network is a reduced atrous VGG-16, extended by the SSD architecture,
    as described in the paper.

    In case you're wondering why this function has so many arguments: All arguments except
    the first two (`image_size` and `n_classes`) are only needed so that the anchor box
    layers can produce the correct anchor boxes. In case you're training the network, the
    parameters passed here must be the same as the ones used to set up `SSDBoxEncoder`.
    In case you're loading trained weights, the parameters passed here must be the same
    as the ones used to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of categories for classification including
            the background class (i.e. the number of positive classes +1 for
            the background calss).
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. Defaults to 0.1.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated. Note that the second to last of the linearly interpolated
            scaling factors will actually be the scaling factor for the last predictor layer, while the last
            scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
            if `two_boxes_for_ar1` is `True`. Defaults to 0.9.
        scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used.
            Defaults to `None`. If a list is passed, this argument overrides `min_scale` and
            `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all prediction layers. Defaults to None.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
            This allows you to set the aspect ratios for each predictor layer individually, which is the case for the
            original SSD300 implementation. If a list is passed, it overrides `aspect_ratios_global`.
            Defaults to the aspect ratios used in the original SSD300 architecture, i.e.:
                [[0.5, 1.0, 2.0],
                 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                 [0.5, 1.0, 2.0],
                 [0.5, 1.0, 2.0]]
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor. Defaults to `True`, following the original
            implementation.
        limit_boxes (bool, optional): If `True`, limits box coordinates to stay within image boundaries.
            This would normally be set to `True`, but here it defaults to `False`, following the original
            implementation.
        variances (list, optional): A list of 4 floats >0 with scaling factors (actually it's not factors but divisors
            to be precise) for the encoded predicted box coordinates. A variance value of 1.0 would apply
            no scaling at all to the predictions, while values in (0,1) upscale the encoded predictions and values greater
            than 1.0 downscale the encoded predictions. Defaults to `[0.1, 0.1, 0.2, 0.2]`, following the original implementation.
            The coordinate format must be 'centroids'.
        coords (str, optional): The box coordinate format to be used. Can be either 'centroids' for the format
            `(cx, cy, w, h)` (box center coordinates, width, and height) or 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids', following the original implementation.
        normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of absolute coordinates,
            i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates. Defaults to `False`.

    Returns:
        model: The Keras SSD model.
        predictor_sizes: A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300

    # Get a few exceptions out of the way first
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios_conv4_3 = aspect_ratios_per_layer[0]
        aspect_ratios_fc7 = aspect_ratios_per_layer[1]
        aspect_ratios_conv6_2 = aspect_ratios_per_layer[2]
        aspect_ratios_conv7_2 = aspect_ratios_per_layer[3]
        aspect_ratios_conv8_2 = aspect_ratios_per_layer[4]
        aspect_ratios_conv9_2 = aspect_ratios_per_layer[5]
    else:
        aspect_ratios_conv4_3 = aspect_ratios_global
        aspect_ratios_fc7 = aspect_ratios_global
        aspect_ratios_conv6_2 = aspect_ratios_global
        aspect_ratios_conv7_2 = aspect_ratios_global
        aspect_ratios_conv8_2 = aspect_ratios_global
        aspect_ratios_conv9_2 = aspect_ratios_global

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for aspect_ratios in aspect_ratios_per_layer:
            if (1 in aspect_ratios) & two_boxes_for_ar1:
                n_boxes.append(len(aspect_ratios) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(aspect_ratios))
        n_boxes_conv4_3 = n_boxes[0] # 4 boxes per cell for the original implementation
        n_boxes_fc7 = n_boxes[1] # 6 boxes per cell for the original implementation
        n_boxes_conv6_2 = n_boxes[2] # 6 boxes per cell for the original implementation
        n_boxes_conv7_2 = n_boxes[3] # 6 boxes per cell for the original implementation
        n_boxes_conv8_2 = n_boxes[4] # 4 boxes per cell for the original implementation
        n_boxes_conv9_2 = n_boxes[5] # 4 boxes per cell for the original implementation
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes_conv4_3 = n_boxes
        n_boxes_fc7 = n_boxes
        n_boxes_conv6_2 = n_boxes
        n_boxes_conv7_2 = n_boxes
        n_boxes_conv8_2 = n_boxes
        n_boxes_conv9_2 = n_boxes

    # Input image format
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ### Design the actual network

    x = Input(shape=(img_height, img_width, img_channels))
    normed = Lambda(lambda z: z/127.5 - 1.0, # Convert input feature range to [-1,1]
                    output_shape=(img_height, img_width, img_channels),
                    name='lambda1')(x)

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(normed)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    # Two layers below uses weights from pretrained networks

    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(pool5)

    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', name='fc7')(fc6)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', name='conv9_2')(conv9_1)

    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

    ### Build the convolutional predictor layers on top of the base network

    # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    conv4_3_norm_mbox_conf = Conv2D(n_boxes_conv4_3 * n_classes, (3, 3), padding='same', name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes_fc7 * n_classes, (3, 3), padding='same', name='fc7_mbox_conf')(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes_conv6_2 * n_classes, (3, 3), padding='same', name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes_conv7_2 * n_classes, (3, 3), padding='same', name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes_conv8_2 * n_classes, (3, 3), padding='same', name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes_conv9_2 * n_classes, (3, 3), padding='same', name='conv9_2_mbox_conf')(conv9_2)
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(n_boxes_conv4_3 * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes_fc7 * 4, (3, 3), padding='same', name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes_conv6_2 * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes_conv7_2 * 4, (3, 3), padding='same', name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes_conv8_2 * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes_conv9_2 * 4, (3, 3), padding='same', name='conv9_2_mbox_loc')(conv9_2)

    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios_conv4_3,
                                             two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios_fc7,
                                    two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='fc7_mbox_priorbox')(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios_conv6_2,
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios_conv7_2,
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios_conv8_2,
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios_conv9_2,
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    model = Model(inputs=x, outputs=predictions)

    # Get the spatial dimensions (height, width) of the predictor conv layers, we need them to
    # be able to generate the default boxes for the matching process outside of the model during training.
    # Note that the original implementation performs anchor box matching inside the loss function. We don't do that.
    # Instead, we'll do it in the batch generator function.
    # The spatial dimensions are the same for the confidence and localization predictors, so we just take those of the conf layers.
    predictor_sizes = np.array([conv4_3_norm_mbox_conf._keras_shape[1:3],
                                 fc7_mbox_conf._keras_shape[1:3],
                                 conv6_2_mbox_conf._keras_shape[1:3],
                                 conv7_2_mbox_conf._keras_shape[1:3],
                                 conv8_2_mbox_conf._keras_shape[1:3],
                                 conv9_2_mbox_conf._keras_shape[1:3]])

    return model, predictor_sizes


"""
CREATE MODEL
"""

K.clear_session() # Clear previous models from memory.
# The output `predictor_sizes` is needed below to set up `SSDBoxEncoder`
model, predictor_sizes = ssd_300(image_size=(img_height, img_width, img_channels),
                                 n_classes=n_classes,
                                 min_scale=None, # You could pass a min scale and max scale instead of the `scales` list, but we're not doing that here
                                 max_scale=None,
                                 scales=scales,
                                 aspect_ratios_global=None,
                                 aspect_ratios_per_layer=aspect_ratios,
                                 two_boxes_for_ar1=two_boxes_for_ar1,
                                 limit_boxes=limit_boxes,
                                 variances=variances,
                                 coords=coords,
                                 normalize_coords=normalize_coords)
vgg16_path = './models/vgg-16_ssd-fcn_ILSVRC-CLS-LOC.h5'

model.load_weights(vgg16_path, by_name=True)

"""
Compile model
"""

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = SSDLoss(neg_pos_ratio=1, n_neg_min=2*batch_size, alpha=0.1)

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
                                aspect_ratios_global=None,
                                aspect_ratios_per_layer=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.7,
                                neg_iou_threshold=0.3,
                                coords=coords,
                                normalize_coords=normalize_coords)



# 5: Set the image processing / data augmentation options and create generator handles.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         equalize=False,
                                         brightness=(0.75, 1.25, 0.25),
                                         flip=False,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=(img_height, img_width, 1, 3), # This one is important because the Pascal VOC images vary in size
                                         full_crop_and_resize=(img_height, img_width, 1, 3,0.5), # This one is important because the Pascal VOC images vary in size
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=gray,
                                         limit_boxes=True, # While the anchor boxes are not being clipped, the ground truth boxes should be
                                         include_thresh=0.9,
                                         diagnostics=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     train=True,
                                     ssd_box_encoder=ssd_box_encoder,
                                     equalize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     max_crop_and_resize=(img_height, img_width, 1, 3), # This one is important because the Pascal VOC images vary in size
                                     full_crop_and_resize=(img_height, img_width, 1, 3,0.5), # This one is important because the Pascal VOC images vary in size
                                     random_crop=False,
                                     crop=False,
                                     resize=False,
                                     gray=gray,
                                     limit_boxes=True,
                                     include_thresh=0.9,
                                     diagnostics=False)

# Get the number of samples in the training and validations datasets to compute the epoch lengths below.
n_train_samples = train_dataset.get_n_samples()
n_val_samples   = val_dataset.get_n_samples()

# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch <= 25: return 0.001
    else: return 0.0001
    
    
history = model.fit_generator(generator = train_generator,
                              steps_per_epoch = ceil(n_train_samples/batch_size),
                              epochs = epochs,
                              callbacks = [ModelCheckpoint(r'./checkpoints/ssd300_weights_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True,
                                           mode='auto',
                                           period=1),
                                           LearningRateScheduler(lr_schedule),
                                           EarlyStopping(monitor='val_loss',
                                           min_delta=0.001,
                                           patience=10),
                                           ReduceLROnPlateau(monitor='val_loss',
                                                             factor=0.5,
                                                             patience=10,
                                                             epsilon=0.001,
                                                             cooldown=0)],
                              validation_data = val_generator,
                              validation_steps = ceil(n_val_samples/batch_size))
                                           

# TODO: Set the filename (without the .h5 file extension!) under which to save the model and weights.
#       Do the same in the `ModelCheckpoint` callback above.
model.save('{}.h5'.format(model_name))
model.save_weights('{}_weights.h5'.format(model_name))

print()
print("Model saved under {}.h5".format(model_name))
print("Weights also saved separately under {}_weights.h5".format(model_name))
print()