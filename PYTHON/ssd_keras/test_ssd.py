# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:54:38 2017

@author: fodrasz
"""
import numpy as np
import time

from matplotlib import pyplot as plt

#%matplotlib inline

### Make predictions

# 1: Set the generator

img_height=272
img_width=272
gray=True


predict_generator = val_dataset.generate(batch_size=1,
                                         train=False,
                                         equalize=False,
                                         brightness=False,
                                         flip=False,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=(img_height, img_width, 1, 3),
                                         full_crop_and_resize=(img_height, img_width, 1, 3, 1),
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=gray,
                                         limit_boxes=gray,
                                         include_thresh=0.8,
                                         diagnostics=False)

# 2: Generate samples

X, y_true, filenames = next(predict_generator)

i =0 # Which batch item to look at

print("Image:", filenames[i])
print()
print("Ground truth boxes:\n")
print(y_true[i])


# 3: Make a prediction
t=time.time()

y_pred = model.predict(X)
# 4: Decode the raw prediction `y_pred`
y_pred_decoded = decode_y2(y_pred,
                           confidence_thresh=0.25,
                           iou_threshold=0.01,
                           top_k='all',
                           input_coords='centroids',
                           normalize_coords=normalize_coords,
                           img_height=img_height,
                           img_width=img_width)
elapsed = time.time() - t
print(elapsed)


np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print(y_pred_decoded[i])

plt.figure(figsize=(20,12))
plt.imshow(np.squeeze(X[i]),cmap='gray')

current_axis = plt.gca()

for box in y_true[i]:
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((box[1], box[3]), box[2]-box[1], box[4]-box[3], color='green', fill=False, linewidth=2))  
    current_axis.text(box[1], box[3], label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

for box in y_pred_decoded[i]:
    label = '{}: {:.2f}'.format(merged_classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))  
    current_axis.text(box[2], box[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

