# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 14:44:14 2018

@author: fodrasz
"""

from src_ssd.ssd_batch_generator import BatchGenerator


train_generator = train_dataset.generate(batch_size=1,
                                         train=True,
                                         equalize=False,
                                         ssd_box_encoder=ssd_box_encoder,
                                         brightness=(0.75, 1.25, 0.25),
                                         flip=True,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=(265, 136, 1, 3), # This one is important because the Pascal VOC images vary in size
                                         full_crop_and_resize=(256, 136, 1, 3,1), # This one is important because the Pascal VOC images vary in size
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=True,
                                         limit_boxes=True, # While the anchor boxes are not being clipped, the ground truth boxes should be
                                         include_thresh=0.9,
                                         diagnostics=True)

for i in range(200):
    out=next(train_generator)

