# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:56:59 2017

@author: fodrasz
"""

import os

import sys
import imp

#imp.reload(sys.modules['ssd_batch_generator'])
from src_ssd.ssd_batch_generator import BatchGenerator


train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])
val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])


# 2: Parse the image and label lists for the training and validation datasets. This can take a while.

# TODO: Set the paths to the datasets here.

# The directories that contain the images.
base_dir=r'E:\\'
#base_dir=r'C:\\Users\\fodrasz\\'
base_data_path=os.path.join(base_dir,'OneDrive','Annotation','IDB_Pylon','pylon1152_output')

PYLON_images_path           = os.path.join(base_data_path,'JPEGImages')
# The directories that contain the annotations.
PYLON_annotations_path      = os.path.join(base_data_path,'Annotations')




# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'concretepylon', 'metalpylon', 'woodpylon']

merge_dict = {'concretepylon':'pylon','metalpylon':'pylon','woodpylon':'pylon'}
merged_classes=['background','pylon']

train_dataset.parse_xml(images_paths=[PYLON_images_path],
                        annotations_paths=[PYLON_annotations_path],
                        image_set_paths=[os.path.join(base_data_path,'ImageSets','Main','concretepylon_train.txt'),
                                         os.path.join(base_data_path,'ImageSets','Main','metalpylon_train.txt'),
                                         os.path.join(base_data_path,'ImageSets','Main','woodpylon_train.txt')],
                        classes=merged_classes,
                        merge_dict=merge_dict,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False,
                        sep=' ')

val_dataset.parse_xml(images_paths=[PYLON_images_path],
                        annotations_paths=[PYLON_annotations_path],
                        image_set_paths=[os.path.join(base_data_path,'ImageSets','Main','concretepylon_val.txt'),
                                         os.path.join(base_data_path,'ImageSets','Main','metalpylon_val.txt'),
                                         os.path.join(base_data_path,'ImageSets','Main','woodpylon_val.txt')],
                        classes=merged_classes,
                        merge_dict=merge_dict,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False,
                        sep=' ')