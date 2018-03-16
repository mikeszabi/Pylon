# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:56:59 2017

@author: fodrasz
"""

import os

"""
import sys
import imp
imp.reload(sys.modules['src_ssd.ssd_batch_generator'])
"""

from src_ssd.ssd_batch_generator import BatchGenerator
from src_helper.file_helper import imagelist_in_depth
import numpy as np
from sklearn.model_selection import train_test_split


train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])
val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])


# 2: Parse the image and label lists for the training and validation datasets. This can take a while.

# TODO: Set the paths to the datasets here.

# The directories that contain the images.
#base_dir=r'E:\\'
base_dir=r'C:\\Users\\fodrasz\\'
base_data_path=os.path.join(base_dir,'OneDrive','Annotation','IDB_Pylon','pylon1152_output')

train_file=os.path.join(base_data_path,'ImageSets','Main','train_list.txt')
test_file=os.path.join(base_data_path,'ImageSets','Main','test_list.txt')

PYLON_images_path           = os.path.join(base_data_path,'JPEGImages')
# The directories that contain the annotations.
PYLON_annotations_path      = os.path.join(base_data_path,'Annotations')

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background','lamp']
merge_dict = {'lamp':'lamp'}

#
#classes = ['background',
#           'concretepylon', 'metalpylon', 'woodpylon']
#merge_dict = {'concretepylon':'pylon','metalpylon':'pylon','woodpylon':'pylon'}

merged_classes=['background']
merged_classes=merged_classes+list(set(merge_dict.values()))

# create train/validation list

image_list=imagelist_in_depth(PYLON_images_path)
image_list=[os.path.splitext(os.path.basename(file))[0] for file in image_list]
train_list ,test_list = train_test_split(image_list,test_size=0.2)

with open(train_file,'wt') as fp:
    for item in train_list:
        fp.write("{}\n".format(item))
with open(test_file,'wt') as fp:
    for item in test_list:
        fp.writelines("{}\n".format(item))

train_dataset.parse_xml(images_paths=[PYLON_images_path],
                        annotations_paths=[PYLON_annotations_path],
                        image_set_paths=[train_file],
                        classes=merged_classes,
                        merge_dict=merge_dict,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False,
                        sep=' ')

val_dataset.parse_xml(images_paths=[PYLON_images_path],
                        annotations_paths=[PYLON_annotations_path],
                        image_set_paths=[test_file],
                        classes=merged_classes,
                        merge_dict=merge_dict,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False,
                        sep=' ')

# Do some tests in datasets
for i in range(len(train_dataset.labels)):
    if train_dataset.labels[i]==[]:
        print(i)