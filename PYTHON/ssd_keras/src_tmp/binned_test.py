# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:51:12 2018

@author: SzMike
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# read binned
im_file=r'D:\DATA\EON\OO\binned1.bin'

with open(im_file, "rb") as binary_file:
    # Read the whole file at once
    data = binary_file.read()
    print(data)
    
im=np.frombuffer(data, dtype=np.uint8, count=-1, offset=0)
im=im.reshape((256,136,3))

# write binned

