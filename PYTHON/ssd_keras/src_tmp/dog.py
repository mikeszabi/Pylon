# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:47:28 2017

@author: fodrasz
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:12:55 2017

@author: SzMike
"""
import skimage.io as io

io.use_plugin('pil') # Use only the capability of PIL
#%matplotlib qt5
from matplotlib import pyplot as plt
import cv2
import scipy.ndimage
import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image                  


prototxt='model\MobileNetSSD_deploy.prototxt.txt'
model='model\MobileNetSSD_deploy.caffemodel'
image_file='images\example_01.jpg'
min_prob=0.2 # minimum probability to filter weak detections

print(cv2.__version__)

ResNet50_model = ResNet50(weights='imagenet')    

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)


# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#ap.add_argument("-p", "--prototxt", required=True,
#	help="path to Caffe 'deploy' prototxt file")
#ap.add_argument("-m", "--model", required=True,
#	help="path to Caffe pre-trained model")
#ap.add_argument("-c", "--confidence", type=float, default=0.2,
#	help="minimum probability to filter weak detections")
#args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))



#
#img = image.load_img(r'd:\Projects\DOG\dogImages\test\004.Akita\Akita_00258.jpg', target_size=(224, 224))
## convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#
#img = preprocess_input(x)

cap = cv2.VideoCapture(r'd:\Projects\DOG\dogVideos\\GreatDane_short.mp4')

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('dogVideos\\GreatDane_processed.avi',fourcc, 24.0, (int(cap.get(3)),int(cap.get(4))))

while(True):

    ret, frame = cap.read()
    if frame is None:
        break
     
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with the
    	# prediction
        	confidence = detections[0, 0, i, 2]
        
        	# filter out weak detections by ensuring the `confidence` is
        	# greater than the minimum confidence
        	if confidence > min_prob:
        		# extract the index of the class label from the `detections`,
        		# then compute the (x, y)-coordinates of the bounding box for
        		# the object
        		idx = int(detections[0, 0, i, 1])
        		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        		(startX, startY, endX, endY) = box.astype("int")
        
        		# display the prediction
        		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        		print("[INFO] {}".format(label))
        		cv2.rectangle(frame, (startX, startY), (endX, endY),
        			COLORS[idx], 2)
        		y = startY - 15 if startY - 15 > 15 else startY + 15
#        		cv2.putText(frame, label, (startX, y),
#        			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
    res = cv2.resize(frame,(224,224), interpolation = cv2.INTER_CUBIC)    

    img=image.array_to_img(res)
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img = preprocess_input(x)
    preds = ResNet50_model.predict(img)
    dec_preds=decode_predictions(preds, top=3)[0]
    #print('Predicted:', dec_preds)
    prob_pred=dec_preds[0][1]
    
    cv2.putText(frame,prob_pred,(10,500), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),3)
    out.write(frame)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()