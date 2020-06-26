import cv2
import argparse
import numpy as np
import os
import time
import imutils

WEIGHTS = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/model/cfg/yolov3-tiny.weights'
CONFIG = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/model/cfg/yolov3-tiny.cfg'
NAMES = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/model/data/coco.names'

with open(NAMES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(WEIGHTS, CONFIG)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

image = cv2.imread('streetview.jpg')

blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)

boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		classId = np.argmax(scores)
		confidence = scores[classId]
		if confidence > 0.5:
			h, w = image.shape[:2]
			box = detection[0:4] * np.array([w, h, w, h])
			centerX, centerY, width, height = box.astype('int')
			x = int(centerX - (width/2))
			y = int(centerY - (height/2))
			box = [x, y, int(x+width), int(y+height)]

			"""
			color = [int(c) for c in COLORS[classId]]
			cv2.rectangle(image, (x, y), (x+width, y+height), color, 2)
			confidence = "{:.2f}".format(confidence)
			text = f"{classes[classId]}: {confidence}"
			cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			"""

			boxes.append(box)
			confidences.append(float(confidence))
			classIDs.append(classId)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
if len(idxs) > 0:
	for i in idxs.flatten():
		x1, y1, x2, y2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
		text = f"{classes[classIDs[i]]}: {round(confidences[i], 2)}"
		cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imwrite('test_yolo.png', image)
