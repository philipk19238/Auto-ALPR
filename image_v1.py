import cv2
import argparse
import numpy as np
import os
import time
import imutils

WEIGHTS = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/model/cfg/yolov3.weights'
CONFIG = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/model/cfg/yolov3.cfg'
NAMES = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/model/data/coco.names'

with open(NAMES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(WEIGHTS, CONFIG)
vs = cv2.VideoCapture('_temp_tempboston.mp4')
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

writer = None
w, h = None, None

while True:
	grabbed, frame = vs.read()
	if not grabbed:
		break

	blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID =  np.argmax(scores)
			confidence = scores[classID]

			if confidence > 0.5:
				h, w = frame.shape[:2]
				box = detection[0:4] * np.array([w, h, w, h])
				centerX, centerY, width, height = box.astype('int')
				x = int(centerX - (width/2))
				y = int(centerY - (height/2))
				box = [x, y, int(x+width), int(y+height)]
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	if len(idxs) > 0:
		for i in idxs.flatten():
			x, y = boxes[i][0], boxes[i][1]
			w, h = boxes[i][2], boxes[i][3]
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		writer = cv2.VideoWriter('output.avi', fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

	writer.write(frame)

writer.release()
vs.release()





