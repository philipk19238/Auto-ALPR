from openalpr import Alpr
import cv2
import time
import argparse
import numpy as np
import os
import imutils

ALPR_CONFIG = '/mnt/c/Users/Philip/Documents/GitHub/openalpr/config/openalpr.defaults.conf'
RUNTIME_DATA = '/mnt/c/Users/Philip/Documents/GitHub/openalpr/runtime_data'

WEIGHTS = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/model/cfg/yolov3.weights'
CONFIG = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/model/cfg/yolov3.cfg'
NAMES = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/model/data/coco.names'

alpr = Alpr('us', ALPR_CONFIG, RUNTIME_DATA)
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    sys.exit(1)

alpr.set_top_n(5)
alpr.set_default_region("md")

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def get_info(frame, file_type):
	if file_type == 'array':
		results = alpr.recognize_ndarray(frame)
	else:
		results = alpr.recognize_file(frame)
	if results['results']:
		top_plate = results['results'][0]['candidates'][0]['plate']
		confidence = results['results'][0]['candidates'][0]['confidence']
		x1 = results["results"][0]["coordinates"][0]["x"]
		y1 = results["results"][0]["coordinates"][0]["y"]
		x2 = results["results"][0]["coordinates"][2]["x"]
		y2 = results["results"][0]["coordinates"][2]["y"]
		return [x1, y1, x2, y2], top_plate, confidence
	else:
		return False

def draw_boxes(regions_of_interest, plate_number, confidence, img):
	x, y, x1, y1 = regions_of_interest[0], regions_of_interest[1], regions_of_interest[2], regions_of_interest[3]
	cv2.rectangle(img, (x,y), (x1,y1), (0, 255, 0), 2)
	text = "{} {:.1f}%".format(plate_number, confidence)
	cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

with open(NAMES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(WEIGHTS, CONFIG)
video = cv2.VideoCapture('boston.mp4')
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(video.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

writer = None
w, h = None, None

while True:
	grabbed, frame = video.read()
	if not grabbed:
		break

	blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)

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
	try:
		regions_of_interest, top_plate, confidence = get_info(frame, 'array')
		print(regions_of_interest, top_plate, confidence)
		draw_boxes(regions_of_interest, top_plate, confidence, frame)
	except:
		print('License Plate not Detected')
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		writer = cv2.VideoWriter('new_test_test.avi', fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
	end = time.time()
	print(f"[INFO] Processing Time - {end-start}")
	writer.write(frame)

writer.release()
video.release()

"""
frame = cv2.imread('license.png')
regions_of_interest, top_plate, confidence = get_info(frame, 'array')
draw_boxes(regions_of_interest, top_plate, confidence, frame)
cv2.imwrite('test.png', frame)
"""
alpr.unload()
