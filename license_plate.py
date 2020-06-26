from openalpr import Alpr
import cv2
import time
import argparse
import numpy as np
import os
import imutils

ALPR_CONFIG = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/openalpr/config/openalpr.defaults.conf'
RUNTIME_DATA = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/openalpr/runtime_data'

WEIGHTS = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/model/cfg/yolov3-tiny.weights'
CONFIG = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/model/cfg/yolov3-tiny.cfg'
NAMES = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/model/data/coco.names'

IMAGE_FOLDER = '/mnt/c/Users/Philip/Documents/GitHub/yolo_dev/app/static/images'

def process_video(video, license_plate, _id):
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

	video = cv2.VideoCapture(video)
	w, h = None, None

	while True:
		grabbed, frame = video.read()
		if not grabbed:
			break
		start = time.time()

		try:
			regions_of_interest, top_plate, confidence = get_info(frame, 'array')
			print(regions_of_interest, top_plate, confidence)
			#confirming license plate 
			if str(top_plate) == license_plate:
				draw_boxes(regions_of_interest, top_plate, confidence, frame)
				cv2.imwrite(f'{IMAGE_FOLDER}/confirmation{_id}.png', frame)
				return f'{IMAGE_FOLDER}/confirmation{_id}.png'
		except:
			print('License Plate not Detected')

		end = time.time()
		print(f"[INFO] Processing Time - {end-start}")

	video.release()
	alpr.unload()


