import cv2, sys, os
import numpy as np
import cv2
import numpy as np
import glob
import random
import os

class Detector:
	# This example of a detector detects faces. However, you have annotations for ears!

	# Load Yolo
	net = cv2.dnn.readNet("yolov3_custom_4000.weights", "yolov3_testing.cfg")

	# Name custom object
	classes = ["Ear"]

	# Images path
	images_path = glob.glob("C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/MOJE/data/ears/temp2/*.png")

	layer_names = net.getLayerNames()
	# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))

	results = dict()
	# Insert here the path of your images
	random.shuffle(images_path)
	# loop through all the images
	count = 0
	def detect(self, img):
		# Loading image
		img = cv2.imread(img)
		# img = cv2.resize(img, None, fx=0.4, fy=0.4)
		layer_names = self.net.getLayerNames()
		output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

		height, width, channels = img.shape

		# Detecting objects
		blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

		self.net.setInput(blob)
		outs = self.net.forward(output_layers)

		# Showing informations on the screen
		class_ids = []
		confidences = []
		boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.3:
					# Object detected
					# print(class_id)
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					# Rectangle coordinates
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)

					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)

		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
		# print(indexes)
		font = cv2.FONT_HERSHEY_PLAIN
		print(boxes)
		for i in range(len(boxes)):
			if i in indexes:
				x, y, w, h = boxes[i]
				label = str(self.classes[class_ids[i]])
				color = self.colors[class_ids[i]]
				cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
				cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
		# results[self.img_path.split(os.sep)[10]] = boxes
		return boxes
	# show image
	# cv2.imshow("Image", img)



	# if __name__ == '__main__':
	# 	fname = sys.argv[1]
	# 	img = cv2.imread(fname)
	# 	# detector = CascadeDetector()
	# 	detected_loc = detectYolo(img)
	# 	for x, y, w, h in detected_loc:
	# 		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	# 	cv2.imwrite(fname + '.detected.jpg', img)