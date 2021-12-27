from flask import Flask, request
from flask_cors import CORS
import json
from PIL import Image
import base64
import io
import os
import shutil
import time
import cv2 as cv
import numpy as np
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

app = Flask(__name__)
CORS(app)

@app.route('/')
def success():
   return 'welcome'
@app.route('/api', methods=['POST', 'GET'])
def api():
	data = request.get_json()
	result = data['data']
	b = bytes(result, 'utf-8')
	image = b[b.find(b'/9'):]
	im = Image.open(io.BytesIO(base64.b64decode(image)))
	cap1 = cv.cvtColor(np.array(im),cv.COLOR_RGB2BGR)
	#directory = './stranger'
	#im.save(directory + '/face.jpeg')
	faceProto = "opencv_face_detector.pbtxt"
	faceModel = "opencv_face_detector_uint8.pb"

	genderProto = "gender_deploy.prototxt"
	genderModel = "gender_net.caffemodel"

	MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
	ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
	genderList = ['Male', 'Female']

	# Load network
	genderNet = cv.dnn.readNet(genderModel, genderProto)
	faceNet = cv.dnn.readNet(faceModel, faceProto)

	# Open a video file or an image file or a camera stream
	#cap = cv.imread('./stranger/face.jpeg')
	padding = 20

	# Read frame
	t = time.time()
	frameFace, bboxes = getFaceBox(faceNet, cap1)
	if not bboxes:
		print("No face Detected, Checking next frame")
	res = 'None'
	if len(bboxes)>1:
		return res
	res = 'None'
	for bbox in bboxes:
		face = cap1[max(0, bbox[1] - padding):min(bbox[3] + padding, cap1.shape[0] - 1),max(0, bbox[0] - padding):min(bbox[2] + padding, cap1.shape[1] - 1)]
		blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
		genderNet.setInput(blob)
		genderPreds = genderNet.forward()
		gender = genderList[genderPreds[0].argmax()]
		res = gender

		#label = "{}".format(gender)
		#cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,cv.LINE_AA)
		#cv.imshow("Age Gender Demo", frameFace)
		#cv.waitKey(0)
	# cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
	#path = "./stranger/face.jpeg"
	#os.remove(path)
	return res

if __name__ == '__main__':
	app.run()