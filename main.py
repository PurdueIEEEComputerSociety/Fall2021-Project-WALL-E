import numpy as np
import cv2
import os
import imutils
import math
from random import randrange

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2
previous = []

def distance(x1,y1,x2,y2):
    return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

def dominant_color(input, x, y, w, h):
    # TODO: have cropped image be smaller / more concentrated on chest area
    cropped_image = input[x:x+w, y:y+h]
    data = np.reshape(cropped_image, (-1, 3))
    print(data.shape)
    if (data.shape[0] > data.shape[1]):
        data = np.float32(data)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(data, 1, None, criteria, 10, flags)

        # TODO: use colors to differentiate people
        print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
        print(str(x) + " " + str(y) + " " + str(x + w) + " " + str(y + h))

def result_analysis(input, previous):
    final = []
    if previous:
        for pointIdx, pastPointData in enumerate(previous):
            validPoint = False
            validPoints = 0
            known = None
            for point in pastPointData:
                if point is not None:
                    validPoints += 1
                    if not validPoint:
                        validPoint = True
                        known = point
            if known is not None:
                lowest = [None, 5000, known[2], known[3]]
                index = -1
                for idx, each in enumerate(input):
                    value = distance(known[0][4], known[0][5], each[1][4], each[1][5])
                    if value < known[3] and value < lowest[1]:
                        lowest[0] = each[1]
                        lowest[1] = value
                if lowest[0] is not None:
                    if validPoints > 0:
                        previous[pointIdx].insert(0, lowest)
                        if len(previous[pointIdx]) > 64:
                            previous[pointIdx].pop(-1)
                        if validPoints > 32:
                            final.append(lowest)
                    input.pop(idx)
                else:
                    previous[pointIdx].insert(0, None)
                    if len(previous[pointIdx]) > 64:
                        previous[pointIdx].pop(-1)
            if validPoints == 0:
                previous.pop(pointIdx)
    while input:
        current = [[input.pop()[1], 0, (randrange(256), randrange(256), randrange(256)), 100]]
        previous.append(current)
    return final, previous

def pedestrian_detection(imagePar, modelPar, layerNamePar, personidz=0):
    (H, W) = imagePar.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(imagePar, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    modelPar.setInput(blob)
    layerOutputs = modelPar.forward(layerNamePar)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)

    if len(idzs) > 0:

        for i in idzs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            res = (confidences[i], (x, y, x + w, y + h, (x + w)/2, (y + h)/2), centroids[i])
            dominant_color(imagePar, x, y, w, h)
            results.append(res)

    return results


labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
'''
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
'''

layer_name = model.getLayerNames()
layer_name = [layer_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]
cap = cv2.VideoCapture(0)
writer = None

while True:
    (grabbed, image) = cap.read()

    if not grabbed:
        break
    image = imutils.resize(image, width=700)
    results = pedestrian_detection(image, model, layer_name,
                                   personidz=LABELS.index("person"))
    list1, previous = result_analysis(results,previous)



    for res in list1:
        cv2.rectangle(image, (res[0][0], res[0][1]), (res[0][2], res[0][3]), res[2], 2)

    cv2.imshow("Detection", image)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
