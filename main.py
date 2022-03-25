import numpy as np
import cv2
import imutils
import math
from random import randrange
from sklearn.cluster import KMeans
from collections import Counter


NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2
previous = []

imageHeight = 0
imageWidth = 0

class color:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

class personData:
    def __init__(self, cords):
        self.confidence = cords[0]
        self.x1 = cords[1][0]
        self.y1 = cords[1][1]
        self.x2 = cords[1][2]
        self.y2 = cords[1][3]
        self.height = abs(self.y2 - self.y1)
        self.width = abs(self.x2 - self.x1)
        self.centerX = cords[1][4]
        self.centerY = cords[1][5]
        xRatio = .2
        yRatio = .2
        self.cropW = int((xRatio) * (self.x2 - self.x1))
        self.cropH = int((yRatio * (self.y2 - self.y1)))
        self.cropX = int(self.x1 + ((1 - xRatio) * (self.x2 - self.x1) / 2))
        self.cropY = int(self.y1 + ((1 - yRatio) * (self.y2 - self.y1) / 2))
        self.croppedImage = image[self.cropY:self.cropY + self.cropH, self.cropX:self.cropX + self.cropW]
        #self.domColor = dominant_color(self.croppedImage)
        self.averageColor = average_color(self.croppedImage)
        self.currentPerson = None

class personObj:
    def __init__(self, personData):
        self.MAXFRAMES = 128
        self.MAXSTORED = 16
        self.MINREL = 32
        self.MAXLIFE = 16
        self.MINFRAMES = 8
        self.frames = [personData]
        self.frameCount = 16
        self.validFrames = []
        for i in range(16 - 1):
            self.validFrames.append(True)
        self.currentScore = -1
        self.currentFrame = None
        self.claimed = True
        self.lifeTime = 16

    def calculateScore(self, frame):
        currentDistanceScore = distanceScore(self.frames[0].centerX, self.frames[0].centerY, frame.centerX, frame.centerY, imageWidth, imageHeight)
        aveColorSum = 0
        for each in self.frames:
            aveColorSum = aveColorSum + colorScore(each.averageColor, frame.averageColor)
        aveColorScore = aveColorSum / len(self.frames)
        return currentDistanceScore + aveColorScore

    def addFrame(self, frame):
        if frame is not None:
            self.currentFrame = frame
            self.validFrames.insert(0,True)
            self.lifeTime = self.MAXLIFE
            self.claimed = True
            self.frames.insert(0, frame)
            self.frameCount = self.frameCount + 1
            if len(self.frames) > self.MAXSTORED:
                self.frames.pop(-1)
        if frame is None:
            self.validFrames.insert(0,False)
            self.lifeTime = self.lifeTime - 1
        if len(self.validFrames) > self.MAXFRAMES:
            if(self.validFrames.pop(-1) is True):
                self.frameCount = self.frameCount - 1

    def reset(self):
        self.currentFrame = None
        self.claimed = False

    def checkPerson(self):
        if (self.lifeTime > 0 and self.frameCount > self.MINFRAMES):
            return False
        else:
            return True

def distance(x1,y1,x2,y2):
    return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

def distanceScore(x1, y1, x2, y2, imageWidth, imageHeight):
    distanceRatio = distance(x1, y1, x2, y2) / math.sqrt((imageWidth ** 2) + (imageHeight ** 2))
    return 1 - distanceRatio

def colorScore(color1, color2):
    colorDist = math.sqrt(((color1.r - color2.r) ** 2) + ((color1.g - color2.g) ** 2) + ((color1.b - color2.b) ** 2))
    colorRatio = colorDist / math.sqrt(((255) ** 2) + ((255) ** 2) + ((255) ** 2))
    return 1 - colorRatio

def dominant_color(cropped_image):
    # TODO: have cropped image be smaller / more concentrated on chest area
    # print(data.shape)
    tempImage = cropped_image.reshape((cropped_image.shape[0] * cropped_image.shape[1], 3))
    clusters = KMeans(n_clusters = 1)
    labels = clusters.fit_predict(tempImage)
    counts = Counter(labels)
    bgr = clusters.cluster_centers_[counts.most_common(1)[0][0]]
    colorObj = color(bgr[0], bgr[1], bgr[2])
    return colorObj

def average_color(cropped_image):

    # Average can possibly have a divide by zero error, but nanmean returns
    # [nan nan nan] in place of [r-value, g-value, b-value] when this happens.
    # This CAN result in runtime errors but it doesn't crash

    avg_per_row = np.nanmean(cropped_image, axis=0)
    avg_color = np.nanmean(avg_per_row, axis=0)
    colorObj = color(avg_color[0], avg_color[1], avg_color[2])
    return colorObj

# detects if person is right or left of screen
# TODO: figure out what to tell hardware if left or right
def left_or_right(person):
    if (person == None or person.frames[0] == None):
        return

    centerY = imageHeight // 2
    centerX = imageWidth // 2

    xDiff = (centerX - person.frames[0].centerX) / centerX
    yDiff = (centerY - person.frames[0].centerY) / centerY

    x1 = person.frames[0].x1
    x2 = person.frames[0].x2
    y1 = person.frames[0].y1
    y2 = person.frames[0].y2

    if (y2 == imageHeight):
        print("TOO TALL TOO CLOSE !!!!!!!!!!!!!!!")

    print("X difference: " + str(xDiff))
    return (xDiff)

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

def dataFormatter(data):
    outputList = []
    for each in data:
        outputList.append(personData(each))
    return outputList

def newAnalysis(newDataList, people):
    newDataList = dataFormatter(newDataList)
    print(len(newDataList))
    print(len(people))
    for person in people:
        person.reset()
    for newDataPoint in newDataList:
        indexOfPerson = -1
        highestScore = -1
        for idx, person in enumerate(people):
            if (not person.claimed):
                currentScore = person.calculateScore(newDataPoint)
                if (currentScore > 1.5):
                    if (currentScore > highestScore):
                        indexOfPerson = idx
                        highestScore = currentScore
        if indexOfPerson == -1:
            print("hjere")
            people.append(personObj(newDataPoint))
        else:
            people[indexOfPerson].addFrame(newDataPoint)
    for person in people:
        if not person.claimed:
            person.addFrame(None)
        if person.checkPerson():
            people.remove(person)
    return people




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
            res = (confidences[i], (x, y, x + w, y + h, centroids[i][0], centroids[i][1]))
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
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]
cap = cv2.VideoCapture(0)
writer = None
people = []

while True:
    (grabbed, image) = cap.read()

    if not grabbed:
        break
    image = imutils.resize(image, width=700)
    if (imageHeight == 0):
        imageHeight = image.shape[:2][0]
        imageWidth = image.shape[:2][1]
    results = pedestrian_detection(image, model, layer_name,
                                   personidz=LABELS.index("person"))
    print(results)
    people = newAnalysis(results, people)
    for each in people:
        left_or_right(each)


    for person in people:
        if (person.frameCount > person.MINREL):
            cv2.rectangle(image, (person.frames[0].x1, person.frames[0].y1), (person.frames[0].x2, person.frames[0].y2), (person.frames[0].averageColor.r, person.frames[0].averageColor.g, person.frames[0].averageColor.b), 3)
            cv2.rectangle(image, (person.frames[0].cropX, person.frames[0].cropY), (person.frames[0].cropX + person.frames[0].cropW, person.frames[0].cropY + person.frames[0].cropH), (0,0,0), 1)

    cv2.imshow("Detection", image)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
