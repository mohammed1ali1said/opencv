import cv2 as cv
import mediapipe as mp
import numpy as np

import handTrackModule as htm
import os

Wcam, Hcam = 640, 480
cap = cv.VideoCapture('handControl/WhatsApp Video 2023-08-29 at 19.51.28.mp4')
cap.set(3, 1080)
cap.set(4, 1080)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    newImage = cv.resize(image, (200, 200))
    # print(f'{folderPath}/{imPath}')
    overlayList.append(newImage)

detector = htm.handTrackModule(detectionconf=0.8)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, frame = cap.read()
    frame = detector.findhands(frame)
    lmList = detector.findpos(frame)

    if len(lmList) != 0:
        fingers = []
        if lmList[tipIds[0]][1]< lmList[tipIds[0]-1][1]:
            fingers.append(0)
        else:
            fingers.append(1)
        for id in range(1, 5):  # thumb is problematic
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers.count(1))

    frame[0:200, 0:200] = overlayList[0]

    cv.imshow("frame", frame)
    cv.waitKey(1)
