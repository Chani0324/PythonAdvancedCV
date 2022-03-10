import cv2
from cv2 import VideoCapture
import mediapipe as mp
import time
import hand_tracking_module as htm

pTime = 0
cTime = 0
cap = VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    img = cv2.flip(img, 1)
    cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)