import cv2
import numpy as np
import Hand_Tracking_Module as htm
import pyautogui
import time

pyautogui.FAILSAFE = False

wCam, hCam = 640, 480
wScr, hScr = pyautogui.size()
frameR = 100

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.handDetector(maxHands=1)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    # 2. Get the tip of the index and thumbs
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[4][1:]
        
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only index finger : moving mode
        if fingers[0] == 0 and fingers[1] == 1:
            
            # 5. Convert coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            # 6. Smoothen values

            # 6. Move mouse
            pyautogui.moveTo(wScr - x3, y3, duration=0.1)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

        # 8. Both index and thumb are up : clicking mode
        if fingers[0] == 1 and fingers[1] == 1:
            pyautogui.leftClick()


    # 11. Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    # 12. Display
    img = cv2.flip(img, 1)

    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)