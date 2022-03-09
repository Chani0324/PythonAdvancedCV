import cv2
from cv2 import VideoCapture
import mediapipe as mp
import time

cap = VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    succes, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # hands 모듈은 RGB에서만 쓸수있어서 바꿔줌
    results = hands.process(imgRGB) # RGB프레임
    # print(results.multi_hand_landmarks) # 손을 detecting 했으면 좌표값 landmark좌표값 표시

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms)

    cv2.imshow("Image", img)
    cv2.waitKey(1)