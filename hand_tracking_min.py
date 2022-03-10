import cv2
from cv2 import VideoCapture
import mediapipe as mp
import time

cap = VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # hands 모듈은 RGB에서만 쓸수있어서 바꿔줌
    results = hands.process(imgRGB) # RGB프레임
    # print(results.multi_hand_landmarks) # 손을 detecting 했으면 좌표값 landmark좌표값 표시

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm) # 비율로 (x, y, z) 좌표값 나옴. z 좌표값은 손목 (id 0번)값이 거의 0에 수렴하도록 설정(origin).
                # z좌표의 값이 더 작을 수록 카메라에 더 가까움. 대충 x scale과 비슷한 값 곱해서 실제 좌표값 얻음.
                h, w, c = img.shape # height, width, channel
                cx, cy = int(lm.x*w), int(lm.y*h) # 실제 x, y 좌표값
                # print(id, cx, cy)
                # if id == 0:
                #     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)