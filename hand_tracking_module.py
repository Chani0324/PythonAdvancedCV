import cv2
from cv2 import VideoCapture
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # hands 모듈은 RGB에서만 쓸수있어서 바꿔줌
        self.results = self.hands.process(imgRGB) # RGB프레임
        # print(results.multi_hand_landmarks) # 손을 detecting 했으면 좌표값 landmark좌표값 표시

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm) # 비율로 (x, y, z) 좌표값 나옴. z 좌표값은 손목 (id 0번)값이 거의 0에 수렴하도록 설정(origin).
                # z좌표의 값이 더 작을 수록 카메라에 더 가까움. 대충 x scale과 비슷한 값 곱해서 실제 좌표값 얻음.
                h, w, c = img.shape # height, width, channel
                cx, cy = int(lm.x*w), int(lm.y*h) # 실제 x, y 좌표값
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = VideoCapture(0)
    detector = handDetector()
    
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        img = cv2.flip(img, 1)
        cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()