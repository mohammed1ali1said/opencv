import cv2 as cv
import mediapipe as mp

class handTrackModule():
    def __init__(self, mode=False, maxnumofhands=2,complexity=1, detectionconf=0.5, trackconf=0.5):
        self.mode = mode
        self.maxnumofhands = maxnumofhands
        self.detectionconf = detectionconf
        self.trackconf = trackconf
        self.mpHands = mp.solutions.hands
        self.complexity = complexity
        self.hands = self.mpHands.Hands(self.mode, self.maxnumofhands, self.complexity, self.detectionconf, self.trackconf)
        self.mpDraw = mp.solutions.drawing_utils

    # ... rest of the code ...

    def findhands(self, frame):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # Fixed: Used imgRGB instead of self.imgRGB
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, handlms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findpos(self, frame, handno=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                cv.circle(frame, (cx, cy), 10, (255, 0, 255), -1)
        return lmList

def main():
    cap = cv.VideoCapture('handControl/WhatsApp Video 2023-08-29 at 19.51.28.mp4')
    detector = handTrackModule()

    while True:
        success, frame = cap.read()
        frame = detector.findhands(frame)
        cv.imshow("frame", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):  # Fixed: Used '&' instead of 'and'
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
