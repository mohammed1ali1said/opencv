import cv2 as cv
import  mediapipe as mp
import time

class FaceDetector():
    def __init__(self,minDetectionCon=0.5):
        self.minDetectionCon=minDetectionCon
        self.mpFaceDetection=mp.solutions.face_detection
        self.mpDraw=mp.solutions.drawing_utils
        self.faceDetection=self.mpFaceDetection.FaceDetection(0.75)

    def findFaces(self, frame):
        imgrgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgrgb)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box  # Fix the typo here
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                cv.rectangle(frame, bbox, (255, 0, 255), 1)
                frame=self.fancyDraw(frame,bbox)
                cv.putText(frame, f'{int(detection.score[0] * 100)}%',
                           (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)

        return frame, bboxs

    def fancyDraw(self,img,bbox,l=30,t=5,rt=1):
        x,y,w,h=bbox
        x1,y1=x+w,y+h
        cv.rectangle(img,bbox,(255,0,255),rt)
        cv.line(img,(x,y),(x+l,y),(255,0,255),t)
        cv.line(img,(x,y),(x,y+l),(255,0,255),t)

        cv.line(img,(x1,y),(x1-l,y),(255,0,255),t)
        cv.line(img,(x1,y),(x1,y+l),(255,0,255),t)

        cv.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        cv.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img



def main():
  cap = cv.VideoCapture("squats/WhatsApp Video 2023-08-27 at 08.57.00.mp4")
  detector =FaceDetector()

  while True:
       success, frame = cap.read()

       if not success:
           break  # Exit the loop if there are no more frames
       frame, bboxs = detector.findFaces(frame)
       cv.imshow("frame", frame)
       cv.waitKey(1)

if __name__=="__main__":
     main()