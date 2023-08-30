import cv2 as cv
import mediapipe as mp

class FaceMeshDetect():
    def __init__(self,staticmode=False,maxFaces=2,minDetect=0.5,minTack=0.5):
        self.staticmode=staticmode
        self.maxFaces=maxFaces
        self.minDetect=minDetect
        self.minTack=minTack
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh()
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)


    def findMeshFace(self,frame,draw=True):
       self.imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
       self.res = self.faceMesh.process(self.imgRGB)
       if self.res.multi_face_landmarks:
           for facelms in self.res.multi_face_landmarks:
              if draw:
               self.mpDraw.draw_landmarks(frame, facelms, self.mpFaceMesh.FACEMESH_CONTOURS,
               self.drawSpec,self.drawSpec)
       return frame

def main():
    cap = cv.VideoCapture("squats/WhatsApp Video 2023-08-27 at 08.57.00.mp4")
    detector= FaceMeshDetect()

    while True:
      success,frame=cap.read()
      frame=detector.findMeshFace(frame)
      cv.imshow("Frame",frame)
      cv.waitKey(1)




if __name__=="__main__":
    main()