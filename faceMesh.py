import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture("squats/WhatsApp Video 2023-08-27 at 08.57.00.mp4")

mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)

while True:
      success,frame=cap.read()
      imgRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
      res=faceMesh.process(imgRGB)
      if res.multi_face_landmarks:
          for facelms in res.multi_face_landmarks:
              mpDraw.draw_landmarks(frame, facelms, mpFaceMesh.FACEMESH_CONTOURS,landmark_drawing_spec=mpDraw.DrawingSpec(thickness=1,circle_radius=1))
      cv.imshow("Frame",frame)
      cv.waitKey(1)