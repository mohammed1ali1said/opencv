import numpy as np
import cv2 as cv
import mediapipe as mp

mpDraw=mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.7 ,min_tracking_confidence=0.7)
cap = cv.VideoCapture('squats/squat.mp4')

circle_1_x, circle_2_x, circle_1_y, circle_2_y = 0,0,0,0
depth=False
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    frame=cv.resize(frame,(640,640))

    # Our operations on the frame come here
    results =pose.process(frame)
    # Display the resulting frame

    if results.pose_landmarks:
         for id,lm in enumerate(results.pose_landmarks.landmark):
             h,w,c=frame.shape
             if id==23:
                cx,cy=int (lm.x*w),int(lm.y*h)
                cv.circle(frame, (cx, cy), 3, (0,0,0), -1)
                circle_1_x=cx
                circle_1_y=cy
                # mpDraw.draw_landmarks(frame,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
             if id==25:
                 cx, cy = int(lm.x * w), int(lm.y * h)
                 cv.circle(frame, (cx, cy), 3, (0, 0, 0), -1)
                 circle_2_x=cx
                 circle_2_y=cy

    if circle_1_y >= circle_2_y:
        depth = True
    if depth == False:
       cv.line(frame, (circle_1_x,circle_1_y), (circle_2_x,circle_2_y), (0,0,255), 1)
    else :
       cv.line(frame, (circle_1_x, circle_1_y), (circle_2_x, circle_2_y), (0, 255, 0), 1)

    cv.imshow('frame', frame)


    if cv.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()
