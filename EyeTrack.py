import  cv2 as cv
import mediapipe as mp
from google.protobuf.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import faceMeshModule
import math
from selenium import webdriver
from selenium.webdriver.chrome.service import Service






facemesh=faceMeshModule.FaceMeshDetect()


def detectEyes(frame,landmarks):
    if facemesh.res.multi_face_landmarks:
        leftEye=[]
        rightEye=[]
        for lm in landmarks:
            for landmark_id, landmark in enumerate(lm.landmark):
                if landmark_id in [33, 133, 145, 159, 362, 263, 374, 386]:  # IDs for eye landmarks

                    x_pixel, y_pixel = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv.circle(frame, (x_pixel, y_pixel), 1, (0, 255, 0), -1)
                    #145 bottom left eye
                    #159 top left eye
                    #374 bottom right eye
                    #386 top right eye
                    if landmark_id==374 or landmark_id==386:
                        rightEye.append(y_pixel)
                    if landmark_id==145 or landmark_id==159:
                       leftEye.append(y_pixel)
    return frame,leftEye,rightEye

def main():
 cap = cv.VideoCapture('eyeMovment/WhatsApp Video 2023-08-30 at 10.13.49.mp4')
 ################################################################################################### this code inside # can be replaced it was just a test
 # since this code only controls webpage view we can change it
 driver = None
 current_location = 0
 zoom = 1
 # Provide the path to the ChromeDriver executable
 driver_path = ChromeDriverManager().install()

 # Set up the Chrome driver service
 service = Service(executable_path=driver_path)

 # Create a Chrome webdriver instance
 driver = webdriver.Chrome(service=service)

 # Open the desired URL
 driver.get("https://en.wikipedia.org/wiki/Wikipedia")

 # Maximize the browser window
 driver.maximize_window()
##########################################################################
 while True:
    ReyeClosed=False
    LeyeClosed=False
    success,frame=cap.read()
    facemesh.findMeshFace(frame,draw=False)
    Leye=[]
    Reye=[]
    frame,Leye,Reye= detectEyes(frame,facemesh.res.multi_face_landmarks)
    if(math.hypot(Leye[1]-Leye[0])<8):
        LeyeClosed=True
    else:
        LeyeClosed=False

    if (math.hypot(Reye[1] - Reye[0]) < 8):
        ReyeClosed=True
    else:
        ReyeClosed =False


    if(LeyeClosed and ReyeClosed):
        zoom+=5/100
        driver.execute_script(f"document.body.style.zoom={zoom}")
    if (LeyeClosed and  not ReyeClosed):
        script = f"window.scrollBy(0, {current_location + 500});"
        driver.execute_script(script)
    if (not LeyeClosed and ReyeClosed):
        script = f"window.scrollBy(0, {current_location - 500});"
        driver.execute_script(script)
 ################################################################# also here the if statements can contain any other code

    cv.imshow('img',frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

 cap.release()
 cv.destroyAllWindows()

if __name__=="__main__":
    main()