import cv2 as cv
import numpy as np
import math
import handTrackModule
import handTrackModule as htm
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume

#params
Wcam,Hcam=640,480

#video reader
cap=cv.VideoCapture('handControl/WhatsApp Video 2023-08-29 at 19.51.28.mp4')

detector=htm.handTrackModule(detectionconf=0.8)

devices= AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))

# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0,None)
volRange=volume.GetVolumeRange()
minVol=volRange[0]
maxVol=volRange[1]

#change camera window size
cap.set(3,Wcam)
cap.set(4,Hcam)

while True:
    success,frame=cap.read()
    frame=detector.findhands(frame)
    lmList=detector.findpos(frame)
    if len(lmList) !=0:
        print(lmList[4],lmList[8])
        x1,y1=lmList[4][1],lmList[4][2]
        x2,y2=lmList[8][1],lmList[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2

        cv.circle(frame,(x1,y1),15,(255,0,255),cv.FILLED)
        cv.line(frame,(x1,y1),(x2,y2),(255,0,255),2)
        length=math.hypot(x2-x1,y2-y1)
        vol=np.interp(length,[50,200],[minVol,maxVol])
        volume.SetMasterVolumeLevel(vol,None)
        if length<50:
            cv.circle(frame,(cx,cy),15,(0,255,0),cv.FILLED)


    cv.imshow("frame",frame)
    cv.waitKey(1)