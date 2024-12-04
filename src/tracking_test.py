import cv2
import numpy as np
import time
import math

def max_distance_2d(points):
    max_distance=0
    for i in range(len(points)):
        for j in range(i+1,len(points)):
            distance=math.sqrt((points[i][0]-points[j][0])**2+(points[i][1]-points[j][1])**2)
            max_distance=max(max_distance,distance)
    return max_distance






#set parameters here
dev_num=2
capture_time=5
use_camera=True

#declear the device number of camera,it is 0 if you have only one camera

# if you know dev_num, set the dev_num, if you do not know dev_num, set dev_num=-1
dev_num=-1
search_range_devices=100
capture_time=200
threshold_of_contour_length=100
frequency=60
# check video devices can connect
if dev_num==-1:
    for i in range(search_range_devices):
        cap=cv2.VideoCapture(i)
        if cap.isOpened():
            dev_num=i
            cap.release()
            print(f'camera device number is {dev_num}')
            break
        else:
            print(f'i checked {search_range_devices} devices, but i could not find it, confirm that the cable connects correctly.')
            exit(1)

#some process below

if use_camera:
    cap=cv2.VideoCapture(dev_num)
else:
    cap=cv2.VideoCapture('./../data/air_hockey.mp4')
cap.set(cv2.CAP_PROP_FPS,60)
start_time=time.time()
previous_cx=0
previous_cy=0

tracker=cv2.TrackerMIL_create()
bbox=(0,0,100,100)

while cv2.waitKey(int(1000/frequency)):
    ref,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_red=np.array([0,100,100])
    upper_red=np.array([10,255,255])
    mask1=cv2.inRange(hsv,lower_red,upper_red)
    lower_red=np.array([170,100,100])
    upper_red=np.array([180,255,255])
    mask2=cv2.inRange(hsv,lower_red,upper_red)
    mask=mask1+mask2
    res=cv2.bitwise_and(frame,frame,mask=mask)
    
    gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    cv2.imshow('video',gray)
    contours,_=cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    _,bbox=tracker.update(frame)

    for contour in contours:
        if cv2.contourArea(contour)>threshold_of_contour_length:
            moment=cv2.moments(contour)
            if moment['m00']!=0:
                cx=int(moment['m10']/moment['m00'])
                cy=int(moment['m01']/moment['m00'])
                cv2.circle(frame,(cx,cy),1,(0,0,0),-1)
                cv2.putText(frame,'center',(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                print(f'center:({cx},{cy})')
                print(f'center_velocity:{(cx-previous_cx)*1000/frequency},{(cy-previous_cy)*1000/frequency}')
                bbox=(cx,cy,10,10)
                cv2.rectangle(frame,(cx,cy),(cx+10,cy+10),(255,0,0),3)

    cv2.drawContours(frame,contours,-1,(0,255,0),2)
    cv2.imshow(f'video_{frequency}Hz',frame)












    if time.time()-start_time>capture_time:
        break





cap.release()
cv2.destroyAllWindows()
