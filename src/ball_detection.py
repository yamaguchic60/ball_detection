import cv2
import numpy as np
import time

#set parameters here
dev_num=2
capture_time=5


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
            break
        else:
            print(f'i checked {search_range_devices} devices, but i could not find it, confirm that the cable connects correctly.')
            exit(1)



#some process below



cap=cv2.VideoCapture(dev_num)
cap.set(cv2.CAP_PROP_FPS,60)
start_time=time.time()



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
    contours,_=cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for countour in contours:
        if cv2.contourArea(countour)>threshold_of_contour_length:
            moment=cv2.moments(countour)
            if moment['m00']!=0:
                cx=int(moment['m10']/moment['m00'])
                cy=int(moment['m01']/moment['m00'])
                cv2.circle(frame,(cx,cy),1,(0,0,0),-1)
                cv2.putText(frame,'center',(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                print(f'center:({cx},{cy})')

    cv2.drawContours(frame,contours,-1,(0,255,0),2)
    cv2.imshow(f'video_{frequency}Hz',frame)












    if time.time()-start_time>capture_time:
        break





cap.release()
cv2.destroyAllWindows()
