import cv2

def drawBox(img,bbox):
    x,y,w,h=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(img,"Tracking",(75,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)


def main():
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    tracker=cv2.TrackerMIL_create()
    bbox=cv2.selectROI("select range",frame,True,True)
    tracker.init(frame,bbox)


    while True:
        _, frame = cap.read()
        _,bbox=tracker.update(frame)
        drawBox(frame,bbox)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()