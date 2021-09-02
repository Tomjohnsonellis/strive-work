from types import FrameType
import cv2
import numpy as np

def make_a_circle(event, x, y, flags, params):
    if not event:
        cv2.circle(overlay, (x,y), 1, (64,64,64), 1)
    print("-"*50)

    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left MB Down")
        cv2.circle(overlay, (x,y), 25, (0,0,255), 10)

    if event == cv2.EVENT_LBUTTONUP:
        print("Left MB Up")
        cv2.circle(overlay, (x,y), 25, (0,255,0), 10)
    
    if event == 3: # Middle MB down
        cv2.rectangle(overlay, pt1=(x-5,y-5), pt2=(x+5,y+5), color=(255,0,255), thickness=2)
        global last_clicked_coords
        last_clicked_coords = (x,y)

    if event == 6: # Middle MB up
        cv2.rectangle(overlay, pt1=(x-5,y-5), pt2=(x+5,y+5), color=(255,255,0), thickness=2)
        cv2.rectangle(overlay, pt1=last_clicked_coords, pt2=(x,y), color=(255,255,255), thickness=int(y/50))
    
    print(f"Event: {event}")
    print(f"X: {x}")
    print(f"Y: {y}")
    print(f"Flags: {flags}")
    print(f"Parameters: {params}")
    



window_name = "I am a webcam"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, make_a_circle)


webcam_video = cv2.VideoCapture(0)
data_captured, freezeframe = webcam_video.read()
overlay = np.full_like(freezeframe, 0)
last_clicked_coords = (0,0)
while(True):
    _, frame = webcam_video.read()    

    frame = cv2.addWeighted(overlay, 1, frame, 1, 0)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) == ord('c'):
        overlay = np.full_like(freezeframe, 0)


    if cv2.waitKey(30) == 27:
        break
webcam_video.release()
cv2.destroyAllWindows()
