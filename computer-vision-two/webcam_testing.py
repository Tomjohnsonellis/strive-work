import cv2
import numpy as np

webcam_video = cv2.VideoCapture(0)

# text_to_display = "^^^TOM^^^"
ft = cv2.freetype.createFreeType2()
ft.loadFontData(fontFileName="computer-vision-two/assets/impact.ttf", id=0)
# font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret, frame = webcam_video.read()
    # print(ret)
    
    # Put some text on the screen pointing to me
    cv2.putText(frame, 
    "^^^TOM^^^", 
    (125,450),
    0,
    2,
    (0,255,0),
    4
    )

    cv2.putText(frame, 
    "v v v TOM v v v", 
    (50,50),
    0,
    2,
    (0,255,0),
    4
    )



    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow("HSV", hsv)
    # cv2.imshow("BGR", frame)
    
    # Join the different colour spaced images together and display
    combo = np.concatenate((frame, hsv), axis=1)
    cv2.imshow("BGR | HSV", combo)

    if cv2.waitKey(1) == ord('q'):
        break


webcam_video.release()
cv2.destroyAllWindows()
    