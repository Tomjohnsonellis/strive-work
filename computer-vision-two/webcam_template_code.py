import cv2

webcam_video = cv2.VideoCapture(0)
while(True):
    data_captured, frame = webcam_video.read()


    """
    Put code here
    """


    if cv2.waitKey(30) == 27: # ESC Key to close
        break
webcam_video.release()
cv2.destroyAllWindows()
