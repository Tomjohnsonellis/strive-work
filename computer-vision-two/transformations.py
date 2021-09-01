import cv2
import numpy as np

webcam_video = cv2.VideoCapture(0)
while(True):
    data_captured, frame = webcam_video.read()

    x = 30
    y = 90
    translation_matrix =np.float32([
        [1, 0, x],
        [0, 1, y]
        ])


    
    warped_frame = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))

    cv2.imshow(warped_frame)
    

    if cv2.waitKey(30) == ord('q'):
        break
webcam_video.release()
cv2.destroyAllWindows()
