import cv2
import numpy as np

webcam_video = cv2.VideoCapture(0)

while(True):
    data_captured, frame = webcam_video.read()
    
    hsv = frame.copy()
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    low_b = (-10, 140, 0)
    high_b = (10, 255, 255)
    # low_b = (0, 0, 0)
    # high_b = (255, 255, 255)
    shirt_mask = cv2.inRange(hsv, low_b, high_b)
    masked_image = cv2.bitwise_and(hsv,hsv, mask=shirt_mask)
    cv2.imshow("Masked image", masked_image)

    # result = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR)
    result = masked_image
    original_and_mask = np.concatenate((frame, result), axis=1)
    cv2.imshow("Mask display", original_and_mask)




    if cv2.waitKey(30) == ord('q'):
        break
webcam_video.release()
cv2.destroyAllWindows()