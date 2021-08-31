import cv2
import numpy as np

webcam_video = cv2.VideoCapture(0)

while(True):
    data_captured, frame = webcam_video.read()
    
    hsv = frame.copy()
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    low_b = (-10, 155, 10)
    high_b = (10, 255, 255)
    # low_b = (0, 0, 0)
    # high_b = (255, 255, 255)
    shirt_mask = cv2.inRange(hsv, low_b, high_b)
    masked_image = cv2.bitwise_and(hsv,hsv, mask=shirt_mask)
    # cv2.imshow("Masked image", masked_image)

    original_and_mask = np.concatenate((frame, hsv, masked_image), axis=1)
    cv2.imshow("Original | HSV | Mask", original_and_mask)

    green_mask = masked_image.copy()
    green_mask = cv2.cvtColor(green_mask, cv2.COLOR_HSV2BGR)
    green_mask[:,:,0][green_mask[:,:,0] > 0] = 0
    green_mask[:,:,1][green_mask[:,:,1] > 0] = 255
    green_mask[:,:,2][green_mask[:,:,2] > 0] = 0

    blue_mask = masked_image.copy()
    blue_mask = cv2.cvtColor(blue_mask, cv2.COLOR_HSV2BGR)
    blue_mask[:,:,0][blue_mask[:,:,0] > 0] = 255
    blue_mask[:,:,1][blue_mask[:,:,1] > 0] = 0
    blue_mask[:,:,2][blue_mask[:,:,2] > 0] = 0

    # print(green_mask.shape)
    # cv2.imshow("Green", green_mask)
    # cv2.imshow("Blue", blue_mask)

    # Time to combine
    green_shirt = cv2.addWeighted(green_mask, 1, frame, 1, 0)
    blue_shirt = cv2.addWeighted(blue_mask, 1, frame, 1, 0)
    final_comparison = np.concatenate((frame, green_shirt, blue_shirt), axis=1)
    cv2.imshow("R | G | B", final_comparison)



    if cv2.waitKey(30) == ord('q'):
        break
webcam_video.release()
cv2.destroyAllWindows()