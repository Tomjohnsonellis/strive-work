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

    not_in_mask = cv2.bitwise_not(shirt_mask)
    unaffected = cv2.bitwise_and(frame, frame, mask=not_in_mask)
    # cv2.imshow("XXXXX", unaffected)

    # Time to combine
    green_shirt = cv2.addWeighted(unaffected, 1, green_mask, 0.5, 5)
    blue_shirt = cv2.addWeighted(unaffected, 1, blue_mask, 0.5, 5)
    final_comparison = np.concatenate((frame, green_shirt, blue_shirt), axis=1)
    cv2.imshow("R | G | B", final_comparison)

    # old_green_shirt = cv2.addWeighted(frame, 1, green_mask, 0.8, 5)
    # old_blue_shirt = cv2.addWeighted(frame, 1, blue_mask, 0.8, 5)
    # old_comparison = np.concatenate((frame, old_green_shirt, old_blue_shirt), axis=1)
    # cv2.imshow("Old Comparison", old_comparison)


    # Let's filter my blue water bottle
    bottle_lb = (100, 40, 100)
    bottle_ub = (110, 200, 255)
    bottle_mask = cv2.inRange(hsv, bottle_lb, bottle_ub)
    bottle_detection = cv2.bitwise_and(hsv, hsv, mask=bottle_mask)

    bottle_detection = cv2.cvtColor(bottle_detection, cv2.COLOR_HSV2BGR)
    shirt_detection = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR)

    shirt_and_bottle = cv2.bitwise_or(bottle_detection, shirt_detection)

    # print("="*50)
    # print(f"Bottle detection: {bottle_detection.shape}")
    # print(f"Shirt detection: {shirt_detection.shape}")
    # print(f"Both: {shirt_and_bottle.shape}")

    tracking_two = np.concatenate((bottle_detection, shirt_and_bottle, shirt_detection ), axis=1)
    cv2.imshow("Tracking two things", tracking_two)
    # cv2.imshow("bottle", bottle_detection)
    # cv2.imshow("bottle", shirt_and_bottle)

    # shirt_mask = cv2.inRange(hsv, low_b, high_b)
    # masked_image = cv2.bitwise_and(hsv,hsv, mask=shirt_mask)

    if cv2.waitKey(30) == ord('q'):
        break
webcam_video.release()
cv2.destroyAllWindows()




