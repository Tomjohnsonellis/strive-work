import cv2
import numpy as np

webcam_video = cv2.VideoCapture(0)




while(True):
    data_captured, frame = webcam_video.read()
    
    # We'll be working with the HSV colour space today so...
    hsv = frame.copy()
    # Convert to hsv
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    # For this task I was wearing a red shirt so figured that could be fun to detect
    # Let's start by detecting reds
    low_b = (-10, 155, 10)
    high_b = (10, 255, 255)
    # Create a mask of any pixels that meet that criteria
    shirt_mask = cv2.inRange(hsv, low_b, high_b)
    # Perform AND to display just what is detected by the mask
    masked_image = cv2.bitwise_and(hsv,hsv, mask=shirt_mask)
    # Stitch the original, hsv and mask views together to compare
    original_and_hsv_and_mask = np.concatenate((frame, hsv, masked_image), axis=1)
    cv2.imshow("Original | HSV | Mask", original_and_hsv_and_mask)

    # Next up, try changing the reds to greens
    green_mask = masked_image.copy()
    # As I am more familiar for BGR, I'll just convert from HSV for this
    green_mask = cv2.cvtColor(green_mask, cv2.COLOR_HSV2BGR)
    # Directly assign values to each colour channel, this would be
    # Blue: 0, Green: 255, Red: 0
    # The colons can make this look confusing, these statements are just
    # "Replace any non-zero values with X"
    green_mask[:,:,0][green_mask[:,:,0] > 0] = 0
    green_mask[:,:,1][green_mask[:,:,1] > 0] = 255
    green_mask[:,:,2][green_mask[:,:,2] > 0] = 0

    # Similar process for changing reds to blue
    blue_mask = masked_image.copy()
    blue_mask = cv2.cvtColor(blue_mask, cv2.COLOR_HSV2BGR)
    blue_mask[:,:,0][blue_mask[:,:,0] > 0] = 255
    blue_mask[:,:,1][blue_mask[:,:,1] > 0] = 0
    blue_mask[:,:,2][blue_mask[:,:,2] > 0] = 0


    # In order to display the replaced values instead of the originals,
    # we need to know what parts of the original image are unaffected
    not_in_mask = cv2.bitwise_not(shirt_mask)
    unaffected = cv2.bitwise_and(frame, frame, mask=not_in_mask)
    # cv2.imshow("XXXXX", unaffected)

    # Now we will overlay the colour-replaced shirt on top of
    # an image where everything not in the mask is pure black
    # The results are the colours being entirely replaced!
    green_shirt = cv2.addWeighted(unaffected, 1, green_mask, 1, 5)
    blue_shirt = cv2.addWeighted(unaffected, 1, blue_mask, 1, 5)
    # Stitch together for easiar viewing
    final_comparison = np.concatenate((frame, green_shirt, blue_shirt), axis=1)
    cv2.imshow("R | G | B", final_comparison)


    # # Originally I was unsure how to handle the replaced colours just being
    # # placed on top of the original image, legacy code below
    # old_green_shirt = cv2.addWeighted(frame, 1, green_mask, 0.8, 5)
    # old_blue_shirt = cv2.addWeighted(frame, 1, blue_mask, 0.8, 5)
    # old_comparison = np.concatenate((frame, old_green_shirt, old_blue_shirt), axis=1)
    # cv2.imshow("Old Comparison", old_comparison)


    # Great, next up, filtering blues
    # Let's filter my blue water bottle
    bottle_lb = (100, 40, 100)
    bottle_ub = (110, 200, 255)
    bottle_mask = cv2.inRange(hsv, bottle_lb, bottle_ub)
    bottle_detection = cv2.bitwise_and(hsv, hsv, mask=bottle_mask)

    bottle_detection = cv2.cvtColor(bottle_detection, cv2.COLOR_HSV2BGR)
    shirt_detection = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR)

    shirt_and_bottle = cv2.bitwise_or(bottle_detection, shirt_detection)


    tracking_two = np.concatenate((bottle_detection, shirt_and_bottle, shirt_detection ), axis=1)
    cv2.imshow("Tracking two things", tracking_two)

    if cv2.waitKey(30) == ord('q'):
        break
webcam_video.release()
cv2.destroyAllWindows()




