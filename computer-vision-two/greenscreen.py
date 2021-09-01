import cv2
import numpy as np

webcam_video = cv2.VideoCapture(0)
while(True):
    data_captured, frame = webcam_video.read()

    # So this will be a simple "greenscreen" or colour replacement program
    # Personally I have a white background in my work area so I'll filter that
    # The steps to do this should be something like:
    # Determine colours for replacement
    # Create a mask that detects those colours
    # Change those colours to something else
    # Overlay the changes onto the original image, that's our final image.

    image = frame.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Reds in HSV space can be tricky, we'll use two masks
    # lower_b = (-60, 0, 0)
    # upper_b = (20, 255, 255)
    lower_b = (0, 0, 0)
    upper_b = (15, 255, 255)
    backdrop_mask = cv2.inRange(image, lower_b, upper_b)
    second_lower_b = (70, 0, 150)
    second_upper_b = (110,150,255)
    second_backdrop_mask = cv2.inRange(image, second_lower_b, second_upper_b)
    combined_mask = cv2.bitwise_or(backdrop_mask, second_backdrop_mask)

    detected_area = cv2.bitwise_and(image, image, mask=backdrop_mask)
    second_detected_area = cv2.bitwise_and(image, image, mask=second_backdrop_mask)
    detections = np.concatenate((detected_area, second_detected_area), axis=1)

    reference_images = np.concatenate((frame, image), axis=1)
    comparison = np.concatenate((reference_images, detections), axis=0)
    cv2.imshow("Originals and Masks", comparison)

    double_masked_area = cv2.bitwise_and(image, image, mask=combined_mask)
    unmasked_area = cv2.bitwise_not(combined_mask)
    scene_area = cv2.bitwise_and(image, image, mask=unmasked_area)
    replacement_info = np.concatenate((double_masked_area, scene_area), axis=0)
    # cv2.imshow("Filter + NOT Filter", replacement_info)

    # After all that, we now have the area that we want to replace (double_masked_area)
    # As well as the area we want to leave unaffected (scene_area)
    # I'll go for an easy "Replace with red"
    # For extension, I could see if I could replace it with an image,
    # But that will require some fiddling with resizing.

    background = double_masked_area.copy()
    main_scene = scene_area.copy()

    background = cv2.cvtColor(background, cv2.COLOR_HSV2BGR)
    b,g,r = cv2.split(background)
    b[b > 0] = 128
    g[g > 0] = 0
    r[r > 0] = 128
    background = cv2.merge((b,g,r))

    main_scene = cv2.cvtColor(main_scene, cv2.COLOR_HSV2BGR)
    changed_scene = cv2.addWeighted(main_scene,1,background,1,0)

    repalcement_area_and_not = np.concatenate((background, main_scene), axis=1)
    scenes_comparison = np.concatenate((frame, changed_scene), axis=1)

    results = np.concatenate((repalcement_area_and_not, scenes_comparison), axis=0)
    cv2.imshow("FINAL", results)







    if cv2.waitKey(30) == ord('q'):
        break
webcam_video.release()
cv2.destroyAllWindows()
