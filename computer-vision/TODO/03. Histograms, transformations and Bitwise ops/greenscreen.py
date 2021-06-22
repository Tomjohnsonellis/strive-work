import numpy as np
import cv2

# Load an image containing a green screen background
scene = cv2.imread("img/greenscreen.png")
scene = cv2.resize(scene, (640, 480))
scene_hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
# Load an image to replace the green screen with
replacement_image = cv2.imread("img/longcat.jpg")
replacement_image = cv2.resize(replacement_image, (640, 480))

# Green screen filter
lower_green = np.array([55, 125, 125])
upper_green = np.array(([70, 255, 255]))
green_screen_mask = cv2.inRange(scene_hsv, lower_green, upper_green)
green_screen_only = cv2.bitwise_and(scene, scene, mask=green_screen_mask)

# Make an image without the background
no_background = scene - green_screen_only
# Make an image with the background replaced
changed_scene = np.where(no_background == 0, replacement_image, no_background)

# Display all the images
cv2.imshow("Background Green Screen", green_screen_only)
cv2.imshow("Original Image", scene)
cv2.imshow("No Background", no_background)
cv2.imshow("Changed Scene", changed_scene)
cv2.waitKey(0)
cv2.destroyAllWindows()