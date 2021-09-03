import cv2
from helper_functions import display_image
import numpy as np

# Note, these methods do expect a black background
image = cv2.imread("computer-vision-two/assets/gamer.jpg", 0)
# display_image(image)
image = cv2.bitwise_not(image)

kernel = np.ones((3,7), np.uint8)
dilated_image = cv2.dilate(image, kernel, iterations=5)
# display_image(dilated_image)

fancy_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (11,11))
fancy_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT,fancy_kernel, iterations=1)
# display_image(fancy_image)


cv2.imshow("Original", image)
cv2.imshow("Changed", fancy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)


