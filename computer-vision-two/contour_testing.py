"""
This is more of a testing ground for getting used to contour detection
and doing things with thouse contours
"""

import cv2
import numpy as np
from helper_functions import display_image, scale_image

# First off, we will need  an image
image = cv2.imread("computer-vision-two/assets/chess.jpg")
# This image is a bit big to work with so I will resize it
factor = 0.5
image = cv2.resize(image,(0,0), fx=factor, fy=factor)

# Next up, convert the image to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# We'll apply some gaussian blur to reduce noise near the edges
image = cv2.GaussianBlur(image, (5,5), 1)

# We'll try applying just a threshold and see how well that
# detects the edges of items of interest
# Note: the type of threshold and the value you need will
# vary from image to image, these values worked well
# FOR THIS SPECIFIC IMAGE
_, threshold = cv2.threshold(image, 250, 255, cv2.THRESH_TOZERO)

# # IF there's a small amount of noise inside the objects,
# # We should be able to clear that up with a morphological transformation
# # "Close" in this case, a dilation and then erosion
# close_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
# improved_threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, close_kernel)

gray_thresh_comparison = np.concatenate((image, threshold), axis=0)
# display_image(gray_thresh_comparison)
# Great! We now have some silhouettes of the objects, onto the contours...
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# "contours" will be a list of lists, contours[0] would be all the data for that specific contour
# "hierarchy" will not be used for now
print(f"Objects Detected: {len(contours)}")







