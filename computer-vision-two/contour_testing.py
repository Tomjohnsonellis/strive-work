"""
This is more of a testing ground for getting used to contour detection
and doing things with thouse contours
"""

import cv2
import numpy as np
from helper_functions import display_image, scale_image

# First off, we will need  an image
original_image = cv2.imread("computer-vision-two/assets/chess.jpg")
# This image is a bit big to work with so I will resize it
factor = 0.5
resized_image = cv2.resize(original_image,(0,0), fx=factor, fy=factor)

# Next up, convert the image to grayscale
grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
# We'll apply some gaussian blur to reduce noise near the edges
blurred_image = cv2.GaussianBlur(grayscale_image, (5,5), 1)

# We'll try applying just a threshold and see how well that
# detects the edges of items of interest
# NOTE: the type of threshold and the value you need will
# vary from image to image, these values worked well
# FOR THIS SPECIFIC IMAGE
_, threshold = cv2.threshold(blurred_image, 250, 255, cv2.THRESH_TOZERO)

# # IF there's a small amount of noise inside the objects,
# # We should be able to clear that up with a morphological transformation
# # "Close" in this case, a dilation and then erosion
# close_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
# improved_threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, close_kernel)

gray_thresh_comparison = np.concatenate((grayscale_image, threshold), axis=0)
# display_image(gray_thresh_comparison)
# Great! We now have some silhouettes of the objects, onto the contours...
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# "contours" will be a list of lists, contours[0] would be all the data for that specific contour
# "hierarchy" might be helpful later
print(f"Objects Detected: {len(contours)}")
# We may as well draw some contours
# To draw them all, we use this, parameters are: 
# 1- The image to draw contours on
# 2- The contours from above
# 3- Which to draw (-1 for all)
# 4- Colour
# 5- Thickness
contour_image = cv2.drawContours(resized_image.copy(), contours, -1, (0,50,255), 4)
display_image(contour_image)
# Pretty good! The contours also seem to have mistaken the entire image for an object,
# Let's isolate that contour and get rid of it
print("="*25," Hierarchy Data ","="*25)
for index, hierarchy_info in enumerate(hierarchy[0]):
    print(f"Index {index}: {hierarchy_info} --- Sum: {hierarchy_info.sum()}")
# By inspecting the output, index 0 seems to have some strange values
# Let's draw just that contour...
single_contour = cv2.drawContours(resized_image.copy(), contours, 0, (0,0,255), 10)
display_image(single_contour, "Is this the problem?")
# Yep! That's the one we don't want. Delete it with:
del contours[0]
print(f"Actual Objects Detected: {len(contours)}")
# And redraw the contours
contour_image = cv2.drawContours(resized_image.copy(), contours, -1, (100,50,255), 4)
display_image(contour_image)
# Excellent! We now only have contours that contain objects in the image.
# We could also isolate each object by finding the bounding rectangle
bounding_boxes = []
print("="*25," Bounding Boxes Data ","="*25)
for contour in contours:
    # This gives some info that can be used to make a rectangle, not just the points needed to.
    x, y, w, h = cv2.boundingRect(contour)
    print([x,y,w,h])
    bounding_boxes.append([x,y,w,h])
    cv2.rectangle(contour_image, (x, y), (x+w, y+h), (255,0,0), 1)

display_image(contour_image)

# We can also use those bounding boxes to isolate the objects we want!
isolated_objects = np.array([])
for number, points in enumerate(bounding_boxes):
    x,y,w,h = points
    base_image = resized_image.copy()
    # NOTE: Careful with array slicing widths and heights, it's easy to get them muddled up.
    display_image(base_image[y:y+h,x:x+w], f"Object {number+1}")
    















