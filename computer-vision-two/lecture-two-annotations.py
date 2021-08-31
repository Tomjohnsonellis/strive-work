import cv2
import numpy as np
from helper_functions import display_image

img = cv2.imread('computer-vision-two/assets/gamer.jpg')
clone = img.copy()

cv2.ellipse(clone, (200,200), (100,50), 0, 0, 360, (100,64,255), -1)

cv2.circle(clone, (250,200), 25, (0,255,0), 10)
cv2.circle(clone, (250,200), 10, (0,0,0), -1)

cv2.circle(clone, (150,200), 25, (0,255,0), 10)
cv2.circle(clone, (150,200), 10, (0,0,0), -1)

cv2.line(clone, (100, 125), (300, 125), (0,0,0), 5)

polygon_points = np.array([[600,400], [500,200], [575,210], [600,175],[625, 210], [700,200]], np.int32)
polygon_points.reshape((-1,1,2))
cv2.polylines(clone, [polygon_points], True, (255,0,0), thickness=5)

# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
# cv2.polylines(img,[pts],True,(0,255,255))


overlay = clone.copy()
overlay = cv2.flip(overlay, 1)
opacity = 0.8
beta = 1 - opacity
clone = cv2.addWeighted(clone, opacity, overlay, beta, gamma=0 )

# center_coordinates = (120, 100)
# axesLength = (100, 50)
# angle = 0
# startAngle = 0
# endAngle = 360
# # Red color in BGR
# color = (0, 0, 255)
# thickness = 5
   
# # Using cv2.ellipse() method
# # Draw a ellipse with red line borders of thickness of 5 px
# cv2.ellipse(clone, center_coordinates, axesLength,
#            angle, startAngle, endAngle, color, thickness)
   




display_image(clone)