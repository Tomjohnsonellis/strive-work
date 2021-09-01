import cv2
from helper_functions import display_colour_histograms

# Let's check the colour distribution of a photo of a bus
img = cv2.imread('computer-vision-two/assets/bus.jpg')
#print(img.shape)
# Hopefully we will be able to see more red than other colours!
display_colour_histograms(img)
