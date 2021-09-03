import cv2
from helper_functions import display_image

image = cv2.imread("computer-vision-two/assets/threshold_test.png", 0)
size = (500,500)
# image = cv2.resize(image, size, cv2.INTER_CUBIC)
display_image(image)



thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25,2)
display_image(thresh)
# Gaussian blue can help give better results when using Otsu
blurred = cv2.GaussianBlur(image, (5,5), 0)
# Otsu has some odd syntax, the 9999 can be anything and it uses thresh+thresh
threshold_value, otsu = cv2.threshold(blurred, 9999, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
display_image(otsu)