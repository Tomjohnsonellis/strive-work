import cv2
import numpy as np

# Load an image, I've chosen some coins as they are nice shapes for this task
coins = cv2.imread("assets/coins.jpg")
# Scale the image down
smaller_image = cv2.resize(coins, (coins.shape[1] // 3, coins.shape[0] // 3))
# Convert to binary image / greyscale
grey = cv2.cvtColor(smaller_image, cv2.COLOR_BGR2GRAY)
# Using a bit of gaussian blur can lead to better results
blurred = cv2.GaussianBlur(grey, (11, 11), 2)

# Using sobel first can lead to better results for Canny
# In this case, it did not.
# sobel_h = cv2.Sobel(blurred, 0, 1, 0, cv2.CV_64F)
# sobel_v = cv2.Sobel(blurred, 0, 0, 1, cv2.CV_64F)
# # Combine
# sobels = cv2.bitwise_or(sobel_h, sobel_v)
# I fiddled around with Canny and binary thresholds, eventually decided on binary
canny_image = cv2.Canny(blurred, 100, 255)
ret, thresh = cv2.threshold(blurred, 180, 255, 1)
# Use cv2 to get all the contour information
contours, cont_info = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
# Draw it on the coloured image
cv2.drawContours(smaller_image, contours, -1, (0, 0, 255), 3)
print(cont_info)
print(f"We have detected: {len(contours)} edges!")

# And display!
# cv2.imshow("Hor", sobel_h)
# cv2.imshow("Ver", sobel_v)
# cv2.imshow("Sobels", sobels)

# cv2.imshow("Canny", canny_image)
# cv2.imshow("Thresholded", thresh)

cv2.imshow("Result", smaller_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
