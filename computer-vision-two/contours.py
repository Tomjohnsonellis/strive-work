import cv2
import numpy as np

# Load an image, I've chosen some coins as they are nice shapes for this task
image = cv2.imread("assets/chess.jpg") # Windows
image = cv2.imread("computer-vision-two/assets/coins.jpg") # Linux
# Scale the image down
smaller_image = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))
# I only want the bottom half of this particular image
# smaller_image = smaller_image[160:,:]
# Convert to binary image / greyscale
grey = cv2.cvtColor(smaller_image, cv2.COLOR_BGR2GRAY)
# Using a bit of gaussian blur can lead to better results
blurred = cv2.GaussianBlur(grey, (5, 5), 2)
cv2.imshow("Blur", blurred)
# Using sobel first can lead to better results for Canny
# In this case, it did not.
# sobel_h = cv2.Sobel(blurred, 0, 1, 0, cv2.CV_64F)
# sobel_v = cv2.Sobel(blurred, 0, 0, 1, cv2.CV_64F)
# # Combine
# sobels = cv2.bitwise_or(sobel_h, sobel_v)
# I fiddled around with Canny and binary thresholds, eventually decided on binary
# canny_image = cv2.Canny(blurred, 100, 255)
ret, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
# There's a little bit of noise inside the coins, we can clear that up with a "close"
# Define the kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# Apply the close morph
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


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
# cv2.imshow("Result", smaller_image)

# Display the threshold being used and the resulting contours being applied to the image
thresh_3d = cv2.merge((thresh,thresh,thresh))
thresh_and_coins = np.concatenate((thresh_3d, smaller_image), axis=1)
cv2.imshow("Threshold and Contours", thresh_and_coins)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Next we can do some sorting
sorted_contours = sorted(contours, key=lambda a_contour: cv2.boundingRect(a_contour)[1])
for i, ctr in enumerate(sorted_contours):
    # The points will give us a bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Use that to find a "Region of interest" (not return on investment)
    roi = smaller_image[y:y+h, x:x+w]

    # Draw a box around it
    cv2.rectangle(smaller_image,(x,y),( x + w, y + h ),(255,0,0),2)
    # Display just that area
    # cv2.imshow(f"Object {i+1}", roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Finally we will display the image with both the contours and regions of interest
cv2.imshow("All the info",smaller_image)
cv2.waitKey(0)
cv2.destroyAllWindows()