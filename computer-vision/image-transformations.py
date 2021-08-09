import cv2
import numpy as np

# Load an image
image = cv2.imread("img/samoyed.jpg")
image_width = image.shape[1]
image_height = image.shape[0]

image_center = (image_width / 2, image_height / 2)

# Define transformation matrices
rotation_matrix = cv2.getRotationMatrix2D(center=image_center, angle=50, scale=1)
base = np.float32([[0, 50],
                   [200, 100],
                   [50, 200]])
transformed = np.float32([[100, 50],
                          [400, 20],
                          [25, 200]])
transformation_matrix = cv2.getAffineTransform(base, transformed)

# Apply transformations
rotated = cv2.warpAffine(src=image, M=rotation_matrix, dsize=(image_width, image_height))
skewed = cv2.warpAffine(src=image, M=transformation_matrix, dsize=(image_width, image_height))

cv2.imshow("Rotated", rotated)
cv2.imshow("Skewed", skewed)

# Upscaling
upscaled = cv2.resize(image, (image_width*2, image_height*2), interpolation=cv2.INTER_LANCZOS4)
cv2.imshow("Upscaled", upscaled)


cv2.waitKey(0)
cv2.destroyAllWindows()
