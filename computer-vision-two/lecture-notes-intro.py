from helper_functions import display_image
import cv2

# print(cv2.__version__)
img = cv2.imread('computer-vision-two/assets/person.jpeg')
print(img.shape)
# OpenCV is BGR
# helper_functions.display_image(img)

# Matplotlib is RGB
# import matp 

# In order to switch colour spaces, we can...
# rgb_version = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(rgb_version)
# plt.show()

# gray_version = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# helper_functions.display_image(gray_version)

# cv2 images are just numpy arrays so we can crop via slices
cropped_image = img[400:800, 300:700]
print(f"Cropped shape: {cropped_image.shape}")
# display_image(cropped_image)

# Before adjusting an image, it's helpful to make a copy in case anything goes wrong
clone = cropped_image.copy()
# For annotation of shapes, it's best to define the points and colour
# So for a rectangle
point1 = (50,50)
point2 = (100, 100)
colour = (0,0,255)
# draws a shape from point 1 to point 2, with a colour, and a thickness
cv2.rectangle(clone, point1, point2, colour, 5)
# You can also just use tuples, but it is harder to read
# This is a green rectangle
cv2.rectangle(clone, (275,50), (375,150), (0,255,0), 10)
# Line thickness -1 or cv2.FILLED will fill in the shape!
# This is a magenta box
cv2.rectangle(clone, (100,300), (350,375), (255, 0 ,255), cv2.FILLED)
display_image(clone)

annotated = clone.copy()
# Text can also be added
cv2.putText(annotated, "The shapes have taken me", (25,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
display_image(annotated)


# We have covered: colour spaces, cropping, shape and text annotations
# That's it for the basics!
print("---END---")
