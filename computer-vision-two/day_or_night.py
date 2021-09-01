import cv2
import numpy as np
import os
from helper_functions import display_colour_histograms

# Task: See if we can guess if an image was taken during day or night
# No fancy ML/NNs here, just the raw image data, histograms and math.

# First, load in the images we'll be using
data_dir = "computer-vision-two/assets/daynight"

# Get all the filenames
filenames = []
with os.scandir(data_dir) as directory:
    for file in directory:
        # print(file.name)
        filenames.append(file.name)

# They're in a weird order, sort them out
filenames.sort()
# Use those filenames to load the images with OpenCV
# Images is a list 
images=[]
for name in filenames:
    images.append(cv2.imread(data_dir +'/'+ name))


# print(filenames)
# print(filenames[:4])
# print(filenames[4:])
# print(len(images))
# print(type(images[0]))
day_images = images[:4]
night_images = images[4:]


for image in day_images:
    print(image.mean())
print("~"*50)
for image in night_images:
    print(image.mean())
print("~"*50)

# Just taking the mean of values shows a pretty big difference
# What if we were to use HSV?
hsv = []
for index, image in enumerate(images):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print(image.mean())
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
# That actually gave us less of a clear distinction, so BGR mean it is!

def classify_image(some_image:np.ndarray) -> str:
    if some_image.mean() > 100:
        print("Day image")
        return "Day"
    else:
        print("Night image")
        return "Night"

for picture in images:
    classify_image(picture)

# You can choose a new image to test this method on
choice = input("Use own image?(Y/N): ")
if choice == "Y":
    print("eeeee")
else:
    pass

