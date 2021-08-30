import cv2
import helper_functions
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import textwrap

import tkinter
from tkinter import Tk, filedialog
"""
This is a version of today's work on image annotations that
can be ran in the command line and will save an image with text
"""

# No idea how to use tkinter, but this code block works fine
# For letting the user choose a filepath
root = Tk()
root.filename =  filedialog.askopenfilename(title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
print (root.filename)

# Load the selected image with PIL
img = Image.open(root.filename)
# img = Image.open("computer-vision-two/assets/meme.jpg")
# To save us some typing...
draw = ImageDraw.Draw(img)

# Grab the image shape
h = img.height
w = img.width
# Define the font we will use, which changes size based on the image
font = ImageFont.truetype("computer-vision-two/assets/impact.ttf", int(w/15))


# Allow the user to choose some text
top_text = str(input("Please enter top text: ").upper())
bottom_text = str(input("Please enter bottom text: ").upper())
# Wrap that text so it doesn't go off the screen
top_text = textwrap.fill(top_text, width = 30)
bottom_text = textwrap.fill(bottom_text, width = 30)

# Finally draw that text
img = helper_functions.draw_text_with_border(img, top_text, font, location="top")
img = helper_functions.draw_text_with_border(img, bottom_text, font, location="bottom")

img.show()


img.save("computer-vision-two/assets/user_made_meme.jpg")
print("Your meme has been saved.")