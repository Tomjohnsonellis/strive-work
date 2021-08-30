import cv2
import helper_functions
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import tkinter
from tkinter import Tk, filedialog

root = Tk()
root.filename =  filedialog.askopenfilename(title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
print (root.filename)

img = Image.open(root.filename)
# img = Image.open("computer-vision-two/assets/meme.jpg")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("computer-vision-two/assets/impact.ttf", 128)

h = img.height
w = img.width

top_text = str(input("Please enter top text: ").upper())
bottom_text = str(input("Please enter bottom text: ").upper())

img = helper_functions.draw_text_with_border(img, top_text, location="top")
img = helper_functions.draw_text_with_border(img, bottom_text, location="bottom")

img.show()


img.save("computer-vision-two/assets/user_made_meme.jpg")
print("Your meme has been saved.")