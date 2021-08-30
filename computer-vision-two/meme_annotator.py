import cv2
import helper_functions
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# original_image = cv2.imread('computer-vision-two/assets/meme.jpg')

# top_text = input("Top text?: ")
bottom_text = "BOTTOM TEXT"

# We'll use PIL for this fun task as we want the meme font "impact"
# Load the image and rename some PIL functions for ease of use
img = Image.open("computer-vision-two/assets/meme.jpg")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("computer-vision-two/assets/impact.ttf", 128)


# These will be needed later
print(img.size)
h = img.height
w = img.width


# draw.text(xy=(25, h-200), text=bottom_text, font=font, align='center', fill=(255,255,255))
# I'll draw some lines so we can check if the text is aligned more easily
vertical_line_points = [(w/2,0), (w/2,h)]
draw.line(vertical_line_points, fill=(255,0,0), width=1)
horizontal_line_points = [(0, h/2), (w, h/2)]
draw.line(horizontal_line_points, fill=(255,0,0), width=1)


img = helper_functions.draw_text_with_border(img, "Center", font, location=None)
img = helper_functions.draw_text_with_border(img, "I am bottom text", font, location="bottom")
img = helper_functions.draw_text_with_border(img, "Top of the image", font, location="top")


img.save("computer-vision-two/assets/meme_with_text.jpg")
