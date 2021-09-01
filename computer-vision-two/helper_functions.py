import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np

def display_image(image_array, image_name="My Image") -> None:
    cv2.imshow(image_name, image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return

def annotate_quadrangle(image_array, points_for_quadrangle, thickness=5):
    thickness = thickness
    image_with_quadrangle = image_array.copy()
    cv2.line(image_with_quadrangle, points_for_quadrangle[0], points_for_quadrangle[1], (0,0,255), thickness)
    cv2.line(image_with_quadrangle, points_for_quadrangle[1], points_for_quadrangle[2], (255,0,0), thickness)
    cv2.line(image_with_quadrangle, points_for_quadrangle[2], points_for_quadrangle[3], (0,255,0), thickness)
    cv2.line(image_with_quadrangle, points_for_quadrangle[3], points_for_quadrangle[0], (255,128,255), thickness)
    return image_with_quadrangle

def draw_text_with_border(image, text, font, x_pos=25, y_pos=25, location=None, border_size=4) -> np.ndarray:
    draw = ImageDraw.Draw(image)
    font = font
    # font = ImageFont.truetype("computer-vision-two/assets/impact.ttf", font)
    # We can use PIL to calculate the size of the text block
    text_w, text_h = draw.textsize(text, font)
    # print(f"W: {text_w}\nH: {text_h}")



    if not location:
        x_pos = (image.width/2) - (text_w / 2)
        y_pos = (image.height/2) - (text_h / 2)
    if location == "bottom":
        x_pos = (image.width/2) - (text_w /2)
        y_pos = (image.height*0.95) - (text_h)
    if location == "top":
        x_pos = (image.width/2) - (text_w /2)
        y_pos = (image.height*0.01)

    draw_text_underneath(text, x_pos, y_pos, draw, font, border_size)


    return image
        

def draw_text_underneath(text, x_pos, y_pos, draw, font, border_size) -> None:
    draw = draw
    font = font
    border_size = border_size
    # Draw the same text in black slightly offset in each direction
    draw.text(xy=(x_pos-border_size, y_pos-border_size), text=text, font=font, align='center', fill=(0,0,0))
    draw.text(xy=(x_pos-border_size, y_pos+border_size), text=text, font=font, align='center', fill=(0,0,0))
    draw.text(xy=(x_pos+border_size, y_pos-border_size), text=text, font=font, align='center', fill=(0,0,0))
    draw.text(xy=(x_pos+border_size, y_pos+border_size), text=text, font=font, align='center', fill=(0,0,0))
    # Draw the actual text
    draw.text(xy=(x_pos, y_pos), text=text, font=font, align='center', fill=(255,255,255))
    return


def display_colour_histograms(some_image_in_BGR_form:np.ndarray, cvt_to_hsv:bool=False) -> None:
    image = some_image_in_BGR_form
    
    if cvt_to_hsv:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        colours = ['r','b','black']
        plt.title("HSV Data for image")
        labels = ["Hue", "Sat", "Val"]
    else:
        colours = ['b','g','r']
        plt.title("Colour values in image")
    
    for i, colour in enumerate(colours):
        hist = cv2.calcHist([image[:,:,i]] , [0], None, [256],[0,255])
        plt.plot(hist, color=colour)

    if cvt_to_hsv: plt.legend(labels)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    return