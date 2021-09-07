import cv2
import numpy as np
from helper_functions import display_image, scale_image

font = cv2.FONT_HERSHEY_TRIPLEX
text_col = (255,255,255)

def draw_titlebox(image, text) -> np.array:
    cv2.rectangle(image, (0,0),(image.shape[1],50), (0,0,0), -1)
    cv2.putText(image, text, (10,35), font, 1, text_col)
    return image


# Topic: Colour Spaces
def demo_colour_spaces() -> None:
    base_image = cv2.imread("computer-vision-two/assets/fruit.jpg")
    small_image = cv2.resize(base_image.copy(),(0,0), fx=0.5, fy=0.5)
    # cv2.rectangle(BGR, (0,0),(image.shape[1],50), (0,0,0), -1)
    # cv2.putText(BGR, "Colourspace: BGR", (10,35), font, 1, text_col)
    BGR = draw_titlebox(small_image.copy(), "Colour space: BGR")
    RGB = cv2.cvtColor(small_image.copy(), cv2.COLOR_BGR2RGB)
    RGB = draw_titlebox(RGB, "Colour space: RGB")
    GRAY = cv2.cvtColor(small_image.copy(), cv2.COLOR_BGR2GRAY)
    GRAY = draw_titlebox(GRAY, "Colour space: Grayscale")
    # Grayscale is 1 dimension, just for the sake of clearer visuals I will make it 3D
    GRAY = cv2.merge((GRAY,GRAY,GRAY))
    HSV = cv2.cvtColor(small_image.copy(), cv2.COLOR_BGR2HSV)
    HSV = draw_titlebox(HSV, "Colour space: HSV")

    # Combine into a final comparison image
    row_one = np.concatenate((BGR,RGB), axis=0)
    row_two = np.concatenate((GRAY, HSV), axis=0)
    square = np.concatenate((row_one, row_two), axis=1)

    # Display the comparison
    display_image(square, "Colour Spaces Comparison")
    return

# Topic: Annotations
def demo_annotations() -> None:
    base_image = cv2.imread("computer-vision-two/assets/gamer.jpg")
    small_image = cv2.resize(base_image.copy(),(0,0), fx=0.5, fy=0.5)
    # TODO: Continue with demos, concepts are in notebook



    return
    





if __name__ == "__main__":
    # demo_colour_spaces()
    pass
