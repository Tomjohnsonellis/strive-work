import cv2
import numpy as np
from helper_functions import display_colour_histograms, display_image, scale_image

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
    
    # In case this needs resizing...
    annotated_image = cv2.resize(base_image.copy(),(0,0), fx=1, fy=1)
    display_image(annotated_image)

    # Various primitive shapes can be drawn on top of images with openCV
    # https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
    cv2.ellipse(annotated_image, (200,200), (100,50), 0, 0, 360, (100,64,255), -1)
    display_image(annotated_image)
    
    # https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
    cv2.circle(annotated_image, (250,200), 25, (0,255,0), 10)
    cv2.circle(annotated_image, (250,200), 10, (0,0,0), -1)
    cv2.circle(annotated_image, (150,200), 25, (0,255,0), 10)
    cv2.circle(annotated_image, (150,200), 10, (0,0,0), -1)
    display_image(annotated_image)
    
    # https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    cv2.line(annotated_image, (100, 125), (300, 125), (0,0,0), 5)
    display_image(annotated_image)

    # https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#gaa3c25f9fb764b6bef791bf034f6e26f5
    polygon_points = np.array([[600,400], [500,200], [575,210], [600,175],[625, 210], [700,200]], np.int32)
    polygon_points.reshape((-1,1,2))
    cv2.polylines(annotated_image, [polygon_points], True, (255,0,0), thickness=5)
    display_image(annotated_image)

    # You can even flip the image
    overlay = annotated_image.copy()
    overlay = cv2.flip(overlay, 1)
    display_image(overlay)

    # And overlay transparant images
    opacity = 0.8
    beta = 1 - opacity
    annotated_image = cv2.addWeighted(annotated_image, opacity, overlay, beta, gamma=0 )

    display_image(annotated_image)
    return
    
def demo_histograms() -> None:
    base_image = cv2.imread("computer-vision-two/assets/bus.jpg")
    resized_image = cv2.resize(base_image.copy(),(0,0), fx=0.5, fy=0.5)
    # Consider this image focusing on a big red bus
    display_image(resized_image)
    # We can use histograms to see how the colours make up an image
    display_colour_histograms(base_image)
    # Also useful for different colour spaces, HSV is typically more useful
    display_colour_histograms(base_image, cvt_to_hsv=True)

def demo_affine_transforms() -> None:
    image = cv2.imread('computer-vision-two/assets/meme.jpg')
    original_size = image.shape
    x = original_size[0]//4
    y = original_size[1]//2
    # Affine transformations are the result of some matrix multiplication
    translation_matrix =np.float32([
        [1, 0, x],
        [0, 1, y]
        ])

    default_matrix = np.float32([
        [0,0],
        [1,0],
        [0,1]
    ])

    altered_matrix = np.float32([
        [0.3, 0],
        [1, 0],
        [0, 1.1]
    ])

    scale_matrix = np.float32([
        [0, 0],
        [0.3, 0],
        [0, 0.3]
    ])
    display_image(image)
    # Scale down
    scale = cv2.getAffineTransform(default_matrix, scale_matrix)    
    image = cv2.warpAffine(image, scale, (image.shape[1], image.shape[0]))
    display_image(image)
    
    # Translation to center
    image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    display_image(image)
    
    # Whatever transform happens with the altered_matrix
    transformation = cv2.getAffineTransform(default_matrix, altered_matrix)    
    image = cv2.warpAffine(image, transformation, (image.shape[1], image.shape[0]))
    display_image(image)

    # Translation to center again
    translation_matrix =np.float32([
        [1, 0, 250],
        [0, 1, 0]
        ])
    image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    display_image(image)

    # Finally, let's do a rotation
    center = (image.shape[0]/2, image.shape[1]/2)
    angle = 45
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    display_image(image)
    return



if __name__ == "__main__":
    # demo_colour_spaces()
    # demo_annotations()
    # demo_histograms()
    # demo_affine_transforms()
    pass
