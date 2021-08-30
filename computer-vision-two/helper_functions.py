import cv2


def display_image(image_array, image_name="My Image"):
    cv2.imshow(image_name, image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)