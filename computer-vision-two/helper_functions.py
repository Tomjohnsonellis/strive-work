import cv2


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