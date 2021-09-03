import cv2
from helper_functions import display_image
import numpy as np

image = cv2.imread("computer-vision-two/assets/paper.jpg")
# print(type(image))

# original_corners = np.float32( [] )
# destination_corners = np.float( [] )
# M = cv2.getperspectivematrix
# fixed_perspective = cv2.transform(m)


def change_perspective(points:list, image:np.array=image) -> np.array:
    starting_points = np.float32(points)
    print(starting_points)
    print("~"*50)
    desired_points = np.float32([ [0,0], [1000,0], [1000,1000], [0,1000] ])

    print(desired_points)
    M = cv2.getPerspectiveTransform(starting_points, desired_points)
    print("X"*50)

    # Find shape of original area, use it to make the output image prettier
    # x_size = int(abs(starting_points[0][1] - starting_points[2][1]))
    x_size = image.shape[1]
    # y_size = int(abs(starting_points[0][0] - starting_points[2][0]))
    y_size = image.shape[0]
    changed_perspective = cv2.warpPerspective(image, M, (x_size,y_size))
    return changed_perspective

    




def mouse_actions(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        global points_clicked
        cv2.circle(image, (x,y), 25, (0,0,255), 10)
        cv2.imshow(window_name, image)
        # points_clicked.append((x,y), axis=0)
        coords = [x,y]
        # coords = np.array([x,y])
        points_clicked.append(coords)
        # points_clicked = np.append(points_clicked, coords, axis=1)
        # points_clicked = np.array([points_clicked, coords])
        # print(points_clicked)
        print(len(points_clicked))
        # print(points_clicked.shape[0])
        if len(points_clicked)== 4:
            cv2.imshow("Wow", change_perspective(points_clicked))
            
            

points_clicked = []
print(points_clicked)

window_name = "I am an image"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_actions)
cv2.imshow(window_name, image)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)