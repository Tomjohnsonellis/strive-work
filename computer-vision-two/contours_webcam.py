import cv2

webcam_video = cv2.VideoCapture(0)


def change_max_edge(trackbar_value):
    global max_edge
    max_edge = trackbar_value
    return


def change_min_edge(trackbar_value):
    global min_edge
    min_edge = trackbar_value
    return


min_edge = 0
max_edge = 255
edge_window_name = "Edges"
cv2.namedWindow(edge_window_name)
cv2.createTrackbar("Max Edge", edge_window_name, 0, 255, change_max_edge)
cv2.createTrackbar("Min Edge", edge_window_name, 0, 255, change_min_edge)

# One frame captured due to low performance laptop
data_captured, frame = webcam_video.read()
frame = cv2.GaussianBlur(frame, (7,7), 1)
while (True):
    # Feel free to uncomment for a live webcam feed
    # data_captured, frame = webcam_video.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold then show image
    canny_image = cv2.Canny(frame, min_edge, max_edge)
    cv2.imshow(edge_window_name, canny_image)

    if cv2.waitKey(30) == 27:  # ESC Key to close
        break
webcam_video.release()
cv2.destroyAllWindows()
