import cv2
import numpy as np


current_space: "string" = "BGR"
desired_space: "string" = "BGR"




def change_to_desired_space(image:"Webcam feed", desired_space:"BGR/GRAY/HSV/RGB"=None) -> "Altered webcam feed":
    if desired_space == "BGR":
        return image
    if desired_space == "GRAY":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    if desired_space == "HSV":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image
    if desired_space == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
        
    
# print(__annotations__)
# print(change_to_desired_space.__annotations__)
webcam_video = cv2.VideoCapture(0)
while(True):
    data_captured, frame = webcam_video.read()
    key = cv2.waitKey(100)

    # I found that the lower case key is all that's needed for this,
    # doing "key == lowercase or uppercase" caused conflicts with the arrow keys
    if key == ord('b'):
        desired_space = "BGR"

    if key == ord('g'):
        desired_space = "GRAY"

    if key == ord('h'):
        desired_space = "HSV"
    
    if key == ord('r'):
        desired_space = "RGB"


    frame = change_to_desired_space(frame, desired_space)
    cv2.putText(frame, f"Current colour space: {desired_space}",(25,50), 0, 1, (0,255,0),2 ,1)
    cv2.putText(frame, f"Key: {key}",(25,125), 0, 2, (0,0,255),2 ,1)
    if key == 32:
        cv2.putText(frame, "~SPACE~",(150,450), 0, 3, (255,0,255),2 ,1)
    
    if key == 27:
        break
    cv2.imshow("Hello", frame)

webcam_video.release()
cv2.destroyAllWindows()
