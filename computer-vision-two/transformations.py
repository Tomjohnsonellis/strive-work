import cv2
import numpy as np

webcam_video = cv2.VideoCapture(0)
while(True):
    data_captured, frame = webcam_video.read()
    frame = cv2.imread('computer-vision-two/assets/meme.jpg')
    original_size = frame.shape
    x = original_size[0]//4
    y = original_size[1]//2
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
    # Scale down
    scale = cv2.getAffineTransform(default_matrix, scale_matrix)    
    warped_frame = cv2.warpAffine(frame, scale, (frame.shape[1], frame.shape[0]))
    # Translation to center
    warped_frame = cv2.warpAffine(warped_frame, translation_matrix, (frame.shape[1], frame.shape[0]))
    # Whatever transform happens with the altered_matrix
    transformation = cv2.getAffineTransform(default_matrix, altered_matrix)    
    warped_frame = cv2.warpAffine(warped_frame, transformation, (frame.shape[1], frame.shape[0]))

    # Translation to center again
    translation_matrix =np.float32([
        [1, 0, 250],
        [0, 1, 0]
        ])
    warped_frame = cv2.warpAffine(warped_frame, translation_matrix, (frame.shape[1], frame.shape[0]))

    # Finally, let's do a rotation
    center = (frame.shape[0]/2, frame.shape[1]/2)
    angle = 70
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    warped_frame = cv2.warpAffine(warped_frame, rotation_matrix, (frame.shape[1], frame.shape[0]))


    cv2.imshow("Result",warped_frame)
    

    if cv2.waitKey(30) == ord('q'):
        break
webcam_video.release()
cv2.destroyAllWindows()
