import cv2

subtractor = cv2.createBackgroundSubtractorMOG2(
    history=50,
    varThreshold=1000,
    detectShadows=False
)

capture = cv2.VideoCapture('computer-vision-two/assets/skateboard.mp4')
k = 0

while True:
    ret, frame = capture.read()
    if ret:
        frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
        foreground_mask = subtractor.apply(frame)
        cv2.imshow("egg", foreground_mask)

        

    if cv2.waitKey(30) == 27: # ESC Key to close
        break
capture.release()
cv2.destroyAllWindows()


