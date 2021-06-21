import numpy as np
import cv2


scene = cv2.imread("img/greenscreen.png")
cv2.imshow("Image", scene)



cv2.waitKey(0)
cv2.destroyAllWindows()