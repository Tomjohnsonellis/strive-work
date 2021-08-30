from helper_functions import display_image, annotate_quadrangle
import cv2
from cv2 import QRCodeDetector

print(cv2.__version__)
# First thing to do is load an image containing a qr code
image_with_qr = cv2.imread('computer-vision-two/assets/qr.jpg')
# This image is a bit big so let's resize it
image_with_qr = cv2.resize(image_with_qr, (950, 600))


# Then we can try to detect it, this returns a "True/False" 
# and coordinates that make a square containing the QR code
data_found, points = QRCodeDetector().detect(image_with_qr)
if data_found:
    print("QR Found!")
else:
    print("No QR detected, exiting...")
    exit()
print("="*50)
# points is actually a 1x4x2 array, we just need the 4x2 info
#print(points.shape)
points = points[0].astype(int)
print(points)

QR_box = annotate_quadrangle(image_with_qr, points)

display_image(QR_box)

# Now for decoding the QR as well
qr_found, points, corrected_qr = QRCodeDetector().detectAndDecode(image_with_qr)
if qr_found:
    print("QR Detected!")
    print(f"It says: {qr_found}")
    print("Saving and displaying detected code...")
    cv2.imwrite("computer-vision-two/assets/corrected_qr.jpg", corrected_qr)
    display_image(corrected_qr)
else:
    print("No QR detected, exiting...")
    exit()



