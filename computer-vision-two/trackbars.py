import cv2



def change_opacity(trackbar_value):
    global display_image
    alpha = max_alpha * (trackbar_value/100)
    beta = 1.0 - alpha
    display_image = cv2.addWeighted(bus, alpha, stars, beta, 0)
    cv2.imshow(blend_window_name, display_image)

def change_red(trackbar_value):
    global display_image
    # display_image[2][display_image[2] < trackbar_value] = trackbar_value
    b,g,r = cv2.split(display_image)
    r[b < 50] = trackbar_value
    red_altered = cv2.merge((b,g,r))
    cv2.imshow(blend_window_name, red_altered)
    


# We'll make a trackbar that changes opacity so we can blend 2 images

image_size = (500,500)
bus = cv2.imread('computer-vision-two/assets/bus.jpg')
bus = cv2.resize(bus, image_size, interpolation=cv2.INTER_LINEAR)
stars = cv2.imread('computer-vision-two/assets/stars.jpg')
stars = cv2.resize(stars, image_size, interpolation=cv2.INTER_LINEAR)
print(bus.shape)
print(stars.shape)
global display_image
display_image = bus

blend_window_name = "Blended Image"
cv2.namedWindow(blend_window_name)
cv2.imshow(blend_window_name, display_image)

trackbar_name = "opacity_slider"
max_alpha = 1
cv2.createTrackbar(trackbar_name, blend_window_name, 0, 100, change_opacity)
cv2.createTrackbar("remove_red", blend_window_name, 0, 255, change_red)

cv2.waitKey(0)
cv2.destroyAllWindows()