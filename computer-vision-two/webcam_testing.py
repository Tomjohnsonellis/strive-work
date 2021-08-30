import cv2

webcam_video = cv2.VideoCapture(0)

# text_to_display = "^^^TOM^^^"
ft = cv2.freetype.createFreeType2()
ft.loadFontData(fontFileName="computer-vision-two/assets/impact.ttf", id=0)
# font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret, frame = webcam_video.read()
    # print(ret)
    

    cv2.putText(frame, 
    "^^^TOM^^^", 
    (125,450),
    0,
    2,
    (0,255,0),
    4
    )

    cv2.putText(frame, 
    "v v v TOM v v v", 
    (50,50),
    0,
    2,
    (0,255,0),
    4
    )

    # cv2.ellipse(frame,
    # (300,50),
    # )
    
    
    # cv2.putText(frame, 
    # ">>>TOM>>>", 
    # (50,225),
    # 0,
    # 2,
    # (0,255,0)
    # )

    # cv2.putText(frame, 
    # "<<<TOM<<<", 
    # (300,225),
    # 0,
    # 2,
    # (0,255,0)
    # )

    
    cv2.imshow("It's a me", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam_video.release()
cv2.destroyAllWindows()
    