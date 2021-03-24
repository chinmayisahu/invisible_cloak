# https://learnopencv.com/invisibility-cloak-using-color-detection-and-segmentation-with-opencv

import cv2
import numpy as np
import time
import imutils

cap = cv2.VideoCapture(0)
time.sleep(3)
background=0

width = 1200

for i in range(30):
    ret,background = cap.read()
    background = imutils.resize(background, width=width)

background = np.flip(background,axis=1)

while(cap.isOpened()):

    ret, img = cap.read()
    img = imutils.resize(img, width=width)
    
    

    # Flip the image 
    img = np.flip(img, axis = 1)

    # Convert the image to HSV color space.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (35, 35), 0)

    # We use the range 0-10 and 170-180 to avoid detection of skin as red. 
    # High range of 120-255 for saturation is used because our cloth should be of highly saturated red color. 
    # The lower range of value is 70 so that we can detect red color in the wrinkles of the cloth as well.

    # Defining lower range for red color detection.
    low = 25 #0: Red
    high = 102 #170: Red
    threshold = 10

    l_s = 52 #120: Red
    l_v = 72 #70: Red
    h_s = 255 
    h_v = 255


    lower = np.array([low,l_s,l_v])
    upper = np.array([low + threshold,255,255])
    mask1 = cv2.inRange(hsv, lower, upper)

    # Defining upper range for red color detection
    lower_red = np.array([high,l_s,l_v])
    upper_red = np.array([high + threshold,255,255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask1 + mask2

    # Addition of the two masks to generate the final mask.
    mask = mask1 + mask2
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # Replacing pixels corresponding to cloak with the background pixels.
    img[np.where(mask == 255)] = background[np.where(mask == 255)]
    cv2.imshow('Display',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# releasing web-cam
cap.release()
# Destroying output window
cv2.destroyAllWindows()