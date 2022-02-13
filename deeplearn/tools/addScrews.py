import cv2
import numpy as np

source = cv2.imread('source.jpg')
screw = cv2.imread('img.png')
screw = cv2.resize(screw, (100, 100))
# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = screw.shape
roi = source[0:rows, 0:cols]
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(screw, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(screw, screw, mask=mask_inv)
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg, img2_fg)
# source[0:rows, 0:cols] = dst
source[100:rows+100, 100:cols+100] = dst
cv2.imshow('res', source)
cv2.waitKey(0)
cv2.destroyAllWindows()
