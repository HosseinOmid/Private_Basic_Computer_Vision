# Skript zur Segmmentierung eines Graubildes mit dem Watersheld Algorithmus
# https://docs.opencv.org/4.5.2/d3/db4/tutorial_py_watershed.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('coins.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow('thresh', thresh)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
cv.imshow('opening', opening)

# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
# a distance transform applied to an image will generate an output image whose pixel values will be the closest
# distance to a zero-valued pixel in the input image. Basically, they will have the closest distance to the background
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

cv.imshow('marker', np.float32(markers))

markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
cv.imshow('markers', np.float32(markers))

cv.waitKey(0)