# kurze Einfuehrung in OpenCV
# siehe: https://www.youtube.com/watch?v=oXlwWbU8l2o

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#img =cv.imread('messi5.jpg', 0)
#testSnap = cv.imread('t2_Depth.png')
#cv.imshow('t2', testSnap[:,:,1])

# Bilder oeffnen und anzeigen
img = cv.imread('lotus320.jpg')
cv.imshow('flower', img)

# Edge detection
edg = cv.Canny(img, 100, 100)
cv.imshow('edges', edg)

# Konvertierung in Grayscale
gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
cv.imshow('gray', gray_img)

# filtern und neu skalieren
bin_img = np.where(gray_img<100, 0, gray_img)
cv.imshow('bin_img', bin_img)
# neu skalieren
#????

gray_edg = cv.Canny(gray_img, 100, 125)
cv.imshow('gray_edg', gray_edg)

# blur
gray_img_blured = cv.GaussianBlur(gray_img, (3, 3), cv.BORDER_CONSTANT)
gray_edg_blured = cv.Canny(gray_img_blured, 100, 100)
cv.imshow('gray_edg_blured', gray_edg_blured)

# colorized_img = cv.cvtColor(gray_img, cv.COLOR_GRAY2RGB)
# cv.imshow('colorized_img', colorized_img)

# Resize
#resized = cv.resize(img, (500, 500), interpolation=cv.INTER_LINEAR)
#cv.imshow('resized', resized)

# contour detection (consider contours are not the same as edges)
contours, hierarchy = cv.findContours(gray_edg, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# CHAIN_APPROX_SIMPLE: gibt nur die beiden Endpunkten einer Gerade aus, anstatt die einzelnen Punkten entlang der Gerade
print(len(contours), ' contours were found!')
contour_img = img*0
mask = cv.drawContours(contour_img, contours, -1, (0, 0, 255), 2)
cv.imshow('contours', contour_img)

# Threshold
# https://docs.opencv.org/4.5.2/d7/d4d/tutorial_py_thresholding.html
cv.threshold(gray_img, 125, 255, cv.THRESH_BINARY)
# Gray image: 0 ist Schwarz, 255 ist Weiß
# THRESH_BINARY: entweder schwarz oder weiß
# THRESH_TRUNC: alle zu helle Stellen (große Grauwerte) werden zu einem Maximalwert gesetzt
# THRESH_TOZERO: alle zu dunklen Stellen werden schwarz
ret, thresh3 = cv.threshold(gray_img, 50, 255, cv.THRESH_TRUNC)
cv.imshow('TRUNC', thresh3)
ret, thresh4 = cv.threshold(gray_img, 127, 255, cv.THRESH_TOZERO)
# adaptive Thresholding
# ...


# Color Sapces
# BGR is the default open cv format
# cv.COLOR_GRAY2BGR: will change single channel image to 3-channel image.
# When converting color image to gray image, all color information will be gone and not recoverable!

# smoothing and bluring
# Gaussian
# Average
# Median Blur
median = cv.medianBlur(img, 5)
cv.imshow('median Blur', median)

# Bilateral: aehnlcih zu Gaussian. heißt bilateral, weil nicht nur Pixelwerte, aber auch Nachbarschaften werden berücksichtigt.
# https://en.wikipedia.org/wiki/Bilateral_filter
# eignet sich sehr gut zur Nois-Reduktion
bil = cv.bilateralFilter(img, 5, 15, 15)
cv.imshow('bilateral filter', bil)

# Masking
blank = gray_img*0
myMask = cv.circle(blank, (img.shape[0]//2, img.shape[1]//2), 100, 255, -1)
myMasked = cv.bitwise_and(img, img, mask=myMask)
cv.imshow('myMasked', myMasked)

# Histogram
gray_hist = cv.calcHist([gray_img], [0], None, [120], [0, 256])
plt.figure()
plt.title('gray histogram')
plt.xlabel('bins')
plt.ylabel('# of pixels')
#plt.xlim()
plt.plot(gray_hist)
plt.show()

#


cv.waitKey(0)