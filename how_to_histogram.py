# wie werden histogramme in Python erstellt?
# https://docs.opencv.org/4.5.2/de/db2/tutorial_py_table_of_contents_histograms.html

# Option 1: mit matplotlib
# Option 2: mit OpenCV


import numpy as np
import matplotlib.pyplot as plt
import cv2


test = np.array([1, 19, 5, 1, 5, 20, 15, 25, 99, 50, 45, 55, 60, 58, 55, 99, 55, 55, 55, 55])

plt.subplot(211)
hist_data = plt.hist(test, bins=11)

hist_cv = cv2.calcHist([np.float32(test)], [0], None, [11], [0, int(test.max()+1)])

plt.subplot(212), plt.plot(hist_cv)
plt.show()

img = cv2.imread('lotus320.jpg')
color = ('b', 'g', 'r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

# Histogram Equalization
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(gray)
cv2.imshow('gray', gray)
cv2.imshow('equ', equ)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(gray)
cv2.imshow('clahe', cl1)

cv2.waitKey(0)