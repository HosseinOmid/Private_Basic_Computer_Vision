# Skript zur Tiefenberechnung anhand eines Stereo-Bildes (zweier Bilder)
# https://docs.opencv.org/4.5.2/dd/d53/tutorial_py_depthmap.htmlimport numpy as np

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

imgL = cv.imread('tsukuba_l.png', 0)
imgR = cv.imread('tsukuba_r.png', 0)
stereo = cv.StereoBM_create(numDisparities=32, blockSize=5)
disparity = stereo.compute(imgL, imgR)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(disparity, 'gray')

gray_hist = cv.calcHist([np.float32(disparity)], [0], None, [120], [-20, 256])
plt.subplot(1, 2, 2)
plt.title('gray histogram')
plt.xlabel('bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.show()