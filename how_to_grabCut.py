# https: // docs.opencv.org / 3.4 / d8 / d83 / tutorial_py_grabcut.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('messi5.jpg')
# mask - It is a mask image where we specify which areas are background, foreground or probable background/foreground
# etc. It is done by the following flags, cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD, or simply pass 0,1,2,
# 3 to image.
mask = np.zeros(img.shape[:2], np.uint8)

# bdgModel, fgdModel - These are arrays used by the algorithm internally.
# You just create two np.float64 type zero arrays of size (1,65).
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# rect - It is the coordinates of a rectangle which includes the foreground object in the format (x,y,w,h)
rect = (50, 50, 480, 340)

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img_fg = img*mask2[:, :, np.newaxis]
img_with_rect = img
cv.rectangle(img_with_rect, rect[:2], rect[-2:], (255, 0, 0), 2)

cv.imshow('img_with_rect', img_with_rect)
cv.imshow('img_fg', img_fg)

cv.waitKey(0)

#plt.subplot(212)
#plt.imshow(img_fg), plt.show()



def trash_code():
    # newmask is the mask image I manually labelled
    newmask = cv.imread('newmask.png',0)
    # wherever it is marked white (sure foreground), change mask=1
    # wherever it is marked black (sure background), change mask=0
    mask[newmask == 0] = 0
    mask[newmask == 255] = 1
    mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask[:,:,np.newaxis]
    plt.imshow(img),plt.colorbar(),plt.show()
    return None