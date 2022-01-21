

import cv2 as cv
import numpy as np


img = cv.imread('person.jpg')
cv.imshow('Person', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blured = cv.GaussianBlur(gray, (9, 9), cv.BORDER_CONSTANT)
cv.imshow('blured', blured)

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_rect = haar_cascade.detectMultiScale(blured, scaleFactor=1.1, minNeighbors=3)

print('number of detected perosn is: ', len(faces_rect))

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+h, y+h), (0, 255, 0), thickness=3)
cv.imshow('detected faces', img)

cv.waitKey(0)