# https://www.youtube.com/watch?v=v5a7pKSOJd8

import cv2 as cv

cap = cv.VideoCapture(0)

arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
arucoParams = cv.aruco.DetectorParameters_create()
#cv.aruco.getPredefinedDictionary()
while True:
    success, img = cap.read()
    #img = cv.imread('singlemarkersoriginal.jpg')
    corners, ids, rejected = cv.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        cv.aruco.drawDetectedMarkers(img, corners, ids)
        # def trashCode:
        #     # flatten the ArUco IDs list
        #     ids = ids.flatten()
        #     # loop over the detected ArUCo corners
        #     for (markerCorner, markerID) in zip(corners, ids):
        #         # extract the marker corners (which are always returned in
        #         # top-left, top-right, bottom-right, and bottom-left order)
        #         corners = markerCorner.reshape((4, 2))
        #         (topLeft, topRight, bottomRight, bottomLeft) = corners
        #         # convert each of the (x, y)-coordinate pairs to integers
        #         topRight = (int(topRight[0]), int(topRight[1]))
        #         bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        #         bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        #         topLeft = (int(topLeft[0]), int(topLeft[1]))
        #         # draw the bounding box of the ArUCo detection
        #         cv.line(img, topLeft, topRight, (0, 255, 0), 2)
        #         cv.line(img, topRight, bottomRight, (0, 255, 0), 2)
        #         cv.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
        #         cv.line(img, bottomLeft, topLeft, (0, 255, 0), 2)

    cv.imshow('Webcam', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
