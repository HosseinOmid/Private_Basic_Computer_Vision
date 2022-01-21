import cv2 as cv

markersX = 2
markersY = 2
markerLength = 0.0365  # length of marker in mm in real world
markerSeparation = 0.1165  # markerLength/2
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
board = cv.aruco.GridBoard_create(
    markersX=markersX,
    markersY=markersY,
    markerLength=markerLength,
    markerSeparation=markerSeparation,
    dictionary=aruco_dict)
if True:
    # draw the board
    img = board.draw((500, 500))
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    img = cv.copyMakeBorder(
        img, 10, 10, 10, 10, cv.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    # detect marker
    arucoParams = cv.aruco.DetectorParameters_create()
    corners, ids, rejected = cv.aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)
    # verify that *at least* one ArUco marker was detected
    if len(corners) > 0:
        cv.aruco.drawDetectedMarkers(img, corners, ids)
    cv.imshow('Marker Board', img)
