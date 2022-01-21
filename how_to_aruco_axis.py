
# https://stackoverflow.com/questions/46363618/aruco-markers-with-opencv-get-the-3d-corner-coordinates?rq=1
# https://github.com/kyle-bersani/opencv-examples/blob/master/CalibrationByArucoGridboard/CalibrateCamera.py
import numpy as np
import cv2 as cv
import cv2.aruco as aruco
import pickle

infile = open('calibration.pckl', 'rb')
new_dict = pickle.load(infile)
infile.close()

# If the vector is NULL/empty, the zero distortion coefficients are assumed.
# https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
distCoeffs = None

cameraMatrix = np.array([[715,  0., 234],
                [0., 705, 312],
                [0., 0., 1.]])

# Creating a theoretical board we'll use to calculate marker positions
markersX = 2
markersY = 2
markerLength = 140
convMarkerLength = 50 # lets say marker has 50 mm in real world
markerSeparation = 440
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.GridBoard_create(
    markersX=markersX,
    markersY=markersY,
    markerLength=markerLength,
    markerSeparation=markerSeparation,
    dictionary=aruco_dict)

margins = 20
img_width = int(markersX * (markerLength + markerSeparation) - markerSeparation + 2 * margins)
img_height = int(markersY * (markerLength + markerSeparation) - markerSeparation + 2 * margins)

img = np.zeros((img_height, img_width, 1), dtype="uint8")
aruco.drawPlanarBoard(board, outSize=(img_width, img_height), img=img, marginSize=margins, borderBits=1)
# cv.imshow('Webcam', img)
# cv.imwrite('myMarkerBoard.png', img)
# cv.waitKey()

# Arrays to store object points and image points from all the images.
objpoints = []  #
all_ids = []  #
all_corners = []  #

# Read an image or a video to calibrate your camera
# I'm using a video and waiting until my entire gridboard is seen before calibrating
# The following code assumes you have a 5X7 Aruco gridboard to calibrate with
cam = cv.VideoCapture(0)

while cam.isOpened():
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()
    if ret:
        # grayscale image
        gray = cv.cvtColor(QueryImg, cv.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()

        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if len(corners) > 0:
            cv.aruco.drawDetectedMarkers(QueryImg, corners, ids)

            # Make sure markers were detected before continuing
            if ids is not None and corners is not None and len(ids) > 3 and len(corners) > 3 and len(corners) == len(ids):
                # The next if makes sure we see all matrixes in our gridboard
                if len(ids) == len(board.ids):
                    # save every single frame to calibrate the camera in one shot
                    all_ids.append(ids)
                    all_corners.append(corners[0])
                    all_corners.append(corners[1])
                    all_corners.append(corners[2])
                    all_corners.append(corners[3])
                    objpoints.append(board.objPoints[0])
                    objpoints.append(board.objPoints[1])
                    objpoints.append(board.objPoints[2])
                    objpoints.append(board.objPoints[3])

            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, markerLength=convMarkerLength,
                                                              cameraMatrix=cameraMatrix, distCoeffs=None)
            # markerLength: size of the marker side in meters or in any other unit
            for i in range(len(rvecs)):
                rvec = rvecs[i]
                tvec = tvecs[i]
                aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs=None, rvec=rvec, tvec=tvec, length=convMarkerLength/2)

        cv.imshow('Webcam', QueryImg)

    # Exit at the end of the video on the EOF key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv.destroyAllWindows()


if all_ids and False:
    # Calibrate the camera now using cv2 method
    ret, cameraMatrix, distCoeffs, _, _ = cv.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=all_corners,
        imageSize=gray.shape, #[::-1], # may instead want to use gray.size
        cameraMatrix=None,
        distCoeffs=None)

    # Calibrate camera now using Aruco method
    # https://docs.opencv.org/3.4/da/d13/tutorial_aruco_calibration.html
    ret, cameraMatrix, distCoeffs, _, _ = aruco.calibrateCameraAruco(
        corners=all_corners,
        ids=all_ids,
        counter=markersX*markersY,
        board=board,
        imageSize=gray.shape[::-1],
        cameraMatrix=None,
        distCoeffs=None)

    # Print matrix and distortion coefficient to the console
    print(cameraMatrix)
    print(distCoeffs)

    # Output values to be used where matrix+dist is required
    f = open('calibration.pckl', 'wb')
    pickle.dump((cameraMatrix, distCoeffs), f)
    f.close()

    # Print to console our success
    print('Calibration successful.')


