"""
Programm zur Messung und Erstellung einer Punktwolke anhand einer RealSense Kamera
Autor: Hossein Omid Beiki
Stand: 2021.07.26

# how to make a 3D point cloud from multiple RGB-D images?
# https://dsp.stackexchange.com/questions/70250/making-a-3d-point-cloud-from-multiple-rgb-d-images
# https://dsp.stackexchange.com/questions/54124/multiple-image-dense-point-cloud-reconstruction-with-camera-extrinsics-intrinsic?rq=1
# https://www.youtube.com/watch?v=w1OsTGySaKM
"""

import copy
import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
import open3d as o3d
# from sklearn import preprocessing
import pyrealsense2 as rs


def main():
    # Creating a theoretical board we'll use to calculate marker positions
    markersX = 2
    markersY = 2
    markerLength = 140
    convMarkerLength = 50  # lets say marker has 50 mm in real world
    markerSeparation = 440
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    board = cv.aruco.GridBoard_create(
        markersX=markersX,
        markersY=markersY,
        markerLength=markerLength,
        markerSeparation=markerSeparation,
        dictionary=aruco_dict)
    # read from already saved data or read now from the camera?
    readFromData = False
    # some initializations
    fx = fy = .5
    o3d_intrinsic = None
    start_time = time.time()
    if readFromData:
        # load and show the input images
        with open('color_image.npy', 'rb') as f:
            color_image = np.load(f)
            # color_image = cv.cvtColor(color_image1, cv.COLOR_BGR2GRAY)
        with open('depth_image.npy', 'rb') as f:
            depth_image = np.load(f)
    else:
        pipeline, align, rs_temporal_filter = init_and_start_realsense()
        # initialize an empty pointcload which will store the final pointcload
        merged_pointcload = o3d.geometry.PointCloud()
        try:
            for i in range(5):
                while True:
                    color_image, depth_image, color_frame, depth_frame, o3d_intrinsic, cameraMatrix = \
                        read_from_realsense(pipeline, align, rs_temporal_filter)
                    # apply marker detection
                    aruco_image = np.copy(color_image)
                    depth_filtered, marker_centers, marker_depths, rotMats, tvecs = \
                        find_aruco_and_filter_depth(aruco_image, depth_image, cameraMatrix, convMarkerLength)

                    # depth_filtered_clipped = clip_image_with_min_max(depth_filtered) # TOO SLOW
                    # resize_and_show(fx, fy, depth_filtered=depth_filtered)
                    resize_and_show(fx, fy, aruco_image=aruco_image)

                    if cv.waitKey(1) & 0xFF == ord('s'):
                        break  # get out of the while loop
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        raise Exception('Quit', 'User Quit')
                pointcload = pointcload_with_open3d(color_image, depth_filtered, marker_centers, marker_depths, rotMats, tvecs,
                                                    visualize=True,
                                                    intrinsic=o3d_intrinsic)
                merged_pointcload = merged_pointcload + pointcload
                o3d.visualization.draw_geometries([merged_pointcload])
        except Exception as inst:
            print(inst)
        finally:
            stop_realsense(pipeline)

    end_time = time.time()
    print(f'Data read successfully in {end_time-start_time} seconds')

    """resize_and_show(fx, fy, color_image=color_image)
    resize_and_show(fx, fy, depth_image=depth_image)
    # clip and show the region of interest from the depth map
    depth_clipped = clip_and_show(depth_image, alpha=0.01)
    resize_and_show(fx, fy, depth_clipped=depth_clipped)
    # filter the images
    color_image, _ = filter_and_show(color_image, depth_clipped)

    # apply marker detection
    aruco_image = np.copy(color_image)
    depth_filtered, marker_centers, marker_depths = find_aruco_and_filter_depth(aruco_image, depth_image)

    depth_filtered_clipped = clip_image_with_min_max(depth_filtered)
    resize_and_show(fx, fy, depth_filtered=depth_filtered)

    pointcload = pointcload_with_open3d(color_image, depth_filtered, marker_centers, marker_depths, visualize=True,
                           intrinsic=o3d_intrinsic)


    cv.waitKey()"""


def init_and_start_realsense():
    # Create a pipeline
    pipeline = rs.pipeline()
    # Create a config and configure the pipeline to stream
    # different resolutions of color and depth streams
    config = rs.config()
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    # config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    # config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 2  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    # Filters
    filters = [rs.disparity_transform(),
               rs.spatial_filter(),
               rs.temporal_filter(),
               rs.disparity_transform(False)]
    temporal = rs.temporal_filter()
    print("pipeline started")
    return pipeline, align, temporal


def read_from_realsense(pipeline, align, temporal, autoexposureFrames=5):
    try:
        # Get frameset of color and depth
        for i in range(autoexposureFrames):
            frames = pipeline.wait_for_frames()

        """
        apply temporal frame 
        The temporal filter is intended to improve the depth data persistency by manipulating per-pixel values
        based on previous frames
        https://github.com/IntelRealSense/librealsense/blob/master/doc/post-processing-filters.md#temporal-filter
        how to apply:
        https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb        
        """
        temporal_depth_frames = []
        for i in range(autoexposureFrames):
            frameset = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = align.process(frameset)
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                raise Exception('not aligned', 'the depth image could not be aligned to the rgb image')

            temporal_depth_frames.append(aligned_depth_frame)

        for i in range(autoexposureFrames):
            depth_frame_temp_filtered = temporal.process(temporal_depth_frames[i])

        print("got the frame!")

        # get data
        depth_img_filtered = np.asanyarray(depth_frame_temp_filtered.get_data())
        depth_img = np.asanyarray(aligned_depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())

        # get camera intrinsics in order to calculate the point-cload regard to that
        depth_intrinsics = rs.video_stream_profile(aligned_depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height
        fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
        px, py = depth_intrinsics.ppx, depth_intrinsics.ppy
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, px, py)

        cameraMatrix = np.array([[fx,  0., px],
                [0., fy, py],
                [0., 0., 1.]])

        return color_img, depth_img_filtered, color_frame, aligned_depth_frame, o3d_intrinsic, cameraMatrix
    except:
        raise Exception('Pipeline', 'pipeline stopped due to an error')


def stop_realsense(pipeline):
    pipeline.stop()
    print("pipeline stopped")


def find_aruco_and_filter_depth(aruco_img, depth_img, cameraMatrix, convMarkerLength):
    fx = fy = .5
    arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
    arucoParams = cv.aruco.DetectorParameters_create()
    corners, ids, rejected = cv.aruco.detectMarkers(aruco_img, arucoDict, parameters=arucoParams)
    # cv.aruco.estimatePoseSingleMarkers(corners[0], 6)
    # initialize output variables
    marker_centers = []
    marker_depths = []
    rotMats = None
    tvecs = None

    depth_filtered = np.copy(depth_img)
    # verify that *at least* one ArUco marker was detected
    if len(corners) > 0:
        cv.aruco.drawDetectedMarkers(aruco_img, corners, ids)
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # sort markers
        ids_sort_index = np.argsort(ids)
        ids = ids[ids_sort_index]
        corners = [corners[i] for i in ids_sort_index]
        # loop over the detected ArUCo corners
        for counter, markerCorner in enumerate(corners):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            iCorners = markerCorner.reshape((4, 2))
            iCornersPix = np.array(iCorners).astype(int)
            marker_centers.append(np.mean(iCornersPix, axis=0))
            # convert each of the (x, y)-coordinate pairs to integers
            iDepths = depth_img[iCornersPix[:, 1], iCornersPix[:, 0]]
            # calculate the nonzero mean (zero depth value is not valid and should be skipped)
            marker_depths.append(iDepths[np.nonzero(iDepths)].mean())

        minY = np.min(np.array(marker_centers)[:, 0]).astype(int)
        maxY = np.max(np.array(marker_centers)[:, 0]).astype(int)
        minDepth = np.min(marker_depths)
        maxDepth = np.max(marker_depths)

        #depth_filtered[:, :minY] = int(maxDepth)
        #depth_filtered[:, maxY:] = int(maxDepth)

        #depth_filtered = np.where(depth_filtered > maxDepth, int(maxDepth), depth_filtered)
        #depth_filtered = np.where(depth_filtered < minDepth, int(maxDepth), depth_filtered)

        rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, markerLength=convMarkerLength,
                                                             cameraMatrix=cameraMatrix, distCoeffs=None)

        rotMats = np.zeros((len(corners), 3, 3))
        # markerLength: size of the marker side in meters or in any other unit
        for i in range(len(rvecs)):
            rvec = rvecs[i]
            tvec = tvecs[i]
            cv.aruco.drawAxis(aruco_img, cameraMatrix, distCoeffs=None, rvec=rvec, tvec=tvec, length=convMarkerLength)
            rotMats[i,:,:], _ = cv.Rodrigues(rvec)
            #print(rotMat)

    return depth_filtered, marker_centers, marker_depths, rotMats, tvecs


def resize_and_show(fx, fy, **kwargs):
    for k in kwargs.keys():
        tmp = cv.resize(kwargs[k], (0,0), fx=fx, fy=fy)
        cv.imshow(k, tmp)


def clip_image_with_min_max(input_img, min_value=None, max_value=None,
                            min_target_value=0, max_target_value=255, as_type='uint8'):
    img = np.copy(input_img)
    if min_value is None:
        min_value = input_img.min()
    if max_value is None:
        max_value = input_img.max()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < min_value:
                img[i, j] = max_target_value
            elif img[i, j] > max_value:
                img[i, j] = max_target_value
            else:
                img[i, j] = np.int(float(max_target_value-min_target_value) /
                                   (max_value - min_value) *
                                   float(img[i, j] - min_value))
    img = img.astype(np.uint8)
    return img


def clip_and_show(gray_image, alpha=0.2, **kwargs):
    clipped_image = cv.convertScaleAbs(gray_image, alpha=alpha, beta=0)
    """ # alternative code:
    for y in range(gray_image.shape[0]):
       for x in range(gray_image.shape[1]):
           new_image[y,x] = np.clip(alpha*gray_image[y,x] + beta, 0, 255) """
    # show the image
    if 'window_name' in kwargs:
        cv.imshow(kwargs['window_name'], clipped_image)
    return clipped_image


def filter_and_show(color_img, depth_img, **kwargs):
    #color_resized = cv.resize(color_image, None, fx=0.5, fy=0.5)
    #depth_resized = cv.resize(depth_image, None, fx=0.5, fy=0.5)
    #images = np.hstack((color_resized, np.dstack((depth_resized, depth_resized, depth_resized))))
    # if 'alpha' in kwargs:
    #gray = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
    #edg = cv.Canny(color_img, 80, 120)
    #cv.imshow('edges', edg)
    # adaptive Gaussian Thresholding
    #adaptive_gaussian = cv.adaptiveThreshold(depth_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    # adaptive Mean Thresholding
    #adaptive_mean = cv.adaptiveThreshold(depth_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    #cv.imshow('adaptive_mean', adaptive_mean)
    #cv.imshow('adaptive_gaussian', adaptive_gaussian)

    #kernel = np.ones((3, 3), np.uint8)
    #opening = cv.morphologyEx(depth_img, cv.MORPH_OPEN, kernel, iterations=2)
    #cv.imshow('opening', opening)
    #closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)
    #cv.imshow('closing', closing)
    # Median Blur
    depth_img_median = cv.medianBlur(depth_img, 3)
    #cv.imshow('median Blur', depth_img_median)
    gray_img_blured = cv.GaussianBlur(depth_img, (3, 3), cv.BORDER_CONSTANT)
    #cv.imshow('g Blur', gray_img_blured)

    # eignet sich sehr gut zur Nois-Reduktion
    #bil = cv.bilateralFilter(depth_img, 3, 15, 15)
    #cv.imshow('bilateral filter', bil)
    #bil_col = cv.bilateralFilter(color_img, 5, 15, 15)
    #cv.imshow('bilateral filter', bil)

    color_img_contrast = cv.convertScaleAbs(color_img, alpha=1.2, beta=40)
    #cv.imshow('color_image_contrast', color_img_contrast)

    depth_img_equalized = cv.equalizeHist(depth_img_median)
    #cv.imshow('Equalized Image', depth_img_equalized)

    if 'hist' in kwargs:
        hist_depth = cv.calcHist([depth_img], [0], None, [100], [0, int(255)])
        hist_depth_filtered = cv.calcHist([depth_img_equalized], [0], None, [100], [0, int(255)])
        plt.subplot(211), plt.plot(hist_depth)
        plt.subplot(212), plt.plot(hist_depth_filtered)
        plt.show()

    return color_img_contrast, depth_img_equalized


def pointcload_with_open3d(bgr, d, marker_centers, marker_depths, rotMats, tvecs,visualize=True, intrinsic=None):
    if not intrinsic:
        intrinsic = o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault

    marker_rgb = np.zeros_like(bgr)
    marker_d = np.zeros_like(d)
    for i in range(len(marker_centers)):
        iMarkerCenterX = int(marker_centers[i][1])
        iMarkerCenterY = int(marker_centers[i][0])
        iMarkerDepth = marker_depths[i]
        marker_rgb[iMarkerCenterX, iMarkerCenterY, :] = (255, 0, 0)
        marker_d[iMarkerCenterX, iMarkerCenterY] = iMarkerDepth

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(bgr),  # cv.cvtColor(bgr, cv.COLOR_BGR2RGB) ???
        o3d.geometry.Image(d))

    rgbd_marker = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(marker_rgb),
            o3d.geometry.Image(marker_d))

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(intrinsic))

    pcd_marker = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_marker,
        o3d.camera.PinholeCameraIntrinsic(intrinsic))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_marker.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame()
    """/ d
      /
     |--------------------->y
     |    p1
     |                   p2
     |
     |                     p4
     |   p3
     x """
    new_coo_points = np.asarray(pcd_marker.points)
    # find the nearest point to the marker 0, 1 and 2
    p0_idx = np.argmin(np.abs(marker_depths[0]/1000+new_coo_points[:, 2]))
    p1_idx = np.argmin(np.abs(marker_depths[1]/1000+new_coo_points[:, 2]))
    p2_idx = np.argmin(np.abs(marker_depths[2]/1000+new_coo_points[:, 2]))
    p0 = new_coo_points[p0_idx, :]
    p1 = new_coo_points[p1_idx, :]
    p2 = new_coo_points[p2_idx, :]

    l01 = p1-p0
    l02 = p2-p0
    x_new = l02 / np.linalg.norm(l02)
    y_new = l01 - x_new*np.dot(x_new, l01)
    y_new = y_new/np.linalg.norm(y_new)
    z_new = np.cross(x_new, y_new)
    o_new = p0
    M = np.array((x_new, y_new, z_new))
    # https://stackoverflow.com/questions/34391968/how-to-find-the-rotation-matrix-between-two-coordinate-systems
    axis_new = copy.deepcopy(axis_pcd).translate(o_new)
    axis_new = axis_new.rotate(M.T)
    axis_new0 = copy.deepcopy(axis_pcd).translate(o_new)
    axis_new0 = axis_new0.rotate(rotMats[0, :, :])
    axis_new1 = copy.deepcopy(axis_pcd).translate(p1)
    axis_new1 = axis_new1.rotate(rotMats[1,:,:])
    axis_new2 = copy.deepcopy(axis_pcd).translate(p2)
    axis_new2 = axis_new2.rotate(rotMats[2, :, :])

    pcd_statistical_filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2)

    # Hidden point removal
    # camera = [0, 0, -1]
    # radius = 1
    # _, pt_map = pcd_filtered.hidden_point_removal(camera, radius)
    # pcd_filtered2 = pcd_filtered.select_by_index(pt_map)

    aabbx_pcd = o3d.geometry.PointCloud.get_axis_aligned_bounding_box(pcd)
    aabbx_marker = o3d.geometry.PointCloud.get_axis_aligned_bounding_box(pcd_marker)
    bbox = o3d.geometry.AxisAlignedBoundingBox((aabbx_marker.min_bound + .01), (aabbx_marker.max_bound - [.01, -1, 0.01]))
    pcd_bbx_filtered = pcd_statistical_filtered.crop(bbox)

    pcd_rot = copy.deepcopy(pcd_bbx_filtered).translate(-o_new)
    pcd_rot = pcd_rot.rotate(np.linalg.inv(M.T), center=(0, 0, 0))

    #bbox2 = o3d.geometry.AxisAlignedBoundingBox((.01, .01, .01), (2, 2, 1))
    #pcd_rot = pcd_rot.crop(bbox2)
    pcd_rot, ind = pcd_rot.remove_statistical_outlier(nb_neighbors=100, std_ratio=2)

    if visualize:
        # o3d.visualization.draw_geometries([pcd, aabbx_pcd])
        o3d.visualization.draw_geometries([pcd_statistical_filtered, aabbx_pcd])
        o3d.visualization.draw_geometries([pcd_statistical_filtered, aabbx_marker, bbox, axis_pcd, pcd_marker,
                                           axis_new0,axis_new1,axis_new2])
        o3d.visualization.draw_geometries([pcd_bbx_filtered, axis_new])
        o3d.visualization.draw_geometries([pcd_rot, axis_pcd])
    return pcd_rot


def apply_grab_cut(img):
    # mask - It is a mask image where we specify which areas are background, foreground or probable
    # background/foreground etc. It is done by the following flags, cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD,
    # or simply pass 0,1,2,3 to image.
    mask = np.zeros(img.shape[:2], np.uint8)
    #rect_bg = (400, 400, 600, 450)
    #cv.rectangle(mask, rect_bg[:2], rect_bg[-2:], (3,3,3), -1)
    #cv.imshow('mask', mask)
    # bdgModel, fgdModel - These are arrays used by the algorithm internally.
    # You just create two np.float64 type zero arrays of size (1,65).
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # rect - It is the coordinates of a rectangle which includes the foreground object in the format (x,y,w,h)
    rect = (150, 150, 460, 330)

    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_fg = img * mask2[:, :, np.newaxis]
    img_with_rect = img.copy()
    cv.rectangle(img_with_rect, rect[:2], rect[-2:], (255, 0, 0), 2)
    #cv.rectangle(img_with_rect, rect_bg[:2], rect_bg[-2:], (0, 0, 255), 2)

    cv.imshow('img_with_rect', img_with_rect)
    cv.imshow('img_fg', img_fg)

    return None


def apply_kmeans(input_features, input_weights, input_k, sort_by_column, **kwargs):
    # K-means clustering segmentation
    # input_features is an array of dimension (imageX, imageY, D)
    # D is the dimension of features
    # reshape and convert to np.float32
    z = input_features.reshape((-1, input_features.shape[2]))
    z = np.float32(z)
    pt = preprocessing.QuantileTransformer(output_distribution='uniform', random_state=0)
    if 'normalize' in kwargs:
        norm = kwargs['normalize']
    else:
        norm = False
    if norm:
        z_norm = pt.fit_transform(z)
    else:
        z_norm = z
    z_weighted = z_norm * input_weights
    # set the criteria to converge
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # use k-means algorithm from openCV
    ret, labels, centers = cv.kmeans(z_weighted, input_k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # make a regular 1D array from the label
    label_flatten = labels.flatten()
    # Now sort the labels
    # Centers is a (k, D) array
    sort_index = centers[:, sort_by_column].argsort()
    centers_sorted = centers[sort_index, :]
    labels_sorted = np.zeros_like(label_flatten)
    for i, value in enumerate(sort_index):
        tmp = (label_flatten == value)
        labels_sorted[tmp] = i
    # Now reshape back to the original image shape
    segmented_gray = labels_sorted.reshape(input_features.shape[0:2])
    return ret, segmented_gray, centers_sorted


def apply_watershed(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # cv.imshow('thresh', thresh)
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
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    cv.imshow('marker', np.float32(markers))

    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv.imshow('markers', np.float32(markers))


if __name__ == "__main__":
    main()
"""
def more_trash_code():
    # apply_grab_cut(color_image)

    # apply_watershed(color_image)

    # prepare data for segmentation
    applySeg = False
    if applySeg:
        x = np.zeros(depth_image.shape).astype(np.float32)
        y = np.zeros(depth_image.shape, dtype=np.float32)
        for i in range(depth_image.shape[0]):
            for j in range(depth_image.shape[1]):
                x[i, j] = i
                y[i, j] = j

        features = np.dstack((np.float32(color_image),
                              np.float32(depth_filtered),
                              x, y))
        k = 8
        weights = np.array([[1, 1, 1, 1, .2, .2]]).astype(np.float32)
        _, segmented, center_sorted = apply_kmeans(features, weights, k, 3, normalize=False)
        res3 = cv.convertScaleAbs(segmented, alpha=255 / segmented.max())
        cv.imshow('res3', res3)

        weights = np.array([[.3, .3, .3, 2, .6, .6]]).astype(np.float32)
        k = 5
        _, segmented2, center_sorted2 = apply_kmeans(features, weights, k, 3, normalize=True)
        res5 = cv.convertScaleAbs(segmented2, alpha=255 / segmented2.max())
        cv.imshow('res5', res5)
        
        
        
def trash_code():
    # threshold filter and normalize depth_image
    # max_distance = 255/0.2
    # max_target_val = 255.0
    # depth_filtered = np.where((depth_image > max_distance), np.uint16(max_distance), np.uint16(depth_image))

    # depth_filtered_scaled = np.uint8(depth_filtered/max_distance*max_target_val)

    # cv.imshow('depth_filtered', depth_filtered/50)
    # cv.imshow('depth_filtered_scaled', depth_filtered_scaled)

    # hist_depth = cv.calcHist([depth_image], [0], None, [200], [0, int(3000)])
    # hist_depth_filtered = cv.calcHist([depth_filtered], [0], None, [200], [0, int(1500)])
    # hist_depth_filtered_scaled = cv.calcHist([depth_filtered_scaled], [0], None, [200], [0, int(255)])

    # plt.subplot(311), plt.plot(hist_depth)
    # plt.subplot(312), plt.plot(hist_depth_filtered)
    # plt.subplot(313), plt.plot(hist_depth_filtered_scaled)
    # plt.show()


    # Z = features.reshape((-1, features.shape[2]))
    # Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 3
    # ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    # depth_column = 3
    # sort_index = center[:, depth_column].argsort()
    # center_sorted = center[:, sort_index]

    # res = label.flatten()

    # label_sorted = np.zeros_like(label)
    # for i, iIndex in enumerate(sort_index):
    #    tmp = (res == iIndex)
    #   label_sorted[tmp] = i
    # center = np.uint8(center[:, (0, 1, 2)])

    # res = center[label.flatten()]
    # res = label_sorted
    return None
"""