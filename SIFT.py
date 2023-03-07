import numpy as np
import open3d as o3d
import cv2
from utils import get_boundary
from registration import match_ransac

########################################################################################################################
# Intrinsic parameter
########################################################################################################################
K = np.array(
     [[597.522, 0.0, 312.885],
     [0.0, 597.522, 239.870],
     [0.0, 0.0, 1.0]], dtype=np.float64)

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.intrinsic_matrix = K

########################################################################################################################
# Feature matching using SIFT algorithm
########################################################################################################################
# Find transformation matrix from corresponding points based on SIFT
def SIFT_Transformation(img1, img2, depth_img1, depth_img2, source_pcd, target_pcd, distance_ratio=0.65):

    # Read image from path
    imgL=cv2.imread(img1)
    imgR=cv2.imread(img2)
    depthL=np.array(o3d.io.read_image(depth_img1),np.float32)
    depthR=np.array(o3d.io.read_image(depth_img2),np.float32)

    # Clip depth value
    threshold=1500 #1.5m limit
    left_idx=np.where(depthL>threshold)
    right_idx=np.where(depthR>threshold)
    depthL[left_idx]=threshold
    depthR[right_idx]=threshold

    # Intel RealSense D415
    depth_scaling_factor = 999.99
    focal_length = 597.522 ## mm
    img_center_x = 312.885
    img_center_y = 239.870


    sift=cv2.SIFT_create()

    # Find keypoints and descriptors using SIFT
    kp1,des1=sift.detectAndCompute(imgL,None)
    kp2,des2=sift.detectAndCompute(imgR,None)

    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask=[[0,0] for i in range(len(matches))]
    good_matches=[]
    pts1,pts2,kp1_1,kp2_1=[],[],[],[]
    source_x_min, source_x_max, source_y_min, source_y_max = get_boundary(source_pcd)
    target_x_min, target_x_max, target_y_min, target_y_max = get_boundary(target_pcd)

    for i, (m, n) in enumerate(matches):
        if m.distance < distance_ratio * n.distance: # 0.6 for castard,
            if (kp1[m.queryIdx].pt[0] >= source_x_min and kp1[m.queryIdx].pt[0] <= source_x_max):
                if (kp1[m.queryIdx].pt[1] >= source_y_min and kp1[m.queryIdx].pt[1] <= source_y_max):
                    if (kp2[m.trainIdx].pt[0] >= target_x_min and kp2[m.trainIdx].pt[0] <= target_x_max):
                        if (kp2[m.trainIdx].pt[1] >= target_y_min and kp2[m.trainIdx].pt[1] <= target_y_max):
                            good_matches.append([m])
                            kp1_1.append(kp1[m.queryIdx])
                            kp2_1.append(kp2[m.trainIdx])
                            pts1.append(kp1[m.queryIdx].pt) # Source pcd
                            pts2.append(kp2[m.trainIdx].pt) # Target pcd
                            matchesMask[i] = [1, 0]

    img_matched = cv2.drawMatchesKnn(imgL, kp1, imgR, kp2, good_matches, None, matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0), flags=2)

    cv2.imshow('img_matches',img_matched)
    cv2.waitKey(0)

    # Set array for keypoints
    pts1=np.array(pts1)
    pts2=np.array(pts2)

    # Correspondence set
    matches_index=np.array([])
    for i in range(len(pts1)):
        matches_index=np.append(matches_index,np.array([i,i]))
    matches_index=matches_index.reshape(-1,2)
    correspondence_points = o3d.utility.Vector2iVector(matches_index)

    pts1_3d=[]
    pts2_3d=[]

    for i in range(pts1.shape[0]):
        # Image plane -> 픽셀값
        u = np.float64(pts1[i][0])
        v = np.float64(pts1[i][1])

        # Normalized image plane -> (u, v, 1) * z = zu, zv, z
        z = np.asarray(depthL, dtype=np.float64)[np.int32(v)][np.int32(u)] / depth_scaling_factor # in mm distance
        x = (u - img_center_x) * z / focal_length
        y = (v - img_center_y) * z / focal_length
        pts1_3d = np.append(pts1_3d, np.array([x, y, z], dtype=np.float32))

    for i in range(pts2.shape[0]):
        # Image plane
        u = np.float64(pts2[i][0])
        v = np.float64(pts2[i][1])

        # Normalized image plane
        z = np.asarray(depthR, dtype=np.float64)[np.int32(v)][np.int32(u)] / depth_scaling_factor # in mm distance
        x = (u - img_center_x) * z / focal_length
        y = (v - img_center_y) * z / focal_length
        pts2_3d = np.append(pts2_3d, np.array([x, y, z], dtype=np.float32))

    pts1_3d = pts1_3d.reshape(-1, 3)
    pts2_3d = pts2_3d.reshape(-1, 3)

    # Declare point cloud
    pcd1_3d = o3d.geometry.PointCloud()
    pcd2_3d = o3d.geometry.PointCloud()

    #  pc_points: array(Nx3), each row composed with x, y, z in the 3D coordinate
    #  pc_color: array(Nx3), each row composed with R G,B in the rage of 0 ~ 1
    pc_points1 = np.array(pts1_3d, np.float32)
    pc_points2 = np.array(pts2_3d, np.float32)
    pc_color1 = np.array([], np.float32)
    pc_color2 = np.array([], np.float32)

    for i in range(pts1.shape[0]):
        u = np.int32(pts1[i][0])
        v = np.int32(pts1[i][1])
        # pc_colors
        pc_color1 = np.append(pc_color1, np.array(np.float32(imgL[v][u] / 255)))
        pc_color1 = np.reshape(pc_color1, (-1, 3))

    for i in range(pts2.shape[0]):
        u = np.int32(pts2[i][0])
        v = np.int32(pts2[i][1])
        # , np.array(np.float32(imgR[v][u] / 255)))
        pc_color2 = np.reshape(pc_color2, (-1, 3))

    # add position and color to point cloud
    pcd1_3d.points = o3d.utility.Vector3dVector(pc_points1)
    pcd1_3d.colors = o3d.utility.Vector3dVector(pc_color1)
    pcd2_3d.points = o3d.utility.Vector3dVector(pc_points2)
    pcd2_3d.colors = o3d.utility.Vector3dVector(pc_color2)
    R_t = match_ransac(pts1_3d, pts2_3d, tol=0.1)



    return R_t,pcd1_3d,pcd2_3d,np.array(pts1),np.array(pts2),kp1_1,kp2_1



