import copy
from utils import get_boundary
import numpy as np
import open3d as o3d
import cv2
from scipy.optimize import least_squares
from utils import *
import matplotlib as plt
import time
import random
from scipy.spatial.transform import Rotation as R
from SIFT import *

if __name__ == "__main__":

    # Intel RealSense D415
    depth_scaling_factor = 999.99
    focal_length = 597.522  ## mm
    img_center_x = 312.885
    img_center_y = 239.870

    # Set image path
    img_path=['./desk/data/align_test{}.png'.format(i) for i in range(1,5)]
    depth_path=['./desk/data/align_test_depth{}.png'.format(i) for i in range(1,5)]
    pcd=[o3d.io.read_point_cloud('./desk/pcd/result{}.pcd'.format(i)) for i in range(1,5)]

    ########################################################################################################################
    # Feature matching using SIFT algorithm
    ########################################################################################################################
    # Find transformation matrix from corresponding points based on SIFT

    # Read image from path
    image=[cv2.imread(path) for path in img_path]
    depth=[np.array(o3d.io.read_image(path), np.float32) for path in depth_path]

    # Find keypoints and descriptors using SIFT
    sift=cv2.SIFT_create()
    sift_result=[(sift.detectAndCompute(img,None)) for img in image]
    boundary=[(get_boundary(p)) for p in pcd]


####################################################################################################################################################################

    #Get Covisiliby Graph
    covisibility_graph,observation=get_covisibility(sift_result,boundary,image,depth)

    print(covisibility_graph)

    every=[]
    for cov in covisibility_graph:
        Check=True
        for c in cov:
            if c==-1.0:
                Check=False
                break

        if Check==True:
            every.append(cov)

    key1=[]
    key2=[]
    key3=[]
    key4=[]

    for e in every:
        key1.append(sift_result[0][0][int(e[0])])
        key2.append(sift_result[1][0][int(e[1])])
        key3.append(sift_result[2][0][int(e[2])])
        key4.append(sift_result[3][0][int(e[3])])



    # 결과 출력
    image_draw=cv2.drawKeypoints(image[0],key1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_draw1=cv2.drawKeypoints(image[1],key2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_draw2=cv2.drawKeypoints(image[2],key3, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_draw3=cv2.drawKeypoints(image[3],key4, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite('result{}.png'.format(0),image_draw)
    cv2.imwrite('result{}.png'.format(1),image_draw1)
    cv2.imwrite('result{}.png'.format(2),image_draw2)
    cv2.imwrite('result{}.png'.format(3),image_draw3)



####################################################################################################################################################################
