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
    img_path=['./desk/data/align_test{}.png'.format(i) for i in range(2,5)]
    depth_path=['./desk/data/align_test_depth{}.png'.format(i) for i in range(2,5)]
    pcd=[o3d.io.read_point_cloud('./desk/pcd/result{}.pcd'.format(i)) for i in range(2,5)]

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

####################################################################################################################################################################

    #Get relative Pose
    relative_pose=[(get_relative_pose(sift_result[i],sift_result[i+1],boundary[i],boundary[i+1],depth[i],depth[i+1],image[i],image[i+1])) for i in range(0,len(image)-1)]
    """
    (relative forward, relative inverse)
    """
    relative_pose_forward=[relative_pose[i][0] for i in range(len(relative_pose))]
    """1->2, 2->3, 3->4"""

    relative_pose_inverse=[relative_pose[i][1] for i in range(len(relative_pose))]
    """1<-2, 2<-3, 3<-4"""

    forward_global_pose = get_global_poses(relative_pose_forward)
    """1->1, 1->2, 1->3, 1->4"""
    inverse_global_pose=get_global_poses(relative_pose_inverse)
    """1<-1, 1<-2, 1<-3, 1<-4"""
    for i in range(len(forward_global_pose)):
        a=np.ones(4)
####################################################################################################################################################################

    #Get 3D coordinates of Points
    coordinate_points=[]
    for point in covisibility_graph:
        for i in range(len(point)):
            if point[i]!=-1.0:
                u,v=sift_result[i][0][int(point[i])].pt
                u=np.float64(u)
                v=np.float64(v)

                # Normalized image plane -> (u, v, 1) * z = zu, zv, z
                z = np.asarray(depth[i], dtype=np.float64)[np.int32(v)][np.int32(u)] / depth_scaling_factor  # in mm distance
                x = (u - img_center_x) * z / focal_length
                y = (v - img_center_y) * z / focal_length
                relative_3d = np.array([x,y,z])

                global_3d=inverse_global_pose[i][:3,:3]@relative_3d+inverse_global_pose[i][:3,3]
                coordinate_points.append(global_3d)

                break
    coordinate_points=np.array(coordinate_points)

#검증끝 이제 문제가 발생하면 아래에서부터 발생하는 것이다.
####################################################################################################################################################################
# Make TXT File for Scipy
    #Make TXT File for Bundle Adjustment
    f=open('problem.txt', 'w')

    # number of Cameras, number of 3D points, number of observation
    data="{}\t{}\t{}\n".format(covisibility_graph.shape[1],len(covisibility_graph),observation)
    f.write(data)

    # Camera Index, Point index, observation image coordinate
    index=0
    for co in covisibility_graph:
        for i in range(covisibility_graph.shape[1]):
            if co[i]==-1:
                continue
            point_2d=sift_result[i][0][int(co[i])].pt
            data="{}\t{}\t{}\t{}\n".format(i,index,point_2d[0],point_2d[1])
            f.write(data)
        index+=1


    # Camera Params -> Rotation Vectors, Translation, Focal length, cx, cy

    for pose in forward_global_pose:
        r=R.from_matrix(pose[:3,:3])
        for vec in r.as_rotvec():
            f.write("{}\n".format(vec))

        t=pose[:3,3]
        for vec in t:
            f.write("{}\n".format(vec))




    # 3D Point
    for point in coordinate_points:
        for p in point:
            f.write("{}\n".format(p))

    f.close()

####################################################################################################################################################################
#Scipy Optimization

    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data("problem.txt")

    # Print information
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 6 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))
    """
    camera_indices = np.empty(n_observations, dtype=int)
    point_indices = np.empty(n_observations, dtype=int)
    points_2d = np.empty((n_observations, 2))
    """

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))

    camera=res.x[:n_cameras*6]

    c1=camera[:6]
    c2 = camera[6:12]
    c3 = camera[12:18]

    R_1=R.from_rotvec(c1[:3])
    t_1=c1[3:6]
    trans1=np.identity(4)
    trans1[:3,:3]=R_1.as_matrix()
    trans1[:3,3]=t_1

    R_2=R.from_rotvec(c2[:3])
    t_2=c2[3:6]
    trans2=np.identity(4)
    trans2[:3,:3]=R_2.as_matrix()
    trans2[:3,3]=t_2

    R_3=R.from_rotvec(c3[:3])
    t_3=c3[3:6]
    trans3=np.identity(4)
    trans3[:3,:3]=R_3.as_matrix()
    trans3[:3,3]=t_3


    result=pcd[0].transform(trans1)+pcd[1].transform(trans2)+pcd[2].transform(trans3)
    o3d.visualization.draw_geometries([result])