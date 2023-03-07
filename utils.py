import numpy as np
import open3d as o3d
import cv2
import copy
from registration import match_ransac
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_boundary(source_pcd):

    # Intrinsic parameter for Realsense D415
    depth_scaling_factor = 999.99
    focal_length = 597.522
    img_center_x = 312.885
    img_center_y = 239.870

    x_min = np.min(np.asarray(source_pcd.points)[:, 0])
    x_max = np.max(np.asarray(source_pcd.points)[:, 0])
    y_min = np.min(np.asarray(source_pcd.points)[:, 1])
    y_max = np.max(np.asarray(source_pcd.points)[:, 1])

    x_min_idx = np.where(np.asarray(source_pcd.points)[:, 0] == x_min)
    x_max_idx = np.where(np.asarray(source_pcd.points)[:, 0] == x_max)
    y_min_idx = np.where(np.asarray(source_pcd.points)[:, 1] == y_min)
    y_max_idx = np.where(np.asarray(source_pcd.points)[:, 1] == y_max)

    u_min = x_min * focal_length / (np.asarray(source_pcd.points)[x_min_idx][0][2]) + img_center_x
    u_max = x_max * focal_length / (np.asarray(source_pcd.points)[x_max_idx][0][2]) + img_center_x
    v_min = y_min * focal_length / (np.asarray(source_pcd.points)[y_min_idx][0][2]) + img_center_y
    v_max = y_max * focal_length / (np.asarray(source_pcd.points)[y_max_idx][0][2]) + img_center_y

    return u_min, u_max, v_min, v_max

# Intel RealSense D415
depth_scaling_factor = 999.99
focal_length = 597.522 ## mm
img_center_x = 312.885
img_center_y = 239.870

def reproject(points,R_T):
    point_copy=copy.deepcopy(points)
    point_copy=point_copy.transform(R_T)
    point_copy=np.asarray(point_copy.points)

    Z=point_copy[:,2]
    X=point_copy[:,0]
    Y=point_copy[:,1]

    u=focal_length*X+img_center_x
    v=focal_length*Y+img_center_y
    result=np.stack((u,v))

    return result.T


def get_good_features(matches,distance_ratio,kp1,kp2,source_x_min,source_x_max,source_y_min,source_y_max,target_x_min,target_x_max,target_y_min,target_y_max):

     good_matches=[]
     pts1=[]
     pts2=[]
     kp=[]

     # depth map에서 위치의 min, max x, y 찾아서 마스킹해서 outlier 제거
     for i, (m, n) in enumerate(matches):
          if m.distance < distance_ratio * n.distance:  # 0.6 for castard,
               if (kp1[m.queryIdx].pt[0] >= source_x_min and kp1[m.queryIdx].pt[0] <= source_x_max):
                    if (kp1[m.queryIdx].pt[1] >= source_y_min and kp1[m.queryIdx].pt[1] <= source_y_max):
                         if (kp2[m.trainIdx].pt[0] >= target_x_min and kp2[m.trainIdx].pt[0] <= target_x_max):
                              if (kp2[m.trainIdx].pt[1] >= target_y_min and kp2[m.trainIdx].pt[1] <= target_y_max):
                                   good_matches.append([m])
                                   #pts1.append(kp1[m.queryIdx].pt)  # Source pcd
                                   #pts2.append(kp2[m.trainIdx].pt)  # Target pcd
                                   kp.append((m.queryIdx,m.trainIdx))


     return good_matches,kp

def get_covisibility(sift_result,boundary,img,depth):
    covisibility_graph = {}
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    for i in range(1, len(sift_result)):
        for j in range(i - 1, -1, -1):
            matches = bf.knnMatch(sift_result[i][1], sift_result[j][1], k=2)
            for k, (m, n) in enumerate(matches):
                if m.distance < 0.4 * n.distance:
                    if (sift_result[i][0][m.queryIdx].pt[0] >= boundary[i][0] and sift_result[i][0][m.queryIdx].pt[0] <=
                            boundary[i][1]):
                        if (sift_result[i][0][m.queryIdx].pt[1] >= boundary[i][2] and sift_result[i][0][m.queryIdx].pt[
                            1] <= boundary[i][3]):
                            if (sift_result[j][0][m.trainIdx].pt[0] >= boundary[j][0] and
                                    sift_result[j][0][m.trainIdx].pt[0] <= boundary[i][1]):
                                if (sift_result[j][0][m.trainIdx].pt[1] >= boundary[i][2] and
                                        sift_result[j][0][m.trainIdx].pt[1] <= boundary[i][3]):
                                    bool = False
                                    for key, value in covisibility_graph.items():
                                        if (j, m.trainIdx) in value and (i, m.queryIdx) not in value:
                                            if any([i == v[0] for v in value]):
                                                continue
                                            value.append((i, m.queryIdx))
                                            bool = True
                                            break
                                    if bool == False:
                                        covisibility_graph[len(covisibility_graph)] = [(i, m.queryIdx), (j, m.trainIdx)]


    covisibility=np.ones((len(covisibility_graph),len(sift_result)))*-1
    for key,value in covisibility_graph.items():
        for v in value:
            covisibility[key][v[0]]=v[1]

    o=0
    for co in covisibility:
        for c in co:
            if c!=-1:
                o+=1

    return covisibility,o

def get_global_poses(relative_poses):

    global_poses=[]
    base_corrdinate=np.identity(4)
    global_poses.append(base_corrdinate)

    for i in range(len(relative_poses)):
        pre_pose=global_poses[i]
        relative_pose=relative_poses[i]

        Rotation=np.dot(pre_pose[:3,:3],relative_pose[:3,:3])
        translation=relative_pose[:3,3]+np.dot(relative_pose[:3,:3],pre_pose[:3,3])

        new_pose=np.identity(4)
        new_pose[:3,:3]=Rotation
        new_pose[:3,3]=translation
        global_poses.append(new_pose)

    return global_poses


# Get relative pose forward and inverse
def get_relative_pose(sift_result1,sift_result2,boundary1,boundary2,depthL,depthR,imgL,imgR):
    kp1,des1=sift_result1
    kp2,des2=sift_result2
    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    pts1 = []
    pts2 = []
    kp1_1 = []
    kp2_1 = []

    source_x_min, source_x_max, source_y_min, source_y_max = boundary1
    target_x_min, target_x_max, target_y_min, target_y_max = boundary2
    # depth map에서 위치의 min, max x, y 찾아서 마스킹해서 outlier 제거
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.4 * n.distance: # 0.6 for castard,
            if (kp1[m.queryIdx].pt[0] >= source_x_min and kp1[m.queryIdx].pt[0] <= source_x_max):
                if (kp1[m.queryIdx].pt[1] >= source_y_min and kp1[m.queryIdx].pt[1] <= source_y_max):
                    if (kp2[m.trainIdx].pt[0] >= target_x_min and kp2[m.trainIdx].pt[0] <= target_x_max):
                        if (kp2[m.trainIdx].pt[1] >= target_y_min and kp2[m.trainIdx].pt[1] <= target_y_max):
                            good_matches.append([m])
                            pts1.append(kp1[m.queryIdx].pt) # Source pcd
                            pts2.append(kp2[m.trainIdx].pt) # Target pcd
                            kp1_1.append(kp1[m.queryIdx])
                            kp2_1.append(kp2[m.trainIdx])


    # Set array for keypoints
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    # Correspondence set
    matches_index = np.array([])
    for i in range(len(pts1)):
        matches_index = np.append(matches_index, np.array([i, i]))
    matches_index = matches_index.reshape(-1, 2)
    correspondence_points = o3d.utility.Vector2iVector(matches_index)

    pts1_3d = []
    pts2_3d = []

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

    R_t = np.array(match_ransac(pts1_3d, pts2_3d, tol=0.1))
    inv_R_t = np.array(match_ransac(pts2_3d, pts1_3d, tol=0.1))

    return R_t,inv_R_t


def read_bal_data(file_name):
    with open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) *v


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]


    f = camera_params[:, 6]
    cx = camera_params[:, 7]
    cy = camera_params[:, 8]
    points_proj[:,0]=f*points_proj[:,0]+cx
    points_proj[:,1]=f*points_proj[:,1]+cy

    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    residual=(points_proj - points_2d).ravel()




    return residual


def fun2(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    residual = (points_proj - points_2d).ravel()


    return points_proj,points_2d


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A