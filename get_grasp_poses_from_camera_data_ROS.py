#!/usr/bin/env
from std_msgs.msg import String
import sys
import os
import json
import cv2
import random
import rospy
import math as mt
import copy
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo ##
from detectron2.engine import DefaultPredictor ##
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer ##
from detectron2.data import MetadataCatalog, DatasetCatalog ##
import matplotlib.pyplot as plt
setup_logger() ##
import open3d as o3d
from argparse import ArgumentParser
from datetime import datetime ##
# import roslib; roslib.load_manifest('visualization_marker_tutorials')
from visualization_msgs.msg import Marker
#from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

## PARAMETERS

objects_list = ['Bottle', 'Box', 'Milk_carton_Valio_1.5L', 'Milk_carton_Coop_1.5L']

# Detectron2 image segmentation
threshold_detectron2 = 0.2    # Set the minimum score to display an object detected by Detectron2
model_detectron2 = "COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml"    # This is the choosed model used by Detectron2. The other models can be found here: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

# Outlier points removal
nb_neighbors = 5 # Which specifies how many neighbors are taken into account in order to calculate the average distance for a given point
std_ratio = 0.005 # Which allows setting the threshold level based on the standard deviation of the average distances across the point cloud. The lower this number the more aggressive the filter will be

# Pose estimation
voxel_size_first_step = 0.01 # 0.01 # Used for the global registration in meter
# voxel_size_second_step = 0.01
threshold_RICP_first_step = 0.5 ##0.5
sigma_RICP_first_step = 0.01 # 0.5
# threshold_RICP_second_step = 0.2 ##0.5
# sigma_RICP_second_step = 0.8


## PROGRAMS

def get_zed_datas():

    try:
        image_rgb, point_cloud = listener_image()
    except rospy.ROSInterruptException:
        pass

    return image_rgb, point_cloud


def listener_image():

    bridge = CvBridge()
    rospy.init_node('grasp_pipeline', anonymous=True)
    image_topic = "/zed2/zed_node/left/image_rect_color"
    image = rospy.wait_for_message(image_topic, Image)
    image_rgb = bridge.imgmsg_to_cv2(image, "bgr8")
    point_cloud_topic = "/zed2/zed_node/point_cloud/cloud_registered"
    point_cloud = rospy.wait_for_message(point_cloud_topic, PointCloud2)
    i = 0
    point_cloud_matrix = np.zeros(image_rgb.shape)
    for p in pc2.read_points(point_cloud, skip_nans=False):
        x_matrix = i // image_rgb.shape[1]
        y_matrix = i % image_rgb.shape[1]
        point_cloud_matrix[x_matrix][y_matrix] = [p[0], p[1], p[2]]
        i += 1

    return image_rgb, point_cloud_matrix


def convertRGBAToRGB( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )


def initialize_detectron2():

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_detectron2))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_detectron2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_detectron2)
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def choose_object_to_grasp(predictor, image_rgb, cfg, predefined_object):

        outputs = predictor(image_rgb)
        v = Visualizer(image_rgb[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        j = 0
        if(predefined_object == 'NULL'):
            cv2.imshow('rectangled image',out.get_image()[:, :, ::-1]) # Display the Masks and class for the detected objects
            print("\nDetected objects:")
            i = 1
            for data in outputs["instances"].pred_classes:
                num = data.item()
                print(i," - ", MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[num])
                i += 1
            print("Which object do you want to grasp?")

            object_number = input()
            object_number = int(object_number)-1

        elif((predefined_object == 'Bottle') | (predefined_object == 'Milk_carton_Valio_1.5L') | (predefined_object == 'Milk_carton_Coop_1.5L')):
            condition_stop = True
            for data in outputs["instances"].pred_classes:
                num = data.item()
                condition = (MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[num] == 'bottle')
                if(condition & condition_stop):
                    object_number = j
                    condition_stop = False
                j = j+1
        
        elif(predefined_object == 'Box'):
            condition_stop = True
            for data in outputs["instances"].pred_classes:
                num = data.item()
                condition = (MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[num] == 'truck')
                if(condition & condition_stop):
                    object_number = j
                    condition_stop = False
                j = j+1
        
        return object_number, outputs


def get_point_cloud_object(object_number, outputs, point_cloud):

    mask = outputs["instances"].pred_masks.cpu().numpy()[object_number]
    box = outputs["instances"].pred_boxes.tensor.cpu().numpy()[object_number]
    mask_h = int(mt.ceil(box[3] - box[1]))
    mask_w = int(mt.ceil(box[2] - box[0]))

    temp_mask = np.zeros((mask_h, mask_w))
    for x_idx in range(int(box[1]), int(box[3])):
        for y_idx in range(int(box[0]), int(box[2])):
            temp_mask[x_idx - int(box[1])][y_idx - int(box[0])] = mask[x_idx][y_idx]

    vector_point_cloud_object = np.array([])
    first_row_vector = False
    for x_idx in range(int(box[1]), int(box[3])):
        for y_idx in range(int(box[0]), int(box[2])):
            if mask[x_idx][y_idx] != 0:
                if not first_row_vector:
                    new_line = (vector_point_cloud_object, point_cloud[x_idx][y_idx][0:3])
                    vector_point_cloud_object = np.hstack(new_line)
                    first_row_vector = True
                else:
                    new_line = (vector_point_cloud_object, point_cloud[x_idx][y_idx][0:3])
                    vector_point_cloud_object = np.vstack(new_line)

    # plt.imshow(temp_mask, cmap='gray') # Display the black and white mask of the object
    # plt.show()

    return vector_point_cloud_object


def remove_outlier_points(vector_point_cloud_object):
    
    # The datas are processed from a numpy format to a o3d format
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(vector_point_cloud_object)

    # The outlier points are removed
    cl, ind = point_cloud_o3d.remove_statistical_outlier(nb_neighbors, std_ratio)    # There are two functions who returns slightly different results: move_radius_outlier or remove_statistical_outlier
    vector_point_cloud_object = np.asarray(cl.points)
    return vector_point_cloud_object


def pose_estimation(vector_point_cloud_object, object_to_grasp):

    file_name = 'objects/point_cloud_' + str(objects_list[object_to_grasp]) + '.txt'
    point_cloud_object_exact = np.loadtxt(file_name)

    # Global registration
    source = o3d.geometry.PointCloud() ## Inversé
    target = o3d.geometry.PointCloud() ##
    target.points = o3d.utility.Vector3dVector(vector_point_cloud_object)
    source.points = o3d.utility.Vector3dVector(point_cloud_object_exact)

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size_first_step, source, target, trans_init)
    result_ransac_1 = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size_first_step)

    # draw_registration_result(source_down, target_down, np.asarray(result_ransac_1.transformation))

    # source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size_second_step, source, target, result_ransac_1.transformation)
    # result_ransac_2 = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size_second_step)

    # draw_registration_result(source_down, target_down, np.asarray(result_ransac_2.transformation))

    # print("RESULT_RANSAC", result_ransac)
    # print(result_ransac.transformation)
    local_registration = 'Point_to_point_ICP'

    # Robust ICP
    if(local_registration == 'Robust_ICP'):
        source_down.estimate_normals()
        target_down.estimate_normals()
        loss = o3d.pipelines.registration.TukeyLoss(k=sigma_RICP_first_step)
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        reg_p2l = o3d.pipelines.registration.registration_icp(source_down, target_down, threshold_RICP_first_step, result_ransac_1.transformation, p2l)

    if(local_registration == 'Point_to_plane_ICP'):
        print("Apply point-to-plane ICP")
        source.estimate_normals()
        target.estimate_normals()
        reg_p2l = o3d.pipelines.registration.registration_icp(source, target, threshold_RICP_first_step, result_ransac_1.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    if(local_registration == 'Point_to_point_ICP'):
        print("Apply point-to-point ICP")
        reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold_RICP_first_step, result_ransac_1.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    """
    source.estimate_normals()
    target.estimate_normals()
    loss = o3d.pipelines.registration.TukeyLoss(k=sigma_RICP_first_step)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    reg_p2l = o3d.pipelines.registration.registration_icp(source, target, threshold_RICP_first_step, reg_p2l.transformation, p2l)
    """

    # loss = o3d.pipelines.registration.TukeyLoss(k=sigma_RICP_second_step)
    # p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    # reg_p2l = o3d.pipelines.registration.registration_icp(source, target, threshold_RICP_second_step, reg_p2l_1.transformation, p2l)

    #print(reg_p2l)
    # print("Spatial transformation matrix of the object: ")
    # print(reg_p2l.transformation)    # Print the transformation matrix of the object
    # draw_registration_result(source_down, target_down, result_ransac_1.transformation)
    draw_registration_result(source, target, reg_p2l.transformation) ########################################################

    
    first_row = reg_p2l.transformation[0][:]
    second_row = reg_p2l.transformation[1][:]
    third_row = reg_p2l.transformation[2][:]
    spatial_transformation_matrix = np.matrix([first_row, second_row, third_row])
    

    return spatial_transformation_matrix


def draw_registration_result(source, target, transformation):

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):

    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, source, target, trans_init):

    # trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
    #                          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def generate_grasps_poses(spatial_transformation_matrix, object_to_grasp):

    grasp_file_name = 'objects/grasps_' + str(objects_list[object_to_grasp]) + '.txt'
    grasps_object_objref = np.loadtxt(grasp_file_name)

    vector_grasps_to_publish = np.zeros(grasps_object_objref.shape[0]*14+13)
    vector_grasps_to_publish[0] = object_to_grasp
    vector_grasps_to_publish[1:5] = spatial_transformation_matrix[0, :]
    vector_grasps_to_publish[5:9] = spatial_transformation_matrix[1, :]
    vector_grasps_to_publish[9:13] = spatial_transformation_matrix[2, :]

    for i in range(len(vector_grasps_to_publish)-13):
        first_line = [grasps_object_objref[i // 14, 3], grasps_object_objref[i // 14, 4], grasps_object_objref[i // 14, 5]]
        mat_obj_objref = np.array(first_line)
        new_line = (mat_obj_objref, [grasps_object_objref[i // 14, 6], grasps_object_objref[i // 14, 7], grasps_object_objref[i // 14, 8]])
        mat_obj_objref = np.vstack(new_line)
        new_line = (mat_obj_objref, [grasps_object_objref[i // 14, 9], grasps_object_objref[i // 14, 10], grasps_object_objref[i // 14, 11]])
        mat_obj_objref = np.vstack(new_line)
        mat = spatial_transformation_matrix[:3, :3]*mat_obj_objref

        if (i % 14 == 0):
            vector_grasps_to_publish[i+13] = mat[0, 0]
        if (i % 14 == 1):
            vector_grasps_to_publish[i+13] = mat[0, 1]
        if (i % 14 == 2):
            vector_grasps_to_publish[i+13] = mat[0, 2]
        if (i % 14 == 3):
            vector_grasps_to_publish[i+13] = grasps_object_objref[i // 14, 0] + spatial_transformation_matrix[0, 3]
        if (i % 14 == 4):
            vector_grasps_to_publish[i+13] = mat[1, 0]
        if (i % 14 == 5):
            vector_grasps_to_publish[i+13] = mat[1, 1]
        if (i % 14 == 6):
            vector_grasps_to_publish[i+13] = mat[1, 2]
        if (i % 14 == 7):
            vector_grasps_to_publish[i+13] = grasps_object_objref[i // 14, 1] + spatial_transformation_matrix[1, 3]
        if (i % 14 == 8):
            vector_grasps_to_publish[i+13] = mat[2, 0]
        if (i % 14 == 9):
            vector_grasps_to_publish[i+13] = mat[2, 1]
        if (i % 14 == 10):
            vector_grasps_to_publish[i+13] = mat[2, 2]
        if (i % 14 == 11):
            vector_grasps_to_publish[i+13] = grasps_object_objref[i // 14, 2] + spatial_transformation_matrix[2, 3]
        if (i % 14 == 12):
            vector_grasps_to_publish[i+13] = grasps_object_objref[i // 14, 12]
        if (i % 14 == 13):
            vector_grasps_to_publish[i+13] = grasps_object_objref[i // 14, 13]

    return vector_grasps_to_publish


def viz_mayavi(points):

    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

    # Plot using mayavi -Much faster and smoother than matplotlib
    import mayavi.mlab

    col = d

    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mayavi.mlab.points3d(x, y, z,
                         col,          # Values used for Color
                         mode="point",
                         colormap='spectral', # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )
    mayavi.mlab.show()


def main(): 

    predefined_object = 'Bottle' # Set to 'NULL' to work without automation ########################################""

    t1 = datetime.now() ##
    # Extraction of the depth map matrix and the colored image matrix from the ZED2 camera
    image_rgb, point_cloud = get_zed_datas()
    t2 = datetime.now() ##
    print("\n\nTime to open the camera and retrieve the data :", t2-t1,"\n\n") ##

    # Initialize Detectron2
    predictor, cfg = initialize_detectron2()

    # Choose the object in the scene for whom the grasp poses will be generated
    object_number, outputs = choose_object_to_grasp(predictor, image_rgb, cfg, predefined_object)

    t3 = datetime.now() ##
    print("\n\nTime for Detectron2 :", t3-t2,"\n\n") ##

    # Merge the mask of the object and the depth map matrix to obtain the partial point cloud of the object
    vector_point_cloud_object = get_point_cloud_object(object_number, outputs, point_cloud)
    
    # np.savetxt('point_cloud_box_2.txt', vector_point_cloud_object, fmt='%s') # This line can be used to store the point cloud in a file
    # viz_mayavi(vector_point_cloud_object) # Display the point cloud of the object

    # Remove the outlier points
    print("Number of point before removing outlier points:",vector_point_cloud_object.shape[0]) ##
    vector_point_cloud_object = remove_outlier_points(vector_point_cloud_object)
    print("Number of point after removing outlier points:",vector_point_cloud_object.shape[0]) ##

    if(predefined_object == 'NULL'):
        for obj in range(len(objects_list)):
            print(obj+1, '-', objects_list[obj])
        object_to_grasp = int(input())-1

    else:
        for obj in range(len(objects_list)):
            if(predefined_object == objects_list[obj]):
                object_to_grasp = obj
                break

    t4 = datetime.now() ##
    print("\n\nTime for outlier removal and pose detection :", t4-t3,"\n\n") ##
    # viz_mayavi(vector_point_cloud_object)

    # Pose estimation

    spatial_transformation_matrix = pose_estimation(vector_point_cloud_object, object_to_grasp)

    vector_grasps_to_publish = generate_grasps_poses(spatial_transformation_matrix, object_to_grasp)
    
    """
    #rospy.init_node('register', anonymous = True)
    topic = 'grasps_poses_camera_coordinate'

    publisher = rospy.Publisher(topic, Marker, queue_size=10)

    #publisher = rospy.Publisher(topic, MarkerArray, queue_size=10)
    
    while publisher.get_num_connections() == 0: pass
    #markerArray = MarkerArray()
    """
    
    """
    print(spatial_transformation_matrix)

    print("X_object =", spatial_transformation_matrix[0, 3])
    print("Y_object =", spatial_transformation_matrix[1, 3])
    print("Z_object =", spatial_transformation_matrix[2, 3])
    print("X_first_grasp =", vector_grasps_to_publish[3+13])
    print("Y_first_grasp =", vector_grasps_to_publish[7+13])
    print("Z_first_grasp =", vector_grasps_to_publish[11+13])
    """

    print("X_object =", spatial_transformation_matrix[0, 3])
    print("Y_object =", spatial_transformation_matrix[1, 3])
    print("Z_object =", spatial_transformation_matrix[2, 3])

    vector_grasps_to_publish = np.array(vector_grasps_to_publish, dtype=np.float32)

    t5 = datetime.now() ##
    print("\n\nTime for pose detection and change frame :", t5-t4,"\n\n") ##
    print("\n\nTime total :", t5-t1,"\n\n") ##

    r11 = vector_grasps_to_publish[13]
    r12 = vector_grasps_to_publish[13+1]
    r13 = vector_grasps_to_publish[13+2]
    r21 = vector_grasps_to_publish[13+4]
    r22 = vector_grasps_to_publish[13+5]
    r23 = vector_grasps_to_publish[13+6]
    r31 = vector_grasps_to_publish[13+8]
    r32 = vector_grasps_to_publish[13+9]
    r33 = vector_grasps_to_publish[13+10]

    q0 = mt.sqrt(1+r11+r22+r33)*0.5
    q1 = 1/4/q0*(r32-r23)
    q2 = 1/4/q0*(r13-r31)
    q3 = 1/4/q0*(r21-r12)

    marker_pub = rospy.Publisher("/grasp_pose", Marker, queue_size = 2)
    
    marker = Marker()
    
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    
    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    marker.type = 1
    marker.id = 0
    marker.ns = "finger"
    
    # Set the scale of the marker
    marker.scale.x = vector_grasps_to_publish[25]
    marker.scale.y = 0.1
    marker.scale.z = 0.05
    
    # Set the color
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    
    # Set the pose of the marker
    marker.pose.position.x = vector_grasps_to_publish[16]
    marker.pose.position.y = vector_grasps_to_publish[20]
    marker.pose.position.z = vector_grasps_to_publish[24]
    marker.pose.orientation.x = q0
    marker.pose.orientation.y = q1
    marker.pose.orientation.z = q2
    marker.pose.orientation.w = q3
    
    while not rospy.is_shutdown():
        marker_pub.publish(marker)
        rospy.rostime.wallsleep(1.0)

    """
    # Publish grasp poses topic
    pub = rospy.Publisher('grasps_poses_camera_coordinate', numpy_msg(Floats), queue_size=10)
    rospy.init_node('publisher_node', anonymous=True)

    rate = rospy.Rate(1)
    rospy.loginfo("Publisher node started, now publishing")
    """

    """
    while not rospy.is_shutdown():

        marker = Marker()
        marker.header.frame_id = "zed2_left_camera_frame"
        #marker.header.frame_id = "map"
        marker.ns = "finger"
        #marker.id = id
        marker.id = 1
        marker.header.stamp = rospy.Time.now()
        marker.type = marker.CUBE
        #marker.type.
        marker.action = marker.ADD
        marker.pose.position.x = vector_grasps_to_publish[16]
        marker.pose.position.y = vector_grasps_to_publish[20]
        marker.pose.position.z = vector_grasps_to_publish[24]
        marker.lifetime = rospy.Duration(1000)
        marker.pose.orientation.x = q0
        marker.pose.orientation.y = q1
        marker.pose.orientation.z = q2
        marker.pose.orientation.w = q3

        marker.scale.x = 0.5 # forward direction
        marker.scale.y = 0.5 # hand closing direction
        marker.scale.z = 0.5 # hand vertical direction

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        print(marker)

        publisher.publish(markerArray)

        rospy.sleep(1)
        
        rospy.spin()

        # pub.publish(vector_apriltag_to_publish)
        # pub.publish(vector_grasps_to_publish)
        # rate.sleep()
    """

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
