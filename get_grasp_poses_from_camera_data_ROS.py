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
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf

## PARAMETERS

objects_list = ['Bottle', 'Box', 'Milk_carton_Valio_1.5L', 'Milk_carton_Coop_1.5L']
predefined_object = 'NULL' # Set to 'NULL' to work without automation 
path_from_catkin_ws = 'src/pipeline/scripts/'

finger_length = 0.1
finger_width = 0.02
finger_height = 0.05
base_width = 0.02
base_height = 0.05
length_approach = 0.1

display_object = False

# Detectron2 image segmentation
threshold_detectron2 = 0.2    # Set the minimum score to display an object detected by Detectron2
model_detectron2 = "COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml"    # This is the choosed model used by Detectron2. The other models can be found here: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

# Outlier points removal
nb_neighbors = 3 # 5# Which specifies how many neighbors are taken into account in order to calculate the average distance for a given point
std_ratio = 0.005 # Which allows setting the threshold level based on the standard deviation of the average distances across the point cloud. The lower this number the more aggressive the filter will be

# Pose estimation
local_registration = 'Point_to_point_ICP' # 'Robust_ICP' or 'Point_to_plane_ICP' or 'Point_to_point_ICP'
voxel_size_first_step = 0.0001 # 0.01 # Used for the global registration in meter
threshold_RICP_first_step = 0.5 ##0.5
sigma_RICP_first_step = 0.5 # 0.5



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


def initialize_detectron2():

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_detectron2))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_detectron2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_detectron2)
    predictor = DefaultPredictor(cfg)

    return predictor, cfg


def choose_object_to_grasp(predictor, image_rgb, cfg, predefined_object):

        #image_rgb = image_rgb.astype('uint8')
        outputs = predictor(image_rgb)
        v = Visualizer(image_rgb[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        j = 0
        object_number = -1

        if(predefined_object == 'NULL'):
            print("\nDetected objects:")
            i = 1
            for data in outputs["instances"].pred_classes:
                num = data.item()
                print(i," - ", MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[num])
                i += 1
            print("Which object do you want to grasp?")

            cv2.imshow('rectangled image',out.get_image()[:, :, ::-1]) # Display the Masks and class for the detected objects
            cv2.waitKey(300)
            #cv2.displayAllWindows()

            object_number = input()
            object_number = int(object_number)-1

            #cv2.destroyAllWindows('rectangled image')

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
        
        if(object_number == -1):
            print("\n\nThe desired object is not detected\n\n")
        
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

    try:
        file_name = path_from_catkin_ws + 'objects/point_cloud_' + str(objects_list[object_to_grasp]) + '.txt'
        point_cloud_object_exact = np.loadtxt(file_name)
    except:
        try:
            file_name = 'objects/point_cloud_' + str(objects_list[object_to_grasp]) + '.txt'
            point_cloud_object_exact = np.loadtxt(file_name)
        except:
            print("\n\nIssue with the path to the folder \"objects\"")

    # Global registration
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(vector_point_cloud_object)
    source.points = o3d.utility.Vector3dVector(point_cloud_object_exact)

    ################
    threshold_list = [0.05, 0.001, 0.005, 0.001]

    voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025, 0.01, 0.005])

    # List of Convergence-Criteria for Multi-Scale ICP:
    criteria_list = [
        o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.5),
        o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.5),
        o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.05),
        o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.005),
        o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.005)
    ]

    # `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
    max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.2, 0.07, 0.05, 0.04])

    # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    
    i = 0
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_sizes[i], source, target, trans_init)
    result1 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False,
        max_correspondence_distances[i],
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                max_correspondence_distances[i] * 1.5)
        ], criteria_list[i])

    i = 1
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_sizes[i], source, target, trans_init)
    result2 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False,
        max_correspondence_distances[i],
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                max_correspondence_distances[i] * 1.5)
        ], criteria_list[i])

    i = 2
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_sizes[i], source, target, trans_init)
    result3 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False,
        max_correspondence_distances[i],
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                max_correspondence_distances[i] * 1.5)
        ], criteria_list[i])

    i = 3
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_sizes[i], source, target, trans_init)
    result4 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False,
        max_correspondence_distances[i],
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                max_correspondence_distances[i] * 1.5)
        ], criteria_list[i])

    list_rmse = [result1.inlier_rmse, result2.inlier_rmse, result3.inlier_rmse, result4.inlier_rmse]
    list_result = [result1, result2, result3, result4]
    min_index = list_rmse.index(min(list_rmse))
    result = list_result[min_index]
    #draw_registration_result(source, target, result.transformation)
    #print(min(list_rmse))
    #print(min_index)
    #print(list_rmse)

    
    """
    i = 0
    source.estimate_normals()
    target.estimate_normals()
    reg_p2l = o3d.pipelines.registration.registration_icp(
    source, target, threshold_list[i], result.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(0.00000001, 0.00000001, 100000))
    
    draw_registration_result(source, target, reg_p2l.transformation)
    print(reg_p2l.inlier_rmse)

    i = 1
    reg_p2l = o3d.pipelines.registration.registration_icp(
    source, target, threshold_list[i], result.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(0.0000001, 0.0000001, 10000))


    draw_registration_result(source, target, reg_p2l.transformation)
    print(reg_p2l.inlier_rmse)

    print(result.inlier_rmse)
    """
    ##################

    """
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size_first_step, source, target, trans_init)
    result_ransac_1 = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size_first_step)
    draw_registration_result(source_down, target_down, result_ransac_1.transformation)
    """

    # Robust ICP
    if(local_registration == 'Robust_ICP'):
        source_down.estimate_normals()
        target_down.estimate_normals()
        loss = o3d.pipelines.registration.TukeyLoss(k=sigma_RICP_first_step)
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        reg_p2l = o3d.pipelines.registration.registration_icp(source_down, target_down, threshold_RICP_first_step, result_ransac_1.transformation, p2l,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200000))

    if(local_registration == 'Point_to_plane_ICP'):
        print("Apply point-to-plane ICP")
        source.estimate_normals()
        target.estimate_normals()
        reg_p2l = o3d.pipelines.registration.registration_icp(source, target, threshold_RICP_first_step, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    if(local_registration == 'Point_to_point_ICP'):
        print("Apply point-to-point ICP")
        reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold_RICP_first_step, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(0.00000000001, 0.00000000001, 100000))
        reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold_RICP_first_step/2, reg_p2l.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(0.00000000001, 0.00000000001, 100000))
        reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold_RICP_first_step/4, reg_p2l.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(0.00000000001, 0.00000000001, 100000))
        reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold_RICP_first_step/8, reg_p2l.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(0.00000000001, 0.00000000001, 100000))
        reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold_RICP_first_step/16, reg_p2l.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(0.00000000001, 0.00000000001, 100000))
        reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold_RICP_first_step/32, reg_p2l.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(0.00000000001, 0.00000000001, 100000))
        print(reg_p2l.fitness)
        print(reg_p2l.inlier_rmse)
    
    if(local_registration == 'Point_to_point_ICP2'):
        voxel_size = 0.05  # means 5cm for this dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)
        draw_registration_result(source, target, reg_p2l.transformation)

    draw_registration_result(source, target, reg_p2l.transformation) ########################################################

    first_row = np.array(reg_p2l.transformation[0][:])
    second_row = np.array(reg_p2l.transformation[1][:])
    third_row = np.array(reg_p2l.transformation[2][:])

    teta = mt.pi/2 ## teta needs maybe to set to -pi/2
    M = np.array([[first_row[:3]], [second_row[:3]], [third_row[:3]]]) ##
    R = np.array([[mt.cos(teta), 0, mt.sin(teta)], [0, 1, 0], [-mt.sin(teta), 0, mt.cos(teta)]]) ##
    M = np.dot(M,R) ##

    first_row[:3] = M[0][:]
    second_row[:3] = M[1][:]
    third_row[:3] = M[2][:]

    spatial_transformation_matrix = np.matrix([first_row, second_row, third_row])
    
    return spatial_transformation_matrix

##########


##########


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
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.8))
    return result


def generate_grasps_poses(spatial_transformation_matrix, object_to_grasp):

    try:
        grasp_file_name = path_from_catkin_ws + 'objects/grasps_' + str(objects_list[object_to_grasp]) + '.txt'
        grasps_object_objref = np.loadtxt(grasp_file_name)
    except:
        try:
            grasp_file_name = 'objects/grasps_' + str(objects_list[object_to_grasp]) + '.txt'
            grasps_object_objref = np.loadtxt(grasp_file_name)
        except:
            print("\n\nIssue with the path to the folder \"grasps\"")

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

        mat = spatial_transformation_matrix[:3, :3]*mat_obj_objref ##

        pose_object = np.array([spatial_transformation_matrix[0, 3], spatial_transformation_matrix[1, 3], spatial_transformation_matrix[2, 3]])
        pose_grasp_in_regard_to_object = np.array([grasps_object_objref[i // 14][0], grasps_object_objref[i // 14][1], grasps_object_objref[i // 14][2]])
        pose_grasp_in_camera_coordinate = pose_object + np.dot(spatial_transformation_matrix[:3, :3], pose_grasp_in_regard_to_object)

        if (i % 14 == 0):
            vector_grasps_to_publish[i+13] = mat[0, 0]
        if (i % 14 == 1):
            vector_grasps_to_publish[i+13] = mat[0, 1]
        if (i % 14 == 2):
            vector_grasps_to_publish[i+13] = mat[0, 2]
        if (i % 14 == 3):
            vector_grasps_to_publish[i+13] = pose_grasp_in_camera_coordinate[0, 0]
        if (i % 14 == 4):
            vector_grasps_to_publish[i+13] = mat[1, 0]
        if (i % 14 == 5):
            vector_grasps_to_publish[i+13] = mat[1, 1]
        if (i % 14 == 6):
            vector_grasps_to_publish[i+13] = mat[1, 2]
        if (i % 14 == 7):
            vector_grasps_to_publish[i+13] = pose_grasp_in_camera_coordinate[0, 1]
        if (i % 14 == 8):
            vector_grasps_to_publish[i+13] = mat[2, 0]
        if (i % 14 == 9):
            vector_grasps_to_publish[i+13] = mat[2, 1]
        if (i % 14 == 10):
            vector_grasps_to_publish[i+13] = mat[2, 2]
        if (i % 14 == 11):
            vector_grasps_to_publish[i+13] = pose_grasp_in_camera_coordinate[0, 2]
        if (i % 14 == 12):
            vector_grasps_to_publish[i+13] = grasps_object_objref[i // 14, 12]
        if (i % 14 == 13):
            vector_grasps_to_publish[i+13] = grasps_object_objref[i // 14, 13]

    return vector_grasps_to_publish


def generate_marker(vector_grasps_to_publish, grasp_indice, markerArray):

    #print(vector_grasps_to_publish[13+14*grasp_indice:13+14*grasp_indice+12])

    r11 = vector_grasps_to_publish[13+14*grasp_indice+0]
    r12 = vector_grasps_to_publish[13+14*grasp_indice+1]
    r13 = vector_grasps_to_publish[13+14*grasp_indice+2]
    r21 = vector_grasps_to_publish[13+14*grasp_indice+4]
    r22 = vector_grasps_to_publish[13+14*grasp_indice+5]
    r23 = vector_grasps_to_publish[13+14*grasp_indice+6]
    r31 = vector_grasps_to_publish[13+14*grasp_indice+8]
    r32 = vector_grasps_to_publish[13+14*grasp_indice+9]
    r33 = vector_grasps_to_publish[13+14*grasp_indice+10]

    qw = mt.sqrt(1+r11+r22+r33)*0.5
    qx = 1/4/qw*(r32-r23)
    qy = 1/4/qw*(r13-r31)
    qz = 1/4/qw*(r21-r12)

    x_grasp_pose = vector_grasps_to_publish[13+14*grasp_indice+3]
    y_grasp_pose = vector_grasps_to_publish[13+14*grasp_indice+7]
    z_grasp_pose = vector_grasps_to_publish[13+14*grasp_indice+11]

    width_grasp = vector_grasps_to_publish[13+14*grasp_indice+12]
    
    base_marker = Marker()
    left_finger_marker = Marker()
    right_finger_marker = Marker()
    approach_marker = Marker()

    base_marker.header.frame_id = "zed2_left_camera_frame"
    left_finger_marker.header.frame_id = "zed2_left_camera_frame"
    right_finger_marker.header.frame_id = "zed2_left_camera_frame"
    approach_marker.header.frame_id = "zed2_left_camera_frame"

    base_marker.header.stamp = rospy.Time.now()
    left_finger_marker.header.stamp = rospy.Time.now()
    right_finger_marker.header.stamp = rospy.Time.now()
    approach_marker.header.stamp = rospy.Time.now()

    rotation_matrix = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

    ## Base marker
    
    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    base_marker.type = 1
    base_marker.id = grasp_indice*4+2
    base_marker.ns = "hand_base"
    
    # Set the scale of the marker
    base_marker.scale.x = base_width
    base_marker.scale.y = width_grasp+2*finger_width
    base_marker.scale.z = base_height
    
    # Set the color
    base_marker.color.r = 0.0
    base_marker.color.g = 1.0
    base_marker.color.b = 1.0
    base_marker.color.a = 1.0
    
    # Set the pose of the marker
    base_marker.pose.position.x = x_grasp_pose
    base_marker.pose.position.y = y_grasp_pose
    base_marker.pose.position.z = z_grasp_pose
    base_marker.pose.orientation.x = qx
    base_marker.pose.orientation.y = qy
    base_marker.pose.orientation.z = qz
    base_marker.pose.orientation.w = qw
    
    ## Approach marker
    
    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    approach_marker.type = 1
    approach_marker.id = grasp_indice*4+3
    approach_marker.ns = "approach"
    
    # Set the scale of the marker
    approach_marker.scale.x = length_approach
    approach_marker.scale.y = base_height
    approach_marker.scale.z = base_height
    
    # Set the color
    approach_marker.color.r = 0.0
    approach_marker.color.g = 1.0
    approach_marker.color.b = 1.0
    approach_marker.color.a = 1.0
    
    offset_x = [-base_width/2-length_approach/2]
    offset_y = [0]
    array_marker = np.array([[x_grasp_pose], [y_grasp_pose], [z_grasp_pose]])+np.dot(rotation_matrix, np.array([offset_x, offset_y, [0]]))
    
    # Set the pose of the marker
    approach_marker.pose.position.x = array_marker[0]
    approach_marker.pose.position.y = array_marker[1]
    approach_marker.pose.position.z = array_marker[2]
    approach_marker.pose.orientation.x = qx
    approach_marker.pose.orientation.y = qy
    approach_marker.pose.orientation.z = qz
    approach_marker.pose.orientation.w = qw
    
    ## Left finger

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    left_finger_marker.type = 1
    left_finger_marker.id = grasp_indice*4
    left_finger_marker.ns = "left_finger"
    
    # Set the scale of the marker
    left_finger_marker.scale.x = finger_length
    left_finger_marker.scale.y = finger_width
    left_finger_marker.scale.z = finger_height

    # Set the color
    left_finger_marker.color.r = 0.0
    left_finger_marker.color.g = 1.0
    left_finger_marker.color.b = 0.0
    left_finger_marker.color.a = 1.0
    
    # Set the pose of the marker
    offset_x = [base_width/2+finger_length/2]
    offset_y = [width_grasp/2+finger_width/2]
    array_marker = np.array([[x_grasp_pose], [y_grasp_pose], [z_grasp_pose]])+np.dot(rotation_matrix, np.array([offset_x, offset_y, [0]]))

    left_finger_marker.pose.position.x = array_marker[0]
    left_finger_marker.pose.position.y = array_marker[1]
    left_finger_marker.pose.position.z = array_marker[2]
    left_finger_marker.pose.orientation.x = qx
    left_finger_marker.pose.orientation.y = qy
    left_finger_marker.pose.orientation.z = qz
    left_finger_marker.pose.orientation.w = qw

    ## Right finger

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    right_finger_marker.type = 1
    right_finger_marker.id = grasp_indice*4+1
    right_finger_marker.ns = "right_finger"
    
    # Set the scale of the marker
    right_finger_marker.scale.x = finger_length
    right_finger_marker.scale.y = finger_width
    right_finger_marker.scale.z = finger_height

    # Set the color
    right_finger_marker.color.r = 0.0
    right_finger_marker.color.g = 1.0
    right_finger_marker.color.b = 0.0
    right_finger_marker.color.a = 1.0
    
    # Set the pose of the marker
    offset_x = [base_width/2+finger_length/2]
    offset_y = [-width_grasp/2-finger_width/2]
    array_marker = np.array([[x_grasp_pose], [y_grasp_pose], [z_grasp_pose]])+np.dot(rotation_matrix, np.array([offset_x, offset_y, [0]]))

    right_finger_marker.pose.position.x = array_marker[0]
    right_finger_marker.pose.position.y = array_marker[1]
    right_finger_marker.pose.position.z = array_marker[2]
    right_finger_marker.pose.orientation.x = qx
    right_finger_marker.pose.orientation.y = qy
    right_finger_marker.pose.orientation.z = qz
    right_finger_marker.pose.orientation.w = qw

    ## Marker array
    markerArray.markers.append(left_finger_marker)
    markerArray.markers.append(right_finger_marker)
    markerArray.markers.append(base_marker)
    markerArray.markers.append(approach_marker)

    return markerArray


def choose_grasp_pose(vector_grasps_to_publish):
    listener = tf.TransformListener()

    trans = False

    while (not rospy.is_shutdown()) and (trans == False):
        try:
            (trans,rot) = listener.lookupTransform("zed2_left_camera_frame", 'tool0_controller', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    
    number_of_grasps = int((len(vector_grasps_to_publish)-13)/14)
    dist = np.zeros(number_of_grasps)
    for i in range(number_of_grasps):
        ind_x = 13+14*i+3
        x = vector_grasps_to_publish[ind_x]
        ind_y = 13+14*i+7
        y = vector_grasps_to_publish[ind_y]
        ind_z = 13+14*i+11
        z = vector_grasps_to_publish[ind_z]
        dist_i = mt.sqrt((trans[0]-x)**2+(trans[1]-y)**2+(trans[2]-z)**2)
        dist[i] = dist_i

    index_closer_grasp = np.argmin(dist)
    vector_grasps_to_publish[13:13+14] = vector_grasps_to_publish[13+14*index_closer_grasp:13+14*(index_closer_grasp+1)]
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
    
    np.savetxt('point_cloud_box_2.txt', vector_point_cloud_object, fmt='%s') # This line can be used to store the point cloud in a file
    # viz_mayavi(vector_point_cloud_object) # Display the point cloud of the object

    # Remove the outlier points
    # print("Number of point before removing outlier points:",vector_point_cloud_object.shape[0]) ##
    vector_point_cloud_object = remove_outlier_points(vector_point_cloud_object)
    # print("Number of point after removing outlier points:",vector_point_cloud_object.shape[0]) ##

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

    vector_grasp_to_publish = generate_grasps_poses(spatial_transformation_matrix, object_to_grasp)

    vector_grasp_to_publish = choose_grasp_pose(vector_grasp_to_publish)

    markerGrasp = MarkerArray()

    grasp_indice = 0
    markerGrasp = generate_marker(vector_grasp_to_publish, grasp_indice, markerGrasp)
    """
    grasp_indice = 1
    markerArray = generate_marker(vector_grasp_to_publish, grasp_indice, markerArray)
    grasp_indice = 2
    markerArray = generate_marker(vector_grasp_to_publish, grasp_indice, markerArray)
    grasp_indice = 3
    """

    if(display_object == True):
        r11 = vector_grasp_to_publish[1]
        r12 = vector_grasp_to_publish[2]
        r13 = vector_grasp_to_publish[3]
        r21 = vector_grasp_to_publish[5]
        r22 = vector_grasp_to_publish[6]
        r23 = vector_grasp_to_publish[7]
        r31 = vector_grasp_to_publish[9]
        r32 = vector_grasp_to_publish[10]
        r33 = vector_grasp_to_publish[11]

        qw = mt.sqrt(1+r11+r22+r33)*0.5
        qx = 1/4/qw*(r32-r23)
        qy = 1/4/qw*(r13-r31)
        qz = 1/4/qw*(r21-r12)

        x_object = vector_grasp_to_publish[4]
        y_object = vector_grasp_to_publish[8]
        z_object = vector_grasp_to_publish[12]

        object = Marker()
        object.header.frame_id = "zed2_left_camera_frame"
        object.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        object.type = 3
        object.id = (grasp_indice+1)*4+1
        object.ns = "object"

        # Set the scale of the marker
        object.scale.x = 0.035
        object.scale.y = 0.035
        object.scale.z = 0.22

        # Set the color
        object.color.r = 0.0
        object.color.g = 1.0
        object.color.b = 1.0
        object.color.a = 1.0

        # Set the pose of the marker
        object.pose.position.x = x_object
        object.pose.position.y = y_object
        object.pose.position.z = z_object
        object.pose.orientation.x = qx
        object.pose.orientation.y = qy
        object.pose.orientation.z = qz
        object.pose.orientation.w = qw
        markerGrasp.markers.append(object)

    # vector_grasps_to_publish = np.array(vector_grasps_to_publish, dtype=np.float32)

    t5 = datetime.now() ##
    print("\n\nTime for pose detection and change frame :", t5-t4,"\n\n") ##
    print("\n\nTime total :", t5-t1,"\n\n") ##

    # marker_pub = rospy.Publisher("/grasp_pose", MarkerArray, queue_size = 2)
    marker_pub = rospy.Publisher("/detect_grasps/plot_grasps", MarkerArray, queue_size = 2)

    while not rospy.is_shutdown():
        marker_pub.publish(markerGrasp)
        rospy.sleep(1)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
