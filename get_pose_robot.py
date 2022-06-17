#!/usr/bin/env python

import numpy as np
from argparse import ArgumentParser
import os
import sys
path_to_AprilTag = os.path.dirname(os.path.abspath(__file__))+'/AprilTag/scripts'
sys.path.append(path_to_AprilTag)
import apriltag
import rospy
import tf
import math as mt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

## Parameters

publish_rate = 2

offset_x = 0.108 #These three values are the distance between the upper right corner of the AprilTag with ID 0 and the center of the arm
offset_y = 0.0425
offset_z = -0.003

spacing_between_at_1_4 = 0.20#This value is the distance between the upper right corner of the AprilTag 1 and 4
spacing_between_at_1_3 = 0.235#This value is the distance between the upper right corner of the AprilTag 1 and 4

## Programs

def get_zed_datas():

    try:
        image_rgb, point_cloud = listener_image()
    except rospy.ROSInterruptException:
        pass

    return image_rgb, point_cloud


def listener_image():

    bridge = CvBridge()
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


def apriltag_image(image_rgb):

    parser = ArgumentParser(description='Detect AprilTags from static images.')
    apriltag.add_arguments(parser)
    options = parser.parse_args()

    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())
    #img = cv2.imread(image_rgb)

    #print('Reading {}...\n'.format(os.path.split(image)[1]))

    result, overlay = apriltag.detect_tags(image_rgb,
                                           detector,
                                           camera_params=(1070.38, 1070.57, 1096.84, 640.26), # (fx, fy, cx, cy) These parameters can be found in the file /usr/local/zed/settings/SN28076925.conf
                                           # But since only the coordinate of the corners is used, there is no need to change the above values
                                           tag_size=0.15,
                                           vizualization=3,
                                           verbose=0, # Set verbose to 3 to display the datas from the AprilTag detection
                                           annotation=True
                                           )

    return result


def compute_robot_pose(result, point_cloud):

    number_of_apriltag = len(result)//4
    # print("Number of apriltag detected :", number_of_apriltag)
    global offset_x, offset_y, offset_z
    
    if(number_of_apriltag == 1):
        ID = result[0][1]

        if(ID == 1):
            bottom_left = result[0][7][1]
            bottom_right = result[0][7][0]
            upper_left = result[0][7][2]
            upper_right = result[0][7][3]

            bottom_left_3d = point_cloud[int(bottom_left[1])][int(bottom_left[0])]
            bottom_right_3d = point_cloud[int(bottom_right[1])][int(bottom_right[0])]
            upper_left_3d = point_cloud[int(upper_left[1])][int(upper_left[0])]
            upper_right_3d = point_cloud[int(upper_right[1])][int(upper_right[0])]

            dX_x = upper_right_3d[0] - bottom_right_3d[0]
            dY_x = upper_right_3d[1] - bottom_right_3d[1]
            dZ_x = upper_right_3d[2] - bottom_right_3d[2]

            dX_y = upper_left_3d[0] - upper_right_3d[0]
            dY_y = upper_left_3d[1] - upper_right_3d[1]
            dZ_y = upper_left_3d[2] - upper_right_3d[2]

            measured_width = mt.sqrt(dX_x**2+dY_x**2+dZ_x**2)
            measured_length = mt.sqrt(dX_y**2+dY_y**2+dZ_y**2)

            r11 = dX_x/measured_width
            r12 = dY_x/measured_width
            r13 = dZ_x/measured_width

            r21 = dX_y/measured_length
            r22 = dY_y/measured_length
            r23 = dZ_y/measured_length

            x1 = dX_x/measured_width
            x2 = dY_x/measured_width
            x3 = dZ_x/measured_width

            y1 = dX_y/measured_length
            y2 = dY_y/measured_length
            y3 = dZ_y/measured_length

            r31 = r12*r23 - r13*r22
            r32 = r13*r21 - r11*r23
            r33 = r11*r22 - r12*r21

            z_distance = mt.sqrt(r31**2+r32**2+r33**2)
            z1 = r31/z_distance
            z2 = r32/z_distance
            z3 = r33/z_distance
            
            r11, r21, r31 = x1, x2, x3
            r12, r22, r32 = y1, y2, y3
            r13, r23, r33 = z1, z2, z3
            M = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

            r11, r12, r13 = M[0][0], M[0][1], M[0][2]
            r21, r22, r23 = M[1][0], M[1][1], M[1][2]
            r31, r32, r33 = M[2][0], M[2][1], M[2][2]

            try:
                qw = mt.sqrt(1+r11+r22+r33)*0.5
                qx = 1/4/qw*(r32-r23)
                qy = 1/4/qw*(r13-r31)
                qz = 1/4/qw*(r21-r12)
                
                quat = [qx, qy, qz, qw]
                quat_norm = quat / np.linalg.norm(quat)
                qx = quat_norm[0]
                qy = quat_norm[1]
                qz = quat_norm[2]
                qw = quat_norm[3]

                offset = np.array([offset_x, offset_y, offset_z])

                M2 = np.dot(M, offset)
                center_point = upper_right_3d+np.array([M2[0], M2[1], M2[2]])
                robot_pose_in_camera_frame = [center_point[0], center_point[1], center_point[2], qx, qy, qz, qw]

            except:
                robot_pose_in_camera_frame = 'NULL'
        

        elif(ID == 3):
            bottom_left = result[0][7][1]
            bottom_right = result[0][7][0]
            upper_left = result[0][7][2]
            upper_right = result[0][7][3]

            bottom_left_3d = point_cloud[int(bottom_left[1])][int(bottom_left[0])]
            bottom_right_3d = point_cloud[int(bottom_right[1])][int(bottom_right[0])]
            upper_left_3d = point_cloud[int(upper_left[1])][int(upper_left[0])]
            upper_right_3d = point_cloud[int(upper_right[1])][int(upper_right[0])]

            dX_x = upper_right_3d[0] - bottom_right_3d[0]
            dY_x = upper_right_3d[1] - bottom_right_3d[1]
            dZ_x = upper_right_3d[2] - bottom_right_3d[2]

            dX_y = upper_left_3d[0] - upper_right_3d[0]
            dY_y = upper_left_3d[1] - upper_right_3d[1]
            dZ_y = upper_left_3d[2] - upper_right_3d[2]

            measured_width = mt.sqrt(dX_x**2+dY_x**2+dZ_x**2)
            measured_length = mt.sqrt(dX_y**2+dY_y**2+dZ_y**2)

            r11 = dX_x/measured_width
            r12 = dY_x/measured_width
            r13 = dZ_x/measured_width

            r21 = dX_y/measured_length
            r22 = dY_y/measured_length
            r23 = dZ_y/measured_length

            x1 = dX_x/measured_width
            x2 = dY_x/measured_width
            x3 = dZ_x/measured_width

            y1 = dX_y/measured_length
            y2 = dY_y/measured_length
            y3 = dZ_y/measured_length

            r31 = r12*r23 - r13*r22
            r32 = r13*r21 - r11*r23
            r33 = r11*r22 - r12*r21

            z_distance = mt.sqrt(r31**2+r32**2+r33**2)
            z1 = r31/z_distance
            z2 = r32/z_distance
            z3 = r33/z_distance
            
            r11, r21, r31 = x1, x2, x3
            r12, r22, r32 = y1, y2, y3
            r13, r23, r33 = z1, z2, z3
            M = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

            r11, r12, r13 = M[0][0], M[0][1], M[0][2]
            r21, r22, r23 = M[1][0], M[1][1], M[1][2]
            r31, r32, r33 = M[2][0], M[2][1], M[2][2]

            try:
                qw = mt.sqrt(1+r11+r22+r33)*0.5
                qx = 1/4/qw*(r32-r23)
                qy = 1/4/qw*(r13-r31)
                qz = 1/4/qw*(r21-r12)
                
                quat = [qx, qy, qz, qw]
                quat_norm = quat / np.linalg.norm(quat)
                qx = quat_norm[0]
                qy = quat_norm[1]
                qz = quat_norm[2]
                qw = quat_norm[3]

                offset = np.array([offset_x+spacing_between_at_1_3, offset_y, offset_z])

                M2 = np.dot(M, offset)
                center_point = upper_right_3d+np.array([M2[0], M2[1], M2[2]])
                robot_pose_in_camera_frame = [center_point[0], center_point[1], center_point[2], qx, qy, qz, qw]

            except:
                robot_pose_in_camera_frame = 'NULL'

        
        elif(ID == 4):
            bottom_left = result[0][7][1]
            bottom_right = result[0][7][0]
            upper_left = result[0][7][2]
            upper_right = result[0][7][3]

            bottom_left_3d = point_cloud[int(bottom_left[1])][int(bottom_left[0])]
            bottom_right_3d = point_cloud[int(bottom_right[1])][int(bottom_right[0])]
            upper_left_3d = point_cloud[int(upper_left[1])][int(upper_left[0])]
            upper_right_3d = point_cloud[int(upper_right[1])][int(upper_right[0])]

            dX_x = upper_right_3d[0] - bottom_right_3d[0]
            dY_x = upper_right_3d[1] - bottom_right_3d[1]
            dZ_x = upper_right_3d[2] - bottom_right_3d[2]

            dX_y = upper_left_3d[0] - upper_right_3d[0]
            dY_y = upper_left_3d[1] - upper_right_3d[1]
            dZ_y = upper_left_3d[2] - upper_right_3d[2]

            measured_width = mt.sqrt(dX_x**2+dY_x**2+dZ_x**2)
            measured_length = mt.sqrt(dX_y**2+dY_y**2+dZ_y**2)

            r11 = dX_x/measured_width
            r12 = dY_x/measured_width
            r13 = dZ_x/measured_width

            r21 = dX_y/measured_length
            r22 = dY_y/measured_length
            r23 = dZ_y/measured_length

            x1 = dX_x/measured_width
            x2 = dY_x/measured_width
            x3 = dZ_x/measured_width

            y1 = dX_y/measured_length
            y2 = dY_y/measured_length
            y3 = dZ_y/measured_length

            r31 = r12*r23 - r13*r22
            r32 = r13*r21 - r11*r23
            r33 = r11*r22 - r12*r21

            z_distance = mt.sqrt(r31**2+r32**2+r33**2)
            z1 = r31/z_distance
            z2 = r32/z_distance
            z3 = r33/z_distance
            
            r11, r21, r31 = x1, x2, x3
            r12, r22, r32 = y1, y2, y3
            r13, r23, r33 = z1, z2, z3
            M = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

            r11, r12, r13 = M[0][0], M[0][1], M[0][2]
            r21, r22, r23 = M[1][0], M[1][1], M[1][2]
            r31, r32, r33 = M[2][0], M[2][1], M[2][2]

            try:
                qw = mt.sqrt(1+r11+r22+r33)*0.5
                qx = 1/4/qw*(r32-r23)
                qy = 1/4/qw*(r13-r31)
                qz = 1/4/qw*(r21-r12)
                
                quat = [qx, qy, qz, qw]
                quat_norm = quat / np.linalg.norm(quat)
                qx = quat_norm[0]
                qy = quat_norm[1]
                qz = quat_norm[2]
                qw = quat_norm[3]

                offset = np.array([offset_x, offset_y+spacing_between_at_1_4, offset_z])

                M2 = np.dot(M, offset)
                center_point = upper_right_3d+np.array([M2[0], M2[1], M2[2]])
                robot_pose_in_camera_frame = [center_point[0], center_point[1], center_point[2], qx, qy, qz, qw]

            except:
                robot_pose_in_camera_frame = 'NULL'

        else:
            print("\n\nAprilTag ID1 is hidden\n\n")
            robot_pose_in_camera_frame = 'NULL'
        
    if(number_of_apriltag == 2):

        ID1 = result[0][1]
        ID2 = result[4][1]

        if((ID1 == 1) and (ID2 == 3)):
        
            bottom_left = result[4][7][1]
            bottom_right = result[4][7][0]
            upper_left = result[0][7][2]
            upper_right = result[0][7][3]

            bottom_left_3d = point_cloud[int(bottom_left[1])][int(bottom_left[0])]
            bottom_right_3d = point_cloud[int(bottom_right[1])][int(bottom_right[0])]
            upper_left_3d = point_cloud[int(upper_left[1])][int(upper_left[0])]
            upper_right_3d = point_cloud[int(upper_right[1])][int(upper_right[0])]

            dX_x = upper_right_3d[0] - bottom_right_3d[0]
            dY_x = upper_right_3d[1] - bottom_right_3d[1]
            dZ_x = upper_right_3d[2] - bottom_right_3d[2]

            dX_y = upper_left_3d[0] - upper_right_3d[0]
            dY_y = upper_left_3d[1] - upper_right_3d[1]
            dZ_y = upper_left_3d[2] - upper_right_3d[2]

            measured_width = mt.sqrt(dX_x**2+dY_x**2+dZ_x**2)
            measured_length = mt.sqrt(dX_y**2+dY_y**2+dZ_y**2)

            r11 = dX_x/measured_width
            r12 = dY_x/measured_width
            r13 = dZ_x/measured_width

            r21 = dX_y/measured_length
            r22 = dY_y/measured_length
            r23 = dZ_y/measured_length

            x1 = dX_x/measured_width
            x2 = dY_x/measured_width
            x3 = dZ_x/measured_width

            y1 = dX_y/measured_length
            y2 = dY_y/measured_length
            y3 = dZ_y/measured_length

            r31 = r12*r23 - r13*r22
            r32 = r13*r21 - r11*r23
            r33 = r11*r22 - r12*r21

            z_distance = mt.sqrt(r31**2+r32**2+r33**2)
            z1 = r31/z_distance
            z2 = r32/z_distance
            z3 = r33/z_distance

            r11, r21, r31 = x1, x2, x3
            r12, r22, r32 = y1, y2, y3
            r13, r23, r33 = z1, z2, z3
            M = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]) 

            r11, r12, r13 = M[0][0], M[0][1], M[0][2]
            r21, r22, r23 = M[1][0], M[1][1], M[1][2]
            r31, r32, r33 = M[2][0], M[2][1], M[2][2]

            try:
                qw = mt.sqrt(1+r11+r22+r33)*0.5
                qx = 1/4/qw*(r32-r23)
                qy = 1/4/qw*(r13-r31)
                qz = 1/4/qw*(r21-r12)
     
                quat = [qx, qy, qz, qw]
                quat_norm = quat / np.linalg.norm(quat)
                qx = quat_norm[0]
                qy = quat_norm[1]
                qz = quat_norm[2]
                qw = quat_norm[3]
     
                offset = np.array([offset_x, offset_y, offset_z])

                M2 = np.dot(M, offset)
                center_point = upper_right_3d+np.array([M2[0], M2[1], M2[2]])
                robot_pose_in_camera_frame = [center_point[0], center_point[1], center_point[2], qx, qy, qz, qw]

            except:
                robot_pose_in_camera_frame = 'NULL'
        
        elif((ID1 == 1) and (ID2 == 4)):

            bottom_left = result[0][7][1]
            bottom_right = result[4][7][0]
            upper_left = result[0][7][2]
            upper_right = result[4][7][3]
            center_point = result[0][7][3]

            bottom_left_3d = point_cloud[int(bottom_left[1])][int(bottom_left[0])]
            bottom_right_3d = point_cloud[int(bottom_right[1])][int(bottom_right[0])]
            upper_left_3d = point_cloud[int(upper_left[1])][int(upper_left[0])]
            upper_right_3d = point_cloud[int(upper_right[1])][int(upper_right[0])]
            center_point_3d = point_cloud[int(center_point[1])][int(center_point[0])]

            dX_x = upper_right_3d[0] - bottom_right_3d[0]
            dY_x = upper_right_3d[1] - bottom_right_3d[1]
            dZ_x = upper_right_3d[2] - bottom_right_3d[2]

            dX_y = upper_left_3d[0] - upper_right_3d[0]
            dY_y = upper_left_3d[1] - upper_right_3d[1]
            dZ_y = upper_left_3d[2] - upper_right_3d[2]

            measured_width = mt.sqrt(dX_x**2+dY_x**2+dZ_x**2)
            measured_length = mt.sqrt(dX_y**2+dY_y**2+dZ_y**2)

            r11 = dX_x/measured_width
            r12 = dY_x/measured_width
            r13 = dZ_x/measured_width

            r21 = dX_y/measured_length
            r22 = dY_y/measured_length
            r23 = dZ_y/measured_length

            x1 = dX_x/measured_width
            x2 = dY_x/measured_width
            x3 = dZ_x/measured_width

            y1 = dX_y/measured_length
            y2 = dY_y/measured_length
            y3 = dZ_y/measured_length

            r31 = r12*r23 - r13*r22
            r32 = r13*r21 - r11*r23
            r33 = r11*r22 - r12*r21

            z_distance = mt.sqrt(r31**2+r32**2+r33**2)
            z1 = r31/z_distance
            z2 = r32/z_distance
            z3 = r33/z_distance

            r11, r21, r31 = x1, x2, x3
            r12, r22, r32 = y1, y2, y3
            r13, r23, r33 = z1, z2, z3
            M = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]) 

            r11, r12, r13 = M[0][0], M[0][1], M[0][2]
            r21, r22, r23 = M[1][0], M[1][1], M[1][2]
            r31, r32, r33 = M[2][0], M[2][1], M[2][2]

            try:
                qw = mt.sqrt(1+r11+r22+r33)*0.5
                qx = 1/4/qw*(r32-r23)
                qy = 1/4/qw*(r13-r31)
                qz = 1/4/qw*(r21-r12)
     
                quat = [qx, qy, qz, qw]
                quat_norm = quat / np.linalg.norm(quat)
                qx = quat_norm[0]
                qy = quat_norm[1]
                qz = quat_norm[2]
                qw = quat_norm[3]
     
                offset = np.array([offset_x, offset_y, offset_z])

                M2 = np.dot(M, offset)
                center_point = center_point_3d+np.array([M2[0], M2[1], M2[2]])
                robot_pose_in_camera_frame = [center_point[0], center_point[1], center_point[2], qx, qy, qz, qw]

            except:
                robot_pose_in_camera_frame = 'NULL'
        
        elif((ID1 == 3) and (ID2 == 4)):

            bottom_left = result[0][7][1]
            bottom_right = result[0][7][0]
            upper_left = result[0][7][2]
            upper_right = result[0][7][3]

            bottom_left_3d = point_cloud[int(bottom_left[1])][int(bottom_left[0])]
            bottom_right_3d = point_cloud[int(bottom_right[1])][int(bottom_right[0])]
            upper_left_3d = point_cloud[int(upper_left[1])][int(upper_left[0])]
            upper_right_3d = point_cloud[int(upper_right[1])][int(upper_right[0])]

            dX_x = upper_right_3d[0] - bottom_right_3d[0]
            dY_x = upper_right_3d[1] - bottom_right_3d[1]
            dZ_x = upper_right_3d[2] - bottom_right_3d[2]

            dX_y = upper_left_3d[0] - upper_right_3d[0]
            dY_y = upper_left_3d[1] - upper_right_3d[1]
            dZ_y = upper_left_3d[2] - upper_right_3d[2]

            measured_width = mt.sqrt(dX_x**2+dY_x**2+dZ_x**2)
            measured_length = mt.sqrt(dX_y**2+dY_y**2+dZ_y**2)

            r11 = dX_x/measured_width
            r12 = dY_x/measured_width
            r13 = dZ_x/measured_width

            r21 = dX_y/measured_length
            r22 = dY_y/measured_length
            r23 = dZ_y/measured_length

            x1 = dX_x/measured_width
            x2 = dY_x/measured_width
            x3 = dZ_x/measured_width

            y1 = dX_y/measured_length
            y2 = dY_y/measured_length
            y3 = dZ_y/measured_length

            r31 = r12*r23 - r13*r22
            r32 = r13*r21 - r11*r23
            r33 = r11*r22 - r12*r21

            z_distance = mt.sqrt(r31**2+r32**2+r33**2)
            z1 = r31/z_distance
            z2 = r32/z_distance
            z3 = r33/z_distance

            r11, r21, r31 = x1, x2, x3
            r12, r22, r32 = y1, y2, y3
            r13, r23, r33 = z1, z2, z3
            M = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]) 

            r11, r12, r13 = M[0][0], M[0][1], M[0][2]
            r21, r22, r23 = M[1][0], M[1][1], M[1][2]
            r31, r32, r33 = M[2][0], M[2][1], M[2][2]


            try:
                qw = mt.sqrt(1+r11+r22+r33)*0.5
                qx = 1/4/qw*(r32-r23)
                qy = 1/4/qw*(r13-r31)
                qz = 1/4/qw*(r21-r12)
     
                quat = [qx, qy, qz, qw]
                quat_norm = quat / np.linalg.norm(quat)
                qx = quat_norm[0]
                qy = quat_norm[1]
                qz = quat_norm[2]
                qw = quat_norm[3]
     
                offset = np.array([offset_x+spacing_between_at_1_3, offset_y, offset_z])

                M2 = np.dot(M, offset)
                center_point = upper_right_3d+np.array([M2[0], M2[1], M2[2]])
                robot_pose_in_camera_frame = [center_point[0], center_point[1], center_point[2], qx, qy, qz, qw]

            except:
                robot_pose_in_camera_frame = 'NULL'
        

        else:
            robot_pose_in_camera_frame = 'NULL'
    
    if(number_of_apriltag == 3):

        bottom_left_1 = result[0][7][1]
        bottom_right_1 = result[0][7][0]
        upper_left_1 = result[0][7][2]
        upper_right_1 = result[0][7][3]

        bottom_left_2 = result[4][7][1]
        bottom_right_2 = result[4][7][0]
        upper_left_2 = result[4][7][2]
        upper_right_2 = result[4][7][3]

        bottom_left_3 = result[8][7][1]
        bottom_right_3 = result[8][7][0]
        upper_left_3 = result[8][7][2]
        upper_right_3 = result[8][7][3]

        bottom_left_3d_1 = point_cloud[int(bottom_left_1[1])][int(bottom_left_1[0])]
        bottom_right_3d_1 = point_cloud[int(bottom_right_1[1])][int(bottom_right_1[0])]
        upper_left_3d_1 = point_cloud[int(upper_left_1[1])][int(upper_left_1[0])]
        upper_right_3d_1 = point_cloud[int(upper_right_1[1])][int(upper_right_1[0])]

        bottom_left_3d_2 = point_cloud[int(bottom_left_2[1])][int(bottom_left_2[0])]
        bottom_right_3d_2 = point_cloud[int(bottom_right_2[1])][int(bottom_right_2[0])]
        upper_left_3d_2 = point_cloud[int(upper_left_2[1])][int(upper_left_2[0])]
        upper_right_3d_2 = point_cloud[int(upper_right_2[1])][int(upper_right_2[0])]

        bottom_left_3d_3 = point_cloud[int(bottom_left_3[1])][int(bottom_left_3[0])]
        bottom_right_3d_3 = point_cloud[int(bottom_right_3[1])][int(bottom_right_3[0])]
        upper_left_3d_3 = point_cloud[int(upper_left_3[1])][int(upper_left_3[0])]
        upper_right_3d_3 = point_cloud[int(upper_right_3[1])][int(upper_right_3[0])]

        dX_x = upper_right_3d_1[0] - bottom_right_3d_2[0]
        dY_x = upper_right_3d_1[1] - bottom_right_3d_2[1]
        dZ_x = upper_right_3d_1[2] - bottom_right_3d_2[2]

        dX_y = bottom_left_3d_1[0] - bottom_right_3d_3[0]
        dY_y = bottom_left_3d_1[1] - bottom_right_3d_3[1]
        dZ_y = bottom_left_3d_1[2] - bottom_right_3d_3[2]

        measured_width = mt.sqrt(dX_x**2+dY_x**2+dZ_x**2)
        measured_length = mt.sqrt(dX_y**2+dY_y**2+dZ_y**2)

        r11 = dX_x/measured_width
        r12 = dY_x/measured_width
        r13 = dZ_x/measured_width

        r21 = dX_y/measured_length
        r22 = dY_y/measured_length
        r23 = dZ_y/measured_length

        x1 = dX_x/measured_width
        x2 = dY_x/measured_width
        x3 = dZ_x/measured_width

        y1 = dX_y/measured_length
        y2 = dY_y/measured_length
        y3 = dZ_y/measured_length

        r31 = r12*r23 - r13*r22
        r32 = r13*r21 - r11*r23
        r33 = r11*r22 - r12*r21

        z_distance = mt.sqrt(r31**2+r32**2+r33**2)
        z1 = r31/z_distance
        z2 = r32/z_distance
        z3 = r33/z_distance

        r11, r21, r31 = x1, x2, x3
        r12, r22, r32 = y1, y2, y3
        r13, r23, r33 = z1, z2, z3
        M = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]) 

        r11, r12, r13 = M[0][0], M[0][1], M[0][2]
        r21, r22, r23 = M[1][0], M[1][1], M[1][2]
        r31, r32, r33 = M[2][0], M[2][1], M[2][2]


        try:
            qw = mt.sqrt(1+r11+r22+r33)*0.5
            qx = 1/4/qw*(r32-r23)
            qy = 1/4/qw*(r13-r31)
            qz = 1/4/qw*(r21-r12)
                
            quat = [qx, qy, qz, qw]
            quat_norm = quat / np.linalg.norm(quat)
            qx = quat_norm[0]
            qy = quat_norm[1]
            qz = quat_norm[2]
            qw = quat_norm[3]
    
            offset = np.array([offset_x, offset_y, offset_z])
            
            M2 = np.dot(M, offset)
            center_point = upper_right_3d_1+np.array([M2[0], M2[1], M2[2]])
            robot_pose_in_camera_frame = [center_point[0], center_point[1], center_point[2], qx, qy, qz, qw]

        except:
            robot_pose_in_camera_frame = 'NULL'


    if(number_of_apriltag == 0):
        robot_pose_in_camera_frame = 'NULL'
        print("No AprilTag detected")

    return robot_pose_in_camera_frame


def get_pose():

    robot_pose_in_camera_frame = 'NULL'

    while(robot_pose_in_camera_frame == 'NULL'):
        image_rgb, point_cloud = get_zed_datas()
        result = apriltag_image(image_rgb)
        robot_pose_in_camera_frame = compute_robot_pose(result, point_cloud)

    return robot_pose_in_camera_frame


def main():

    rospy.init_node('get_pose_robot', anonymous=True)
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(publish_rate)

    x, y, z = 0, 0, 0

    while not rospy.is_shutdown():
        robot_pose_in_camera_frame = get_pose()
        t = rospy.Time.now()

        print("delta x :",abs(robot_pose_in_camera_frame[0]-x)*100, "cm, delta y :",abs(robot_pose_in_camera_frame[1]-y)*100, "cm, delta z :",abs(robot_pose_in_camera_frame[2]-z)*100, "cm")

        x, y, z = robot_pose_in_camera_frame[0], robot_pose_in_camera_frame[1], robot_pose_in_camera_frame[2]

        br.sendTransform((robot_pose_in_camera_frame[0], robot_pose_in_camera_frame[1], robot_pose_in_camera_frame[2]),
                         (robot_pose_in_camera_frame[3], robot_pose_in_camera_frame[4], robot_pose_in_camera_frame[5], robot_pose_in_camera_frame[6]),
                         rospy.Time.now(),
                         "base_link",
                         "zed2_left_camera_frame")

        print("Publish")
        rate.sleep()
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
