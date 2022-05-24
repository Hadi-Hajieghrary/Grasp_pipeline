#!/usr/bin/env python

import pyzed.sl as sl
import numpy as np
from argparse import ArgumentParser
import os
import cv2
import sys
sys.path.append('/home/tristan/AprilTag/scripts')
import apriltag
import roslib
import rospy
import tf
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import math as mt ##
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

## Parameters

width_robot = 0.37
length_robot = 0.68
width_apriltag = 0.16
publish_rate = 2

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
                                           tag_size=0.15,
                                           vizualization=3,
                                           verbose=0, # Set verbose to 3 to display the datas
                                           annotation=True
                                           )

    return result


def compute_robot_pose(result, point_cloud):

    number_of_apriltag = len(result)//4
    # print("Number of apriltag detected :", number_of_apriltag)

    if(number_of_apriltag == 0):
        robot_pose_in_camera_frame = 'NULL'
    
    if(number_of_apriltag >= 1):
        bottom_left = result[0][7][1]
        bottom_right = result[0][7][0]
        upper_left = result[0][7][2]
        upper_right = result[0][7][3]

        bottom_left_3d = point_cloud[int(bottom_left[1])][int(bottom_left[0])]
        bottom_right_3d = point_cloud[int(bottom_right[1])][int(bottom_right[0])]
        upper_left_3d = point_cloud[int(upper_left[1])][int(upper_left[0])]
        upper_right_3d = point_cloud[int(upper_right[1])][int(upper_right[0])]

        dX_x = bottom_right_3d[0] - upper_right_3d[0]
        dY_x = bottom_right_3d[1] - upper_right_3d[1]
        dZ_x = bottom_right_3d[2] - upper_right_3d[2]

        dX_y = upper_right_3d[0] - upper_left_3d[0]
        dY_y = upper_right_3d[1] - upper_left_3d[1]
        dZ_y = upper_right_3d[2] - upper_left_3d[2]

        measured_width = mt.sqrt(dX_x**2+dY_x**2+dZ_x**2)
        measured_length = mt.sqrt(dX_y**2+dY_y**2+dZ_y**2)

        r11 = dX_x/measured_width
        r12 = dY_x/measured_width
        r13 = dZ_x/measured_width

        r21 = dX_y/measured_length
        r22 = dY_y/measured_length
        r23 = dZ_y/measured_length

        r31 = r12*r23 - r13*r22
        r32 = r13*r21 - r11*r23
        r33 = r11*r22 - r12*r21

        qw = mt.sqrt(1+r11+r22+r33)*0.5
        qx = 1/4/qw*(r32-r23)
        qy = 1/4/qw*(r13-r31)
        qz = 1/4/qw*(r21-r12)
        
        qw_ = qw
        qx_ = qx
        qy_ = qy
        qz_ = qz
        upper_right_3d[0], upper_right_3d[1], upper_right_3d[2]
        robot_pose_in_camera_frame_ = [upper_right_3d[0], upper_right_3d[1], upper_right_3d[2], qw_, qx_, qy_, qz_]

        """
        teta = mt.pi/2

        Q = [qw, qx, qy, qz]
        Q1 = [mt.cos(teta/2), mt.sin(teta/2), 0, 0]
        # Q1 = [mt.cos(teta/2), mt.sin(teta/2), mt.sin(teta/2), 0]
        Q2 = Q
        Q1_prod_Q2_first = Q1[0]*Q2[0]-Q1[1]*Q2[1]-Q1[2]*Q2[2]-Q1[3]*Q2[3]
        Q1_prod_Q2_second = Q1[0]*Q2[1]+Q1[1]*Q2[0]+Q1[2]*Q2[3]-Q1[3]*Q2[2]
        Q1_prod_Q2_third = Q1[0]*Q2[2]-Q1[1]*Q2[3]+Q1[2]*Q2[0]+Q1[3]*Q2[1]
        Q1_prod_Q2_fourth = Q1[0]*Q2[3]+Q1[1]*Q2[2]-Q1[2]*Q2[1]+Q1[3]*Q2[1]
        Q1 = [Q1_prod_Q2_first, Q1_prod_Q2_second, Q1_prod_Q2_third, Q1_prod_Q2_fourth]
        Q2 = [mt.cos(teta/2), mt.sin(teta/2), 0, 0]
        # Q2 = [mt.cos(teta/2), mt.sin(teta/2), mt.sin(teta/2), 0]
        Q1_prod_Q2_first = Q1[0]*Q2[0]-Q1[1]*Q2[1]-Q1[2]*Q2[2]-Q1[3]*Q2[3]
        Q1_prod_Q2_second = Q1[0]*Q2[1]+Q1[1]*Q2[0]+Q1[2]*Q2[3]-Q1[3]*Q2[2]
        Q1_prod_Q2_third = Q1[0]*Q2[2]-Q1[1]*Q2[3]+Q1[2]*Q2[0]+Q1[3]*Q2[1]
        Q1_prod_Q2_fourth = Q1[0]*Q2[3]+Q1[1]*Q2[2]-Q1[2]*Q2[1]+Q1[3]*Q2[1]
        qw = Q1_prod_Q2_first
        qx = Q1_prod_Q2_second
        qy = Q1_prod_Q2_third
        qz = Q1_prod_Q2_fourth
        """
        robot_pose_in_camera_frame__ = [upper_right_3d[0], upper_right_3d[1], upper_right_3d[2], qw_, qx_, qy_, qz_]

        print(qw*qw+qx*qx+qy*qy+qz*qz)

        quat = [qx, qy, qz, qw]
        quat_norm = quat / np.linalg.norm(quat)
        qx = quat_norm[0]
        qy = quat_norm[1]
        qz = quat_norm[2]
        qw = quat_norm[3]

        r11 = qw*qw+qx*qx-qy*qy-qz*qz
        r12 = 2*qx*qy-2*qw*qz
        r13 = 2*qx*qz+2*qw*qy
        r21 = 2*qx*qy+2*qw*qz
        r22 = qw*qw-qx*qx+qy*qy-qz*qz
        r23 = 2*qy*qz-2*qw*qx
        r31 = 2*qx*qz-2*qw*qy
        r32 = 2*qy*qz+2*qw*qx
        r33 = qw*qw-qx*qx-qy*qy+qz*qz

        rotation_matrix = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

        offset_x = [width_apriltag-length_robot/2]
        offset_y = [width_robot/2-width_apriltag]
        center_point = np.array([[upper_right_3d[0]], [upper_right_3d[1]], [upper_right_3d[2]]])+np.dot(rotation_matrix, np.array([offset_x, offset_y, [0]]))

        robot_pose_in_camera_frame = [center_point[0], center_point[1], center_point[2], qx, qy, qz, qw] #### Regarder labo ordre (qz, qx, qy, qz)

    return robot_pose_in_camera_frame__


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
    while not rospy.is_shutdown():
        robot_pose_in_camera_frame = get_pose()
        t = rospy.Time.now()

        br.sendTransform((robot_pose_in_camera_frame[0], robot_pose_in_camera_frame[1], robot_pose_in_camera_frame[2]),
                         (robot_pose_in_camera_frame[3], robot_pose_in_camera_frame[4], robot_pose_in_camera_frame[5], robot_pose_in_camera_frame[6]),
                         rospy.Time.now(),
                         "robot_base", 
                         "zed2_left_camera_frame")

        # print("Publish")
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
