#!/usr/bin/env
from std_msgs.msg import String
import sys, os, json, cv2, random, detectron2, rospy, math, copy
import numpy as np
import pyzed.sl as sl
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
setup_logger()

## PARAMETERS

# Detectron2
threshold_detectron2 = 0.2      # This number enable to set the threshold for the detection of an object
model_detectron2 = "COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml"        # The different models in detectron2 have different mask qualities and time of computation

## PROGRAMS

def get_zed_datas():

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD2K
    init.depth_mode = sl.DEPTH_MODE.NEURAL    # NEURAL : End to End Neural disparity estimation, requires AI module or ULTRA
    init.coordinate_units = sl.UNIT.MILLIMETER
    coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)
    
    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution

    camera_model = zed.get_camera_information().camera_model

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    err = zed.grab(runtime)

    if err == sl.ERROR_CODE.SUCCESS :

        # Retrieve the left image, depth image in the half-resolution
        zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, image_size)

        image_ocv = image_zed.get_data()
        point_cloud = point_cloud.get_data()

        image_rgb = convertRGBAToRGB(image_ocv)

    zed.close()

    return image_rgb, point_cloud, camera_model, image_size


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


def generate_box_mask():

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_detectron2))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_detectron2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_detectron2)
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def choose_object_to_grasp(predictor, image_rgb, cfg):

        outputs = predictor(image_rgb)
        v = Visualizer(image_rgb[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('rectangled image',out.get_image()[:, :, ::-1])
        print("\n\nNumber of detected objects:",outputs["instances"].pred_classes[0].item())
        print("\nDetected objects:")
        i = 1
        for data in outputs["instances"].pred_classes:
            num = data.item()
            print(i," - ", MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[num])
            i += 1
        print("Which object do you want to grasp?")

        object_number = input()
        object_number = int(object_number)-1
        return object_number, outputs

def main():

    # Extraction of the depth matrix and the image matrix from the camera
    image_rgb, point_cloud, camera_model, image_size = get_zed_datas()

    # Generate masks using Detectron2
    predictor,cfg = generate_box_mask()

    # Choose the object in the image for whom we want to generate the point cloud
    object_number, outputs = choose_object_to_grasp(predictor, image_rgb, cfg)

    cv2.destroyAllWindows()
