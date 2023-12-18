import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import os
print("Current working directory:", os.getcwd())

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

def load_yolov5_model(weights_path, device='cpu'):
    model = DetectMultiBackend(weights_path, device=device)
    return model

def detect_objects(image, model, device='cpu'):
    stride, names, pt = model.stride, model.names, model.pt
    img = torch.from_numpy(image).to(device)
    img = img.float()  # float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.shape).round()

            # 为每个检测对象添加标签和边框
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # 类别ID
                label = f'{names[c]} {conf:.2f}'
                annotator = Annotator(image, line_width=3, example=str(names))
                annotator.box_label(xyxy, label, color=colors(c, True))
    
    # TODO: 在此处发布检测结果到ROS话题

def callback(data, yolo_model, device):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

    # Detect objects using YOLOv5
    detections = detect_objects(cv_image, yolo_model, device)

    if detections is not None:
        # Process the detections as needed
        for det in detections:
            # Extract information from the detection
            conf, cls, xyxy = det[4], int(det[5]), det[0:4].cpu().numpy()

            # Draw bounding box on the image
            cv2.rectangle(cv_image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow("Image from Chatter", cv_image)
    cv2.waitKey(1)  # Necessary to display the image

def listener(yolo_weights_path='best.pt'):
    # Initialize YOLOv5 model
    device = select_device('cpu')
    yolo_model = load_yolov5_model(weights_path=yolo_weights_path, device=device)

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('chatter', Image, callback, callback_args=(yolo_model, device))
    rospy.spin()

if __name__ == '__main__':
    listener()
