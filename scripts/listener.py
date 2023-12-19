# import rospy
# from sensor_msgs.msg import Image
# import cv2
# from cv_bridge import CvBridge
# import torch
# from ultralytics.utils.plotting import Annotator, colors, save_one_box
# import os
# print("Current working directory:", os.getcwd())

# from models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# from utils.torch_utils import select_device, smart_inference_mode

# yolo_weights_path='best.pt'
# device='cpu'

# def load_yolov5_model(weights_path, device):
#     model = DetectMultiBackend(weights_path, device)
#     return model

# def detect_objects(image, model, device):
#     stride, names, pt = model.stride, model.names, model.pt
#     img = torch.from_numpy(image).to(device)
#     img = img.float()  # float32
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
#     if img.ndimension() == 3:
#       img = img.unsqueeze(0)

#     # # Inference
#     pred = model(img, augment=False , visualize=False)

#     # # Apply NMS
#     # pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

#     # # Process detections
#     # for i, det in enumerate(pred):  # detections per image
#     #     if len(det):
#     #         # Rescale boxes from img_size to im0 size
#     #         det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.shape).round()

#     #         # 为每个检测对象添加标签和边框
#     #         for *xyxy, conf, cls in reversed(det):
#     #             c = int(cls)  # 类别ID
#     #             label = f'{names[c]} {conf:.2f}'
#     #             annotator = Annotator(image, line_width=3, example=str(names))
#     #             annotator.box_label(xyxy, label, color=colors(c, True))
    
#     # TODO: 在此处发布检测结果到ROS话题

# def callback(msg):
#     bridge = CvBridge()
#     cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
#     yolo_model = load_yolov5_model(yolo_weights_path, device)

#     # Detect objects using YOLOv5
#     detections = detect_objects(cv_image, yolo_model, device)

#     # if detections is not None:
#     #     # Process the detections as needed
#     #     for det in detections:
#     #         # Extract information from the detection
#     #         conf, cls, xyxy = det[4], int(det[5]), det[0:4].cpu().numpy()

#     #         # Draw bounding box on the image
#     #         cv2.rectangle(cv_image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

#     # Display the image with bounding boxes
#     cv2.imshow("Image from Chatter", cv_image)
#     cv2.waitKey(1)  # Necessary to display the image

# def listener():
#     # Initialize YOLOv5 model
#     rospy.init_node('listener', anonymous=True)
#     rospy.Subscriber('chatter', Image, callback)
#     rospy.spin()

# if __name__ == '__main__':
#     listener()

# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


flag_turn_flag = 0 #0需要判断转向，1正在判断，-1已无需判断

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    left_counter,right_counter,seen, windows, dt = 0, 0, 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'


                    # 需要打印的变量
                    #number of centre
                    #打印最后的seen
                    # centre
                    centre_x = (xyxy[0] + xyxy[2]) / 2
                    centre_y = (xyxy[1] + xyxy[3]) / 2
                    #height,width
                    height = xyxy[3] - xyxy[1]
                    width = xyxy[2] - xyxy[0]
                    #   判断中心点位置在原图中心线左还是右
                    if centre_x < 0.5:
                         flag = -1
                         left_counter = left_counter + 1
                    else:
                         flag = 1
                         right_counter = right_counter + 1
                    #判断最高值及对应的中心点
                    temp_height = height
                    if temp_height > height:
                        height = temp_height
                        heightmax_centre_x = centre_x
                        heightmax_centre_y = centre_y
                        #判断最高值对应的中心点位置在原图中心线左还是右
                        flag_turn = flag


                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    # cv2.imwrite(save_path, im0)
                    # cv2.imshow("Image from detect.py", im0)
                    # cv2.waitKey(1)  # Necessary to display the image
                    print("Image saved to", save_path)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    
    return pred


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

#opt = Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, data=PosixPath('src/drug_delivery/scripts/data/coco128.yaml'), device='', dnn=False, exist_ok=False, half=False, hide_conf=False, hide_labels=False, imgsz=[640, 640], iou_thres=0.45, line_thickness=3, max_det=1000, name='exp', nosave=False, project=PosixPath('src/drug_delivery/scripts/runs/detect'), save_conf=False, save_crop=False, save_csv=False, save_txt=False, source='test.jpg', update=False, vid_stride=1, view_img=False, visualize=False, weights=PosixPath('src/drug_delivery/scripts/yolov5s.pt'))
    

def main(opt):
    pred = run(**vars(opt))
    return pred

def listener():
    # Initialize YOLOv5 model
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('chatter', Image, callback)
    rospy.spin()


def callback(data):
    opt = parse_opt()
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(data, "bgr8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    opt.source = img
    pred = main(opt)

    # #假设已有需要参数
    # a = 0.2 #偏移权重,需要调整
    # turn_threshold_height = 1 #转向高度阈值,需要调整

    # number_of_centre = 0
    # flag_turn = 0 #判断左右转
    # height = 0 #最高值
    # flag_exec_turn = 0 #执行转向
    # goal_pos = 0 #目标位置

    # while True:
    #     #判断左右转
    #     if number_of_centre <= 3:
    #     # //最近的即最高的
    #     # //找到（最近的）最高的边框中心点，判断在左或在右
    #         if(flag_turn_flag == 0): 
    #             flag_turn_flag = 1 #需要转向
    #         if(flag_turn_flag == 1): 
    #             flag_turn_flag = -1 #无需转向


    #     # //判断执行左右转
    #     if flag_turn_flag ==1 and height >= turn_threshold_height:
    #         flag_exec_turn = 1
    
    
    #     #//直行
    #     if flag == 0:
    #         vel_msg.linear.x = 0.05
    #         vel_msg.linear.y = vel_msg.linear.x * a * goal_pos
    #         vel_msg.angular.z = 1
    #         vel_pub.publish(vel_msg)
        
    #     #执行左转
    #     elif flag_turn == -1 and flag_exec_turn == 1:
    #         vel_msg.linear.x = 0.05
    #         vel_msg.linear.y = 0
    #         vel_msg.angular.z = 1
    #         vel_pub.publish(vel_msg)
    #         #执行1s
    #         flag_turn = 0   #或者放在while函数第一行,如果没运行过去，就添加一个flag_turn_flag
    
    #     #//执行右转
    #     elif flag_turn == 1 and flag_exec_turn == 1:
    #         vel_msg.linear.x = 0.05
    #         vel_msg.linear.y = 0
    #         vel_msg.angular.z = 1
    #         vel_pub.publish(vel_msg)
    #         #//执行1s
    #         flag_turn = 0
    

    #     # 目标范围内桶数量小于1，执行停止


    
if __name__ == '__main__':
    listener()
    # pub = rospy.Publisher('vel_msg' , geometry_msgs , queue_size=10)
    # rospy.init_node('vel_sub', anonymous=True)
    # rate = rospy.Rate(10)
    