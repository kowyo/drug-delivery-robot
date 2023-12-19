import rospy
import cv2
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')  # local model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def callback(data):
    # Convert the ROS Image message to OpenCV format
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(data, "passthrough")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img)  # inference
    # print(results.pandas().xyxy[0])  # print results
    # print(results.seen)  # print results
    temp = results.pandas().xyxy[0]
    # print(temp['xmin'])
    print(f"number of detected objects: {len(temp)}")
    for i in range(len(temp)):
        print(f"object {i}: {temp['name'][i]}")
        print(f"object {i}: {temp['xmin'][i]}")
        print(f"object {i}: {temp['ymin'][i]}")
        print(f"object {i}: {temp['xmax'][i]}")
        print(f"object {i}: {temp['ymax'][i]}")
    print("Done")
    cv2.imshow("Image window", img)
    cv2.waitKey(1)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('chatter', Image, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()