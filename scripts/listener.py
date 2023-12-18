#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def callback(data):
    # Convert the ROS Image message to OpenCV format
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

    # Process the OpenCV image as needed
    # For example, display the image
    cv2.imshow("Image from Chatter", cv_image)
    cv2.waitKey(1)  # Necessary to display the image, because cv2.imshow() is non-blocking, waitKey(1) means wait 1ms

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('chatter', Image, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
