import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def talker():
    pub = rospy.Publisher('chatter', Image, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    # Open the default camera (usually camera index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        rospy.logerr("Error: Could not open camera.")
        return

    bridge = CvBridge()

    while not rospy.is_shutdown():
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("Error: Could not read frame.")
            break

        # Convert the frame to ROS image message
        image_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        original_image_msg = bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # Publish the ROS image message
        pub.publish(image_msg)
        cv2.imshow("Image from Camera", frame)
        cv2.waitKey(1)  # Necessary to display the image
        rate.sleep()

    # Release the camera
    cap.release()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass