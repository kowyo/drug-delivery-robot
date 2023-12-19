import rospy
import cv2
import torch
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # local model
vel_pub = rospy.Publisher('vel_msg', Twist, queue_size=10)
# rospy.init_node('vel_publisher', anonymous=True)
turn_flag = False
state_flag = 'stop' #cone,nurse,stop

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
def move(direction):
    global vel_pub
    vel_msg = Twist()
    if direction == 'straight':
        vel_msg.linear.x = 0.2
        vel_msg.angular.z = 0
        print("I am going straight")
    elif direction == 'left':
        vel_msg.linear.x = 0.2
        vel_msg.angular.z = 0.5
        print("I am going left")
        rospy.sleep(2)
    elif direction == 'right':
        vel_msg.linear.x = 0.2
        vel_msg.angular.z = -0.5
        print("I am going right")
        rospy.sleep(2)
    vel_pub.publish(vel_msg)
    print("I have published a velocity message")

def callback(data):
    print("I have received an image")
    global turn_flag, vel_pub
    direction = 'straight'
    # Convert the ROS Image message to OpenCV format
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(data, "rgb8")

    results = model(img)  # inference
    cones = results.pandas().xyxy[0]

    # Confidence
    cones = cones[cones['confidence'] > 0.5]

    print(f"Number of detected cones: {len(cones)}")

    # Get the center of detected cones
    for i in range(len(cones)):
        cones['center_x'] = (cones['xmin'] + cones['xmax']) / 2
        cones['center_y'] = (cones['ymin'] + cones['ymax']) / 2
        cones['height'] = cones['ymax'] - cones['ymin']

    if len(cones) > 0 and len(cones) < 3 and (not turn_flag):
        # Get the closest cone
        closest_cone = cones[cones['height'] == cones['height'].max()]
        print(f"Closest cone: {closest_cone['center_x'].values[0]}, {closest_cone['center_y'].values[0]}")
        # Get the farthest cone
        farthest_cone = cones[cones['height'] == cones['height'].min()]
        print(f"Farthest cone: {farthest_cone['center_x'].values[0]}, {farthest_cone['center_y'].values[0]}")

        if closest_cone['center_x'].values[0] < img.shape[1] / 2:
            print("Turn right")
            direction = 'right'
            turn_flag = True
        else:
            print("Turn left")
            direction = 'left'
            turn_flag = True
    
    if turn_flag and len(cones) == 0:
        turn_flag = False

    # Publish velocity message
    move(direction)
    
    # Draw bounding boxes and confidence
    for i in range(len(cones)):
        cv2.putText(img, f"{cones['confidence'][i]:.2f}", (int(cones['xmin'][i]), int(cones['ymin'][i])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        center = (int(cones['center_x'][i]), int(cones['center_y'][i]))
        cv2.circle(img, center, 5, (0, 255, 0), -1)
        cv2.putText(img, f"{center}", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (int(cones['xmin'][i]), int(cones['ymin'][i])), (int(cones['xmax'][i]), int(cones['ymax'][i])), (0, 255, 0), 2)
    cv2.imshow("Detection", img)
    cv2.waitKey(1)


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('chatter', Image, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()