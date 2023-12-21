import rospy
import cv2
import torch
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local') # local model
vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10) # local model
turn_happen_flag = False
turn_start_flag = False
turn_end_flag = False
nurse_start_flag = False
direction = 'straight'
turn_counter = 0 

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
def move(direction, target, nurse_state):
    global vel_pub
    vel_msg = Twist()
    if target == 'stop':
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.angular.z = 0
        print("I am stopped")
    elif target == 'cone':
        if direction == 'straight':
            vel_msg.linear.x = 0.2
            vel_msg.linear.y = 0
            vel_msg.angular.z = 0
            print("I am going straight")
        elif direction == 'left':
            vel_msg.linear.x = 0.12
            vel_msg.linear.y = 0
            vel_msg.angular.z = 0.4
            print("I am going left")
        elif direction == 'right':
            vel_msg.linear.x = 0.12
            vel_msg.linear.y = 0
            vel_msg.angular.z = -0.4
            print("I am going right")
    elif target == 'nurse':
        forward_flag = 0 #前后方向判断
        if nurse_state[1] == 'forward':
            forward_flag = 1
        elif nurse_state[1] == 'backword':
            forward_flag = -1
        elif nurse_state[1] == 'stop':
            forward_flag = 0
        if nurse_state[0] == 'straight':
            vel_msg.linear.x = forward_flag * 0.05
            vel_msg.linear.y = 0
            vel_msg.angular.z = 0
            
        elif nurse_state[0] == 'left':
            vel_msg.linear.x = forward_flag * 0.05
            vel_msg.linear.y = 0
            vel_msg.angular.z = 0.2
        elif nurse_state[0] == 'right':
            vel_msg.linear.x = forward_flag * 0.05
            vel_msg.linear.y = 0
            vel_msg.angular.z = -0.2

    vel_pub.publish(vel_msg)
    
    print(f"I have published a velocity message:")
    print(f"linear.x: {vel_msg.linear.x}")
    print(f"linear.y: {vel_msg.linear.y}")
    print(f"angular.z: {vel_msg.angular.z}")

def callback(data):
    global turn_happen_flag, vel_pub, direction, turn_start_flag, turn_end_flag , turn_counter, nurse_start_flag
    target = 'stop'
    follow_nurse = ['stop','stop']
    # Convert the ROS Image message to OpenCV format
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(data, "rgb8")
    height_orgin = img.shape[0]
    width_orgin = img.shape[1]

    results = model(img)  # inference
    cones = results.pandas().xyxy[0]

    # Confidence
    cones = cones[cones['confidence'] > 0.5]
    print(f"Number of detected cones: {len(cones)}")

    if len(cones) > 0:
        target = 'cone'
         # Get the center of detected cones
        for i in range(len(cones)):
            cones['center_x'] = (cones['xmin'] + cones['xmax']) / 2
            cones['center_y'] = (cones['ymin'] + cones['ymax']) / 2
            cones['height'] = cones['ymax'] - cones['ymin']

        if len(cones) > 0 and len(cones) <4 and (not turn_happen_flag):
            # Get the closest cone
            closest_cone = cones[cones['height'] == cones['height'].max()]
            print(f"Closest cone: {closest_cone['center_x'].values[0]}, {closest_cone['center_y'].values[0]}")
            # Get the farthest cone
            farthest_cone = cones[cones['height'] == cones['height'].min()]
            print(f"Farthest cone: {farthest_cone['center_x'].values[0]}, {farthest_cone['center_y'].values[0]}")

            if closest_cone['center_x'].values[0] < img.shape[1] / 2:
                print("Turn right")
                direction = 'right'
                turn_happen_flag = True
                turn_start_flag = True
            else:
                print("Turn left")
                direction = 'left'
                turn_happen_flag = True
                turn_start_flag = True

        # Draw bounding boxes and confidence
        for i in range(len(cones)):
            cv2.putText(img, f"{cones['confidence'][i]:.2f}", (int(cones['xmin'][i]), int(cones['ymin'][i])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            center = (int(cones['center_x'][i]), int(cones['center_y'][i]))
            cv2.circle(img, center, 5, (0, 255, 0), -1)
            cv2.putText(img, f"{center}", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (int(cones['xmin'][i]), int(cones['ymin'][i])), (int(cones['xmax'][i]), int(cones['ymax'][i])), (0, 255, 0), 2)

    # å∂nurse
    else:
        a = 0.05 # 面积阈值权重
        b = 0.05 # 追踪护士距离的阈值
        c = 0.05 # 追踪护士的正负误差
        k = 0.1 * width_orgin # Center of the image
        area_threshold = a * height_orgin * width_orgin
        area_track_threshold = b * height_orgin * width_orgin

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_red = np.array([158, 0, 100])
        upper_red = np.array([182, 255, 190])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        edges = cv2.Canny(mask, 50, 150)

        # 膨胀核的大小，可以根据需要调整
        dilate_kernel_size = 3
        dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)

        # 对图像进行膨胀
        edges = cv2.dilate(edges, dilate_kernel, iterations=1)

        
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        total_area = 0
        if contours:
            # Get the sum of area of all contours
            for contour in contours:
                area = cv2.contourArea(contour)
                total_area += area
            # Get the max area contour
            contour_max = max(contours, key=cv2.contourArea)
            
            print(f"duty ratio: {total_area / (height_orgin * width_orgin)}")
            if total_area >= area_threshold and not nurse_start_flag:
                nurse_start_flag = True
                print("I have found a nurse")

        if nurse_start_flag:
            target = 'nurse'
            # Get the center of detected nurse
            M = cv2.moments(contour_max)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
            # Control the direction of the car
            if  cX <  width_orgin/2 - k:
                follow_nurse[0] = 'left'
            elif cX > width_orgin/2 + k:
                follow_nurse[0] = 'right'
            else:
                follow_nurse[0] = 'straight'

            # Control the speed of the car
            if total_area > area_track_threshold * (1 + c):
                follow_nurse[1] = 'backword'
            elif total_area < area_track_threshold * (1 - c):
                follow_nurse[1] = 'forward'
            else:
                follow_nurse[1] = 'stop'

    # Publish velocity message
    if turn_start_flag:
        move(direction, target, follow_nurse)

        # Find if middle of the image is empty
        # if empty, turn_end_flag = True
        # for i in range(len(cones)):
        #     if (cones['center_x'][i] > (img.shape[1] * 0.43))  and (cones['center_x'][i] < (img.shape[1] * 0.57)):
        #         turn_end_flag = True
        turn_counter += 1
        if turn_counter >=18:
            turn_start_flag = False
            turn_end_flag = False
            direction = 'straight' 
    else:
        move(direction, target, follow_nurse)

    print(f"target: {target}, direction: {direction}, nurse_state: {follow_nurse}")

    cv2.imshow("Detection", img)
    cv2.waitKey(1)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('chatter', Image, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()